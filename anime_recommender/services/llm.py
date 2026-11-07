"""Small, lazy provider adapters with bounded requests and text-only streaming."""
from dataclasses import dataclass
from functools import lru_cache
from anime_recommender.config import API_KEY_NAMES, model_name, require_api_key, settings


def text_content(value):
    """Extract text without displaying tool calls, reasoning or provider objects."""
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        return text_content(value.content)
    if isinstance(value, list):
        return "".join(block if isinstance(block, str) else block.get("text", "")
                       for block in value if isinstance(block, str) or (
                           isinstance(block, dict) and block.get("type") in ("text", "output_text")
                           and not block.get("thought", False)))
    return ""


@dataclass
class Response:
    content: str


def canonical_messages(messages):
    result = []
    for message in messages:
        if isinstance(message, dict):
            result.append({"role": message["role"], "content": text_content(message["content"])})
        else:
            role = {"human": "user", "ai": "assistant"}.get(message.type, message.type)
            result.append({"role": role, "content": text_content(message.content)})
    return result


class ProviderClient:
    def __init__(self, provider):
        self.provider = provider
        self.model = model_name(provider)
        self.timeout = settings().llm_timeout
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.client = ChatGoogleGenerativeAI(model=self.model,
                google_api_key=require_api_key(API_KEY_NAMES[provider]), timeout=self.timeout,
                max_retries=1, temperature=0.4, max_output_tokens=1200,
                include_thoughts=False)
        elif provider == "mistral":
            from openai import OpenAI
            self.client = OpenAI(api_key=require_api_key(API_KEY_NAMES[provider]),
                base_url="https://api.mistral.ai/v1", timeout=self.timeout, max_retries=1)
        else:
            raise ValueError("Choose gemini or mistral.")

    def invoke(self, messages):
        if self.provider == "gemini":
            response = self.client.invoke(canonical_messages(messages))
            text = text_content(response)
        else:
            response = self.client.chat.completions.create(model=self.model,
                messages=canonical_messages(messages), temperature=0.4, max_tokens=1200)
            text = text_content(response.choices[0].message.content)
        if not text.strip():
            raise RuntimeError("The provider returned no text. Retry or choose another provider.")
        return Response(text)

    def stream(self, messages):
        if self.provider == "gemini":
            for chunk in self.client.stream(canonical_messages(messages), generation_config={'max_output_tokens': 400}):
                text = text_content(chunk)
                if text:
                    yield text
                reason = str(chunk.response_metadata.get('finish_reason') or '')
                if reason and 'STOP' not in reason:
                    raise RuntimeError('The provider interrupted or truncated the response.')
        else:
            stream = self.client.chat.completions.create(model=self.model,
                messages=canonical_messages(messages), temperature=0.4, max_tokens=400, stream=True)
            try:
                for chunk in stream:
                    if chunk.choices:
                        text = text_content(chunk.choices[0].delta.content)
                        if text:
                            yield text
                        if chunk.choices[0].finish_reason not in (None, 'stop'):
                            raise RuntimeError('The provider interrupted or truncated the response.')
            finally:
                stream.close()


class LLMProvider:
    @staticmethod
    @lru_cache(maxsize=4)
    def get_llm(provider="gemini"):
        return ProviderClient(provider)
