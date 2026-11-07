"""Load local configuration without overriding existing environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def api_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        try:
            from streamlit import secrets

            value = str(secrets.get(name, "")).strip()
        except (FileNotFoundError, KeyError):
            pass
    return value


def require_api_key(name: str) -> str:
    value = api_key(name)
    if not value:
        raise ValueError(f"Set {name} in {PROJECT_ROOT / '.env'} or your environment.")
    return value


API_KEY_NAMES = {"gemini": "GOOGLE_API_KEY", "mistral": "MISTRAL_API_KEY"}
MODEL_DEFAULTS = {
    "gemini": ("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    "mistral": ("MISTRAL_MODEL", "mistral-large-latest"),
}


def model_name(provider: str) -> str:
    if provider not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown provider: {provider}")
    variable, default = MODEL_DEFAULTS[provider]
    return os.getenv(variable, "").strip() or default


def provider_api_key(provider: str) -> str:
    return api_key(API_KEY_NAMES[provider])


def _number(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class Settings:
    """Bound configuration so mistaken values cannot trigger unlimited work."""

    candidate_budget: int
    cache_ttl: int
    request_timeout: float
    request_retries: int
    llm_timeout: float
    embedding_model: str
    cache_dir: Path


def settings() -> Settings:
    return Settings(
        candidate_budget=int(_number("CANDIDATE_BUDGET", 50, 5, 100)),
        cache_ttl=int(_number("CATALOGUE_CACHE_TTL", 86400, 60, 604800)),
        request_timeout=_number("CATALOGUE_TIMEOUT", 10, 2, 30),
        request_retries=int(_number("CATALOGUE_RETRIES", 2, 0, 3)),
        llm_timeout=_number("LLM_TIMEOUT", 30, 5, 90),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
        or "all-MiniLM-L6-v2",
        cache_dir=PROJECT_ROOT / ".cache",
    )
