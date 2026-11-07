import streamlit as st
from typing import List, Dict, Optional
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

from anime_recommender.services.llm import LLMProvider
from anime_recommender.services.jikan import JikanAPI

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'conversation_round' not in st.session_state:
    st.session_state.conversation_round = 0
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'emotions': [],
        'genres': [],
        'themes': [],
        'preferences': {},
        'description': ''
    }
if 'chat_complete' not in st.session_state:
    st.session_state.chat_complete = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = 'gemini'  # Default provider

# ============ MULTI-LLM SETUP ============
# Supports both Mistral and Gemini APIs

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field

class UserAnalysis(BaseModel):
    """Structured output from LLM analysis"""
    emotions: List[str] = Field(description="List of detected emotions (e.g., stressed, happy, melancholic)")
    genres: List[str] = Field(description="Preferred anime genres (e.g., slice of life, action, romance)")
    themes: List[str] = Field(description="Themes that resonate (e.g., friendship, growth, adventure)")
    mood_level: str = Field(description="Overall mood: uplifting_needed, comfortable, excited, thoughtful")
    description: str = Field(description="Brief description of ideal anime for this user")
    confidence: float = Field(description="Confidence in analysis (0-1)")

# ============ LLM PROVIDER ABSTRACTION ============

@st.cache_resource
def init_llm(provider: str = 'gemini'):
    """Initialize LLM with caching"""
    return LLMProvider.get_llm(provider)

# ============ JIKAN API INTEGRATION ============

# ============ NLP EMBEDDING MODEL ============
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer for semantic similarity"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def generate_user_embedding(user_profile: Dict, model: SentenceTransformer) -> np.ndarray:
    """Generate embedding from user profile"""
    text = f"""
    Emotions: {', '.join(user_profile['emotions'])}
    Genres: {', '.join(user_profile['genres'])}
    Themes: {', '.join(user_profile['themes'])}
    Description: {user_profile['description']}
    """
    return model.encode(text)

def generate_anime_embedding(anime: Dict, model: SentenceTransformer) -> np.ndarray:
    """Generate embedding from anime data"""
    genres = ', '.join([g['name'] for g in anime.get('genres', [])])
    themes = ', '.join([t['name'] for t in anime.get('themes', [])])
    synopsis = anime.get('synopsis', '')
    
    text = f"Genres: {genres}. Themes: {themes}. Synopsis: {synopsis}"
    return model.encode(text)

# ============ CONVERSATIONAL AGENT ============
def get_conversation_prompt(round_num: int) -> str:
    """Generate contextual prompts for each conversation round"""
    prompts = {
        1: """You are an empathetic anime expert. Start a warm conversation by asking the user how they're feeling today and what kind of mood they're in. Make them feel comfortable sharing their emotional state. Keep it casual and friendly.""",
        
        2: """Based on their emotional state, ask about their recent experiences with anime or shows. What did they enjoy? What resonated with them emotionally? Dig deeper into their preferences.""",
        
        3: """Explore their genre preferences. Do they prefer action-packed shows or slower, thoughtful ones? Adventure or slice-of-life? Try to understand their taste.""",
        
        4: """Ask about themes that appeal to them. Do they like stories about friendship, personal growth, overcoming challenges, romance, mystery? What messages resonate with them?""",
        
        5: """Summarize what you've learned and ask one final clarifying question about what they hope to feel after watching the anime you'll recommend.""",
        
        6: """Thank them for sharing. Confirm your understanding of their emotional needs and preferences. Let them know you're preparing perfect recommendations."""
    }
    return prompts.get(round_num, prompts[6])

def chat_with_llm(user_message: str, round_num: int, provider: str = 'gemini') -> str:
    """Conduct conversation using selected LLM provider"""
    try:
        llm = init_llm(provider)
        
        # Build conversation history
        system_prompt = f"""You are an emotionally intelligent anime recommender assistant. 
        {get_conversation_prompt(round_num)}
        
        Analyze the user's emotions, preferences, and needs carefully. Be warm, understanding, and engaging.
        Keep responses concise (2-3 sentences max) and conversational."""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for entry in st.session_state.conversation_history:
            if entry['role'] == 'user':
                messages.append(HumanMessage(content=entry['content']))
            else:
                messages.append(AIMessage(content=entry['content']))
        
        # The UI normally stores the current message before this call.
        history = st.session_state.conversation_history
        if not history or history[-1] != {'role': 'user', 'content': user_message}:
            messages.append(HumanMessage(content=user_message))
        
        # Get response
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        return "I'm having trouble connecting right now. Could you tell me more about what kind of anime you enjoy?"

def analyze_conversation(provider: str = 'gemini') -> UserAnalysis:
    """Analyze complete conversation to extract user profile"""
    try:
        llm = init_llm(provider)
        
        # Compile conversation
        conversation_text = "\n".join([
            f"{entry['role']}: {entry['content']}" 
            for entry in st.session_state.conversation_history
        ])
        
        analysis_prompt = f"""Analyze this conversation and extract the following information in JSON format:

Conversation:
{conversation_text}

Provide your analysis as a JSON object with these exact fields:
- emotions: list of emotional states (e.g., ["stressed", "seeking comfort"])
- genres: list of anime genres they'd enjoy (e.g., ["slice of life", "comedy"])
- themes: list of themes that would resonate (e.g., ["friendship", "personal growth"])
- mood_level: one of "uplifting_needed", "comfortable", "excited", or "thoughtful"
- description: brief description of ideal anime (1-2 sentences)
- confidence: confidence score between 0 and 1

Return ONLY the JSON object, no other text."""
        
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # Parse JSON response
        try:
            # Clean response - remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            content = content.strip()
            
            data = json.loads(content)
            analysis = UserAnalysis(**data)
            return analysis
        except json.JSONDecodeError:
            # Fallback parsing
            st.warning("Using fallback analysis parsing")
            return UserAnalysis(
                emotions=['curious', 'seeking entertainment'],
                genres=['action', 'adventure', 'comedy'],
                themes=['friendship', 'growth'],
                mood_level='comfortable',
                description='Engaging anime with good story and characters',
                confidence=0.6
            )
        
    except Exception as e:
        st.error(f"Analysis error: {e}")
        # Return default analysis
        return UserAnalysis(
            emotions=['curious'],
            genres=['action', 'adventure'],
            themes=['friendship'],
            mood_level='comfortable',
            description='Entertaining anime with engaging story',
            confidence=0.5
        )

# ============ RECOMMENDATION ENGINE ============
def generate_recommendations(user_profile: Dict, provider: str = 'gemini') -> List[Dict]:
    """Generate top 5 anime recommendations using NLP embeddings"""
    
    # Load embedding model
    model = load_embedding_model()
    
    # Generate user embedding
    user_embedding = generate_user_embedding(user_profile, model)
    
    # Fetch anime from Jikan API
    with st.spinner("🔍 Searching anime database..."):
        anime_list = JikanAPI.search_anime_by_genres(
            user_profile['genres'], 
            limit=25
        )
    
    if not anime_list:
        st.error("No anime found. Try different preferences.")
        return []
    
    # Generate embeddings and calculate similarities
    similarities = []
    
    with st.spinner("🧠 Analyzing matches using NLP embeddings..."):
        for anime in anime_list:
            try:
                anime_embedding = generate_anime_embedding(anime, model)
                similarity = cosine_similarity(
                    user_embedding.reshape(1, -1),
                    anime_embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append({
                    'anime': anime,
                    'similarity': similarity
                })
            except Exception as e:
                continue
    
    # Sort by similarity and get top 5
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    top_5 = similarities[:5]
    
    # Generate personalized reasons
    recommendations = []
    for item in top_5:
        anime = item['anime']
        reason = generate_recommendation_reason(anime, user_profile, provider)
        
        recommendations.append({
            'title_english': anime.get('title_english') or anime.get('title'),
            'title_japanese': anime.get('title_japanese', ''),
            'synopsis': anime.get('synopsis') or 'No description available.',
            'score': anime.get('score', 'N/A'),
            'episodes': anime.get('episodes', 'Unknown'),
            'genres': [g['name'] for g in anime.get('genres', [])],
            'themes': [t['name'] for t in anime.get('themes', [])],
            'image_url': anime.get('images', {}).get('jpg', {}).get('large_image_url', ''),
            'reason': reason,
            'similarity': item['similarity']
        })
    
    return recommendations

def generate_recommendation_reason(anime: Dict, user_profile: Dict, provider: str = 'gemini') -> str:
    """Generate personalized reason using LLM"""
    genres = ', '.join([g['name'] for g in anime.get('genres', [])])
    try:
        llm = init_llm(provider)
        themes = ', '.join([t['name'] for t in anime.get('themes', [])])
        synopsis = (anime.get('synopsis') or '')[:200]
        
        prompt = f"""User Profile:
- Emotions: {', '.join(user_profile['emotions'])}
- Seeking: {user_profile['description']}

Anime:
- Title: {anime.get('title')}
- Genres: {genres}
- Themes: {themes}
- Synopsis: {synopsis}

Write ONE compelling sentence (max 25 words) explaining why this user should watch this anime based on their emotional needs. Be specific and empathetic."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        reason = response.content.strip()
        
        # Ensure it's not too long
        if len(reason.split()) > 30:
            reason = ' '.join(reason.split()[:30]) + '...'
        
        return reason
        
    except Exception as e:
        return f"A {genres or 'recommended'} anime selected from the available catalogue metadata."

# ============ STREAMLIT UI ============
def apply_custom_css():
    """Apply aesthetic styling"""
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
        }
        .anime-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: white;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .stSelectbox {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Emotion-Based Anime Recommender",
        page_icon="🎭",
        layout="wide"
    )
    
    apply_custom_css()
    
    # Header
    st.title("🎭 Emotion-Based Anime Recommender")
    st.markdown("### *Discover anime that matches your mood and soul*")
    
    # LLM Provider Selection (in sidebar)
    with st.sidebar:
        st.header("⚙️ Settings")
        provider = st.selectbox(
            "Choose AI Provider",
            options=['gemini', 'mistral'],
            format_func=lambda x: "🤖 Google Gemini" if x == 'gemini' else "🔮 Mistral AI",
            key='provider_select'
        )
        st.session_state.llm_provider = provider
        
        st.markdown("---")
        st.markdown("**API Keys Required:**")
        if provider == 'gemini':
            st.info("Set `GOOGLE_API_KEY` environment variable")
            st.markdown("[Get Gemini API Key](https://aistudio.google.com/app/apikey)")
        else:
            st.info("Set `MISTRAL_API_KEY` environment variable")
            st.markdown("[Get Mistral API Key](https://console.mistral.ai/)")
        
        st.markdown("---")
        st.markdown("**Tech Stack:**")
        st.markdown("- LangChain")
        st.markdown("- Sentence Transformers")
        st.markdown("- Jikan API")
        st.markdown("- Streamlit")
    
    key_name = 'GOOGLE_API_KEY' if provider == 'gemini' else 'MISTRAL_API_KEY'
    if not os.getenv(key_name, '').strip():
        st.warning(f"Add {key_name} to the local .env file before starting a conversation.")
        st.stop()

    # Chat Interface
    if not st.session_state.chat_complete:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Conversation Round: {st.session_state.conversation_round + 1}/6**")
        with col2:
            st.markdown(f"**Using: {'🤖 Gemini' if provider == 'gemini' else '🔮 Mistral'}**")
        
        # Display chat history
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Share your thoughts..."):
            # Add user message
            st.session_state.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st.session_state.conversation_round += 1
                    response = chat_with_llm(prompt, st.session_state.conversation_round, provider)
                    st.write(response)
                    
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
            
            # Check if conversation is complete
            if st.session_state.conversation_round >= 6:
                st.success("✨ Conversation complete! Analyzing your preferences...")
                
                # Analyze conversation
                with st.spinner("🧠 Analyzing conversation with AI..."):
                    analysis = analyze_conversation(provider)
                
                # Update user profile
                st.session_state.user_profile = {
                    'emotions': analysis.emotions,
                    'genres': analysis.genres,
                    'themes': analysis.themes,
                    'mood_level': analysis.mood_level,
                    'description': analysis.description,
                    'preferences': {}
                }
                
                # Generate recommendations
                st.session_state.recommendations = generate_recommendations(
                    st.session_state.user_profile,
                    provider
                )
                st.session_state.chat_complete = True
                st.rerun()
    
    # Recommendations Display
    else:
        st.markdown("---")
        st.header("🌟 Your Personalized Anime Recommendations")
        
        if st.button("🔄 Start New Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Display user profile summary
        with st.expander("📊 Your Profile Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**🎭 Emotions:**")
                st.write(", ".join(st.session_state.user_profile['emotions']))
            with col2:
                st.write("**🎬 Genres:**")
                st.write(", ".join(st.session_state.user_profile['genres']))
            with col3:
                st.write("**💭 Themes:**")
                st.write(", ".join(st.session_state.user_profile['themes']))
            
            st.write("**📝 Your Ideal Anime:**")
            st.write(st.session_state.user_profile['description'])
        
        # Display recommendations
        for i, rec in enumerate(st.session_state.recommendations, 1):
            st.markdown(f"""
                <div class="anime-card">
                    <h2>#{i} {rec['title_english']}</h2>
                    <h4>{rec['title_japanese']}</h4>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if rec['image_url']:
                    st.image(rec['image_url'], width='stretch')
                st.metric("⭐ Rating", rec['score'])
                st.metric("📺 Episodes", rec['episodes'])
            
            with col2:
                st.markdown("**📖 Synopsis:**")
                st.write(rec['synopsis'][:300] + "..." if len(rec['synopsis']) > 300 else rec['synopsis'])
                
                st.markdown("**🎬 Genres:**")
                st.write(", ".join(rec['genres']))
                
                if rec['themes']:
                    st.markdown("**🎨 Themes:**")
                    st.write(", ".join(rec['themes']))
                
                st.markdown("**💭 Why You Should Watch:**")
                st.info(rec['reason'])
                
                st.markdown(f"**🎯 Match Score:** {rec['similarity']:.2%}")
        
        st.markdown("---")
        st.markdown(f"*Powered by {('🤖 Google Gemini' if st.session_state.llm_provider == 'gemini' else '🔮 Mistral AI')}, Sentence Transformers & Jikan API*")

if __name__ == "__main__":
    main()
