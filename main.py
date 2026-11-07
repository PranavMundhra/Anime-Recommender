"""Streamlit V2 entry point. Run with: streamlit run main.py."""
import time
import streamlit as st

st.set_page_config(page_title="Anime companion", page_icon="🌸", layout="centered")

from anime_recommender import conversation
from anime_recommender.config import API_KEY_NAMES, model_name, provider_api_key, settings
from anime_recommender.profile import UserProfile, extract_profile, local_profile
from anime_recommender.recommendations import get_embedding_model, recommend
from anime_recommender.services.anilist import AniListAPI
from anime_recommender.services.catalogue import record_is_eligible, resilient_search
from anime_recommender.services.llm import LLMProvider
from anime_recommender.ui import (STARTERS, continue_conversation, edit_preferences, initialize_ui,
    profile_editor, recommendation_cards, reset_session, retrieve_again, review_preferences, saved_list,
    send_starter, submit_message)

COMPANION = "🌸"  # Assistant avatar


@st.cache_resource(max_entries=2)
def embedding_resource(name):
    return get_embedding_model(name)


def analyse_profile():
    history = state['conversation_history']
    return extract_profile(history, provider) if has_key else local_profile(history)


def start_retrieval(**changes):
    state.update(stage='retrieving', recommendations=[], refinement_ids=[], refresh_requested=False,
                 pipeline_error='', **changes)


initialize_ui(st.session_state)
state = st.session_state
conversation.recover_interrupted_turn(state)
if state["pipeline_active"] or state["stage"] in ("ranking", "generating_explanations"):
    state.update(stage="error", pipeline_active=False,
                 pipeline_error="The previous search was interrupted. Retry the catalogue search; your preferences are preserved.")
options = settings()
pending = state['pending_turn']
busy = bool(pending and pending['status'] in ('pending', 'processing')) or state['stage'] in (
    'retrieving', 'ranking', 'generating_explanations')

st.caption('おかえり · welcome home')
st.title('Hey, you made it')
st.caption('Pull up a chair and tell me about your day. When it feels right, I’ll find a story for how you feel.')

with st.sidebar:
    st.subheader('Your cozy corner', icon=':material/cottage:', anchor=False)
    st.caption('Everything here stays in this session only.')
    st.button('Choose my own filters', on_click=review_preferences, icon=':material/tune:',
              disabled=busy or state['stage'] == 'review', width='stretch')
    st.button('Start fresh', on_click=reset_session, disabled=busy, icon=':material/restart_alt:', width='stretch')
    shelf = saved_list()
    if shelf:
        st.markdown('**On your list tonight**')
        for entry in shelf:
            st.markdown(f"- [{entry['title']}]({entry['source_url']})" if entry.get('source_url') else f"- {entry['title']}")
    with st.expander('Settings', icon=':material/settings:'):
        provider = st.selectbox('AI provider', ['gemini', 'mistral'], key='provider_select', disabled=bool(pending) or busy)
        st.caption('Model: ' + model_name(provider))
        semantic = st.toggle('Use semantic ranking', value=True, key='semantic_ranking', disabled=busy)
        st.caption('The embedding model loads only when I search for shows.')

key_name = API_KEY_NAMES[provider]
has_key = bool(provider_api_key(provider))
if not has_key:
    st.info(f'I can’t chat yet: add {key_name} to .env or Streamlit secrets. Until then, you can still pick '
            'your own filters and I’ll search the catalogue for you.', icon=':material/key:')
    if state['stage'] == 'collecting':
        st.button('Choose my own filters', key='filters_no_key', on_click=review_preferences, icon=':material/tune:')

for entry in state['conversation_history']:
    with st.chat_message(entry['role'], avatar=COMPANION if entry['role'] == 'assistant' else None):
        st.markdown(entry['content'])

if not state['conversation_history']:
    with st.chat_message('assistant', avatar=COMPANION):
        st.markdown('Hey, welcome back! How has your day been? Tell me what’s on your mind, '
                    'or how you’d like to feel by the end of tonight.')
    if has_key and not pending:
        st.pills('Start with', list(STARTERS), key='starter', on_change=send_starter, label_visibility='collapsed')

if pending and pending['status'] == 'failed':
    st.error(pending['error'] or 'The response was interrupted. Retry this turn.')
    if pending['partial']:
        with st.expander('Incomplete response'):
            st.markdown(pending['partial'])
    with st.container(horizontal=True):
        st.button('Retry response', on_click=conversation.retry_turn, args=(state,), disabled=not has_key)
        st.button('Discard response', on_click=conversation.discard_turn, args=(state,))

if state['stage'] == 'results':
    titles = {item['mal_id']: item['title'] for item in state['recommendations']}
    st.selectbox('Ask me about one of these', list(titles), index=None, format_func=lambda value: titles[value],
                 key='followup_title', placeholder='Pick a show (optional)')

prompt_disabled = bool(pending) or busy or state['stage'] not in ('collecting', 'results', 'error') or (not has_key and state['stage'] != 'results')
st.chat_input('Tell me how these feel, or ask about one' if state['stage'] == 'results' else 'Tell me about your day…',
              key='chat_message', on_submit=submit_message, disabled=prompt_disabled, submit_mode='disable', max_chars=4000)

if pending and pending['status'] == 'pending':
    with st.chat_message('assistant', avatar=COMPANION):
        status = st.status(':shimmer[Thinking it over]', type='compact')
        try:
            st.write_stream(conversation.stream_turn(state))
            status.update(label='Replied', state='complete')
        except Exception:
            status.update(label='Response interrupted; retry is available', state='error')
    if state['pending_turn'] is None:
        draft = local_profile(state['conversation_history'])
        if state['user_profile'] is None:
            state['user_profile'] = draft
        if conversation.should_search(state):
            state['stage'] = 'preparing'
        elif state['recommendations']:
            state['stage'] = 'results'
        state['search_requested'] = False
    st.rerun()

if state['stage'] == 'preparing':
    with st.status(':shimmer[Getting a feel for what you need]', type='compact'):
        previous = state['user_profile']
        profile = analyse_profile()
        if previous:
            for field in ('excluded_genres', 'watched_ids', 'watched_titles'):
                setattr(profile, field, list(dict.fromkeys(getattr(previous, field) + getattr(profile, field))))
            data = profile.model_dump()
            for field in ('min_episodes', 'max_episodes', 'min_score'):
                if data[field] is None:
                    data[field] = getattr(previous, field)
            profile = UserProfile.model_validate(data)
        state['user_profile'] = profile
    start_retrieval()
    st.rerun()

if state['stage'] == 'review':
    if state['profile_needs_analysis']:
        state['profile_needs_analysis'] = False
        with st.status(':shimmer[Remembering what you told me]', type='compact'):
            state['user_profile'] = analyse_profile()
    st.subheader('Pick your own filters', icon=':material/tune:', anchor=False)
    st.caption('Totally optional. I’ve filled in what I picked up from our chat; change anything you like.')
    st.button('Back to our chat', on_click=continue_conversation, icon=':material/forum:')
    confirmed = profile_editor(state['user_profile'] or UserProfile())
    if confirmed:
        start_retrieval(user_profile=confirmed)
        st.rerun()

if state['stage'] == 'error':
    st.error(state['pipeline_error'] or 'I couldn’t find anything that fits yet.', icon=':material/search_off:')
    for warning in state["retrieval_diagnostics"].get("warnings", []):
        st.warning(warning)
    with st.container(horizontal=True):
        st.button('Retry catalogue', on_click=retrieve_again, kwargs={'refresh': True}, icon=':material/refresh:')
        st.button('Edit filters', on_click=edit_preferences, icon=':material/tune:')
        st.button('Keep talking', on_click=continue_conversation, icon=':material/forum:')
    st.caption('Our conversation and your exclusions are safe. Filters only change when you edit them.')

if state['stage'] == 'results':
    state["recommendations"] = [item for item in state["recommendations"]
                                 if record_is_eligible(item, state["user_profile"])]
    st.subheader('Here’s what I’d queue up for you', icon=':material/live_tv:', anchor=False)
    if state['user_profile'].recommendation_focus:
        st.markdown('I went looking for ' + state['user_profile'].recommendation_focus + '. Tell me if you need a different direction.')
    retrieval = state['retrieval_diagnostics']
    ranking = state['ranking_diagnostics']
    st.caption(f"I looked through {retrieval.get('retrieved_count', 0)} shows and kept the {len(state['recommendations'])} that fit everything you told me.")
    for warning in retrieval.get('warnings', []) + ranking.get('warnings', []):
        st.warning(warning)
    with st.container(horizontal=True):
        st.button('Get alternatives', on_click=retrieve_again, kwargs={'alternatives': True}, icon=':material/shuffle:')
        st.button('Edit preferences', on_click=edit_preferences, icon=':material/tune:')
        st.button('Refresh catalogue', on_click=retrieve_again, kwargs={'refresh': True}, icon=':material/refresh:')
    recommendation_cards(state['recommendations'])
    st.caption('Your feedback shapes picks in this session only; nothing is saved or used for training.')

if state['stage'] == 'retrieving':
    state['pipeline_active'] = True
    started = time.perf_counter()
    progress = st.status(':shimmer[Browsing the shelves]', expanded=True)
    try:
        profile_data = state['user_profile'].model_dump()
        profile_data['watched_ids'] = list(dict.fromkeys(profile_data['watched_ids'] + state['refinement_ids']))
        search_profile = UserProfile(**profile_data)
        catalogue = AniListAPI(cache_dir=options.cache_dir / 'catalogue', cache_ttl=options.cache_ttl,
                              timeout=(min(4, options.request_timeout), options.request_timeout),
                              retries=options.request_retries, deadline_seconds=18)
        result = resilient_search(catalogue, search_profile, budget=options.candidate_budget, refresh=state['refresh_requested'])
        warnings = list(result.warnings)
        if result.status == "partial" and result.error:
            warnings.append(result.error)
        state['retrieval_diagnostics'] = {'retrieved_count': result.retrieved_count, 'warnings': warnings,
                                          'status': result.status, 'cache_hit': result.cache_hit,
                                          'error_kind': result.error_kind}
        if not result.records:
            raise RuntimeError(result.error or 'The catalogue search returned no matches. Review the filters or try again.')
        state['candidate_records'] = result.records
        state['stage'] = 'ranking'
        progress.update(label=':shimmer[Picking the best fits]')
        model = None
        model_warning = None
        if semantic:
            try:
                model = embedding_resource(options.embedding_model)
            except Exception:
                model_warning = 'The embedding model could not load. Ranking uses catalogue word overlap for this search.'
        llm = None
        if has_key:
            try:
                llm = LLMProvider.get_llm(provider)
            except Exception:
                pass
        def show_stage(stage):
            state['stage'] = stage
            progress.update(label=':shimmer[Noting why each one fits]' if stage == 'generating_explanations' else ':shimmer[Picking the best fits]')
        ranked = recommend(search_profile, result.records, llm=llm, embedding_model=model,
                           cache_dir=options.cache_dir / 'embeddings', cache_ttl=options.cache_ttl,
                           feedback=state['feedback'], progress=show_stage)
        if model_warning:
            ranked.diagnostics['warnings'].append(model_warning)
        state['ranking_diagnostics'] = ranked.diagnostics
        ranked.items = [item for item in ranked.items if record_is_eligible(item, search_profile)]
        if not ranked.items:
            raise RuntimeError('No sourced titles satisfy your confirmed constraints. Edit the filters to choose any relaxation.')
        state.update(recommendations=ranked.items, stage='results', refresh_requested=False, pipeline_active=False)
        state['ranking_diagnostics']['pipeline_seconds'] = time.perf_counter() - started
        progress.update(label='Found a few for you', state='complete', expanded=False)
    except Exception as error:
        state.update(stage='error', pipeline_error=str(error), refresh_requested=False, pipeline_active=False)
        progress.update(label='The search hit a snag', state='error')
    st.rerun()
