"""Native Streamlit forms and callbacks; expensive work stays outside widgets."""
from uuid import uuid4
import streamlit as st
from pydantic import ValidationError
from anime_recommender import conversation
from anime_recommender.profile import GENRES, UserProfile, local_profile, readiness, emotional_context
from anime_recommender.recommendations import follow_up
from anime_recommender import config


def _pipeline_defaults():
    return {"profile_needs_analysis": False, "refinement_ids": [], "candidate_records": [],
            "ranking_diagnostics": {}, "pipeline_error": "", "refresh_requested": False, "pipeline_active": False}


def initialize_ui(state):
    conversation.initialize(state)
    for key, value in {"profile_revision": 0, **_pipeline_defaults()}.items():
        state.setdefault(key, value)


def reset_session():
    state = st.session_state
    conversation.reset(state)
    state.update(_pipeline_defaults())
    state["profile_revision"] += 1
    for key in list(state):
        if key.startswith(("feedback_value_", "feedback_reason_")) or key in ("followup_title", "chat_message", "starter"):
            del state[key]


def review_preferences():
    state = st.session_state
    state["user_profile"] = state.get("user_profile") or local_profile(state["conversation_history"])
    state["profile_needs_analysis"] = bool(state["conversation_history"])
    state["stage"] = "review"
    state["profile_revision"] += 1


def edit_preferences():
    st.session_state["stage"] = "review"
    st.session_state["profile_needs_analysis"] = False
    st.session_state["profile_revision"] += 1


def retrieve_again(refresh=False, alternatives=False):
    state = st.session_state
    if alternatives:
        state["refinement_ids"] = list(dict.fromkeys(
            state["refinement_ids"] + [item["mal_id"] for item in state["recommendations"]]))
    state.update(stage="retrieving", pipeline_error="", refresh_requested=refresh)


def continue_conversation():
    st.session_state['stage'] = 'collecting'


# Conversation starters shown before the first message; label -> first message.
STARTERS = {
    ":material/bedtime: Long day, need to unwind": "It's been a long day and I just want to unwind.",
    ":material/cloud: Feeling a bit low": "I'm feeling a bit low today.",
    ":material/sentiment_very_satisfied: Need a good laugh": "I could really use a good laugh tonight.",
    ":material/bolt: Want something epic": "I want something epic that pulls me right in.",
}


def submit_message():
    send_message(str(st.session_state.get("chat_message", "")))


def send_starter():
    choice = st.session_state.get("starter")
    if choice in STARTERS:
        send_message(STARTERS[choice])


def send_message(message):
    state = st.session_state
    message = message.strip()
    if not message:
        return
    provider = state.get('provider_select', 'gemini')
    has_key = bool(config.provider_api_key(provider))
    if state["stage"] == "results" and not has_key:
        answer = follow_up(message, state["recommendations"], state.get("followup_title"))
        turn_id = uuid4().hex
        state["conversation_history"].extend([
            {"role": "user", "content": message, "turn_id": turn_id},
            {"role": "assistant", "content": answer.answer, "turn_id": turn_id}])
        if answer.intent == "refine":
            target = next((item for item in state["recommendations"] if item["mal_id"] == answer.record_id), None)
            if target and state["user_profile"]:
                profile = state["user_profile"]
                state["user_profile"] = UserProfile(**{
                    **profile.model_dump(),
                    "genres": list(dict.fromkeys(profile.genres + target.get("genres", []))),
                    "themes": list(dict.fromkeys(profile.themes + target.get("themes", []))),
                })
            retrieve_again(alternatives=True)
        return
    state['stage'] = 'collecting'
    conversation.begin_turn(state, message, provider)


def _list(value):
    return list(dict.fromkeys(part.strip() for part in value.replace(",", "\n").splitlines() if part.strip()))


def profile_editor(profile, disabled=False):
    state = st.session_state
    prefix = f"profile_{state['profile_revision']}"
    if profile.is_fallback:
        st.warning("AI analysis was unavailable. This draft uses your words; check it before searching.")
    for notice in profile.notices:
        st.info(notice)
    with st.form(f"{prefix}_form"):
        intent = st.text_area("What would you like to watch?", value=profile.viewing_intent or profile.description,
                              key=f"{prefix}_intent", disabled=disabled)
        choices = sorted(set(GENRES) | set(profile.genres) | set(profile.excluded_genres))
        genres = st.multiselect("Preferred genres", choices, default=profile.genres, accept_new_options=True,
                               key=f"{prefix}_genres", disabled=disabled)
        excluded = st.multiselect("Genres to exclude", choices, default=profile.excluded_genres,
                                 accept_new_options=True, key=f"{prefix}_excluded", disabled=disabled)
        themes = st.text_input("Preferred themes (comma separated)", value=", ".join(profile.themes),
                               key=f"{prefix}_themes", disabled=disabled)
        watched = st.text_area("Already watched titles (one per line)", value="\n".join(profile.watched_titles),
                               key=f"{prefix}_watched", disabled=disabled)
        left, right = st.columns(2)
        minimum = left.number_input("Minimum episodes (0 means any)", min_value=0, max_value=10000,
                                    value=profile.min_episodes or 0, key=f"{prefix}_min", disabled=disabled)
        maximum = right.number_input("Maximum episodes (0 means any)", min_value=0, max_value=10000,
                                     value=profile.max_episodes or 0, key=f"{prefix}_max", disabled=disabled)
        rating = st.slider("Minimum catalogue rating (0 means any)", 0.0, 10.0,
                           value=float(profile.min_score or 0), step=0.1, key=f"{prefix}_rating", disabled=disabled)
        st.caption("Episode limits exclude unknown episode counts. Emotions are optional. Catalogue ratings are separate from preference fit.")
        submitted = st.form_submit_button("Find recommendations", type="primary", disabled=disabled)
    if submitted:
        try:
            confirmed = UserProfile(**{**profile.model_dump(), "viewing_intent": intent.strip(), "description": intent.strip(),
                "recommendation_focus": emotional_context([{'role': 'user', 'content': intent}])[1],
                "genres": genres, "excluded_genres": excluded, "themes": _list(themes), "watched_titles": _list(watched),
                "min_episodes": int(minimum) or None, "max_episodes": int(maximum) or None,
                "min_score": float(rating) or None, "is_fallback": False, "notices": []})
            if readiness(confirmed):
                st.error("Add a viewing intent, preferred genre, or theme before searching.")
                return None
            if set(confirmed.genres) & set(confirmed.excluded_genres):
                confirmed.notices.append("An excluded genre takes precedence over a preferred genre.")
            return confirmed
        except ValidationError:
            st.error("Check your profile fields. Minimum episodes cannot exceed maximum episodes.")
    return None


def save_feedback(anime_id, value, reason):
    state = st.session_state
    target = next((item for item in state["recommendations"] if item["mal_id"] == anime_id), {})
    state["feedback"][str(anime_id)] = {"value": value, "reason": reason.strip(), "genres": target.get("genres", []),
                                        "title": target.get("title", ""), "source_url": target.get("source_url", "")}
    if value == "Already watched" and state["user_profile"]:
        profile = state["user_profile"]
        profile.watched_ids = list(dict.fromkeys(profile.watched_ids + [int(anime_id)]))
    if value in ("Already watched", "Not for me"):
        state["recommendations"] = [item for item in state["recommendations"] if item["mal_id"] != anime_id]


FEEDBACK_LABELS = {"Relevant": "Sounds good", "Not for me": "Not for me", "Already watched": "Seen it already"}


def _badges(item):
    badges = []
    if item.get("score") is not None:
        badges.append(f":orange-badge[:material/star: {item['score']:.1f}]")
    if item.get("episodes") is not None:
        count = item["episodes"]
        badges.append(f":blue-badge[:material/live_tv: {count} episode{'s' * (count != 1)}]")
    badges += [f":violet-badge[{genre}]" for genre in item.get("genres", [])[:4]]
    return " ".join(badges)


def recommendation_cards(items):
    if not items:
        st.caption("That's everything from this batch. Ask me for alternatives or tell me what to change.")
    for item in items:
        anime_id = item["mal_id"]
        with st.container(border=True):
            poster, details = st.columns([1, 2.6], gap="medium")
            if item.get("image_url"):
                poster.image(item["image_url"], width="stretch", alt=f"Poster for {item['title']}")
            with details:
                st.subheader(item["title"], anchor=False)
                badges = _badges(item)
                if badges:
                    st.markdown(badges)
                st.markdown(item["reason"])
                source = item.get("source_url", "")
                if source:
                    st.link_button("See it on AniList" if "anilist.co" in source else "Catalogue source",
                                   source, icon=":material/open_in_new:", type="tertiary")
            with st.expander("More about this show", icon=":material/menu_book:"):
                st.markdown(item.get("synopsis") or "A synopsis is unavailable in the retrieved record.")
                st.caption(item.get("evidence") or "Limited catalogue metadata is available.")
                if item.get("semantic_similarity") is not None:
                    st.caption(f"Semantic similarity: {item['semantic_similarity']:.3f} · not an enjoyment probability")
                if item.get("retrieved_at"):
                    st.caption("Retrieved: " + str(item["retrieved_at"]))
                if item.get("catalogue_origin"):
                    st.caption("Source: " + item["catalogue_origin"] + ". Story descriptions and experience tags are editorial notes.")
            existing = st.session_state["feedback"].get(str(anime_id), {})
            values = list(FEEDBACK_LABELS)
            with st.expander("Tell me how this one feels", icon=":material/favorite:"):
                with st.form(f"feedback_{anime_id}", border=False):
                    value = st.selectbox("How does it sound?", values, format_func=FEEDBACK_LABELS.get,
                                         index=values.index(existing.get("value", "Relevant")),
                                         key=f"feedback_value_{anime_id}")
                    reason = st.text_input("Anything you'd like me to know? (optional)",
                                           value=existing.get("reason", ""), key=f"feedback_reason_{anime_id}")
                    if st.form_submit_button("Save feedback", icon=":material/check:"):
                        save_feedback(anime_id, value, reason)
                        st.rerun()


def saved_list():
    """Titles marked "Sounds good" this session, for the sidebar."""
    return [entry for entry in st.session_state.get("feedback", {}).values()
            if entry.get("value") == "Relevant" and entry.get("title")]
