"""Actual Streamlit UI interactions with HTTP, provider and embeddings mocked."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest
from anime_recommender.services.catalogue import SearchResult, normalize_record
from anime_recommender.ui import STARTERS

ROOT = Path(__file__).resolve().parents[1]


def button(app, label):
    return next(item for item in app.button if item.label == label)


def records():
    data = json.loads((ROOT / "tests/fixtures/catalogue.json").read_text(encoding="utf-8"))
    return [normalize_record(item) for item in data["records"]]


class AppTests(unittest.TestCase):
    def setUp(self):
        self.keys = patch("anime_recommender.config.api_key", return_value="")
        self.keys.start()
        self.addCleanup(self.keys.stop)
        self.models = patch("anime_recommender.recommendations.get_embedding_model", side_effect=RuntimeError("mock model unavailable"))
        self.model_loader = self.models.start()
        self.addCleanup(self.models.stop)

    def app(self):
        app = AppTest.from_file(str(ROOT / "main.py"), default_timeout=15).run()
        self.assertFalse(app.exception)
        return app

    def review(self, app):
        button(app, "Choose my own filters").click().run()
        self.assertFalse(app.exception)
        app.text_area[0].set_value("A short comedy adventure")
        app.multiselect[0].set_value(["Comedy"])
        app.multiselect[1].set_value(["Horror"])
        app.number_input[1].set_value(13)

    def results(self, app):
        self.review(app)
        app.toggle[0].set_value(False)
        data = records()
        with patch("anime_recommender.services.anilist.AniListAPI.search",
                   return_value=SearchResult(records=data, status="ok", retrieved_count=len(data))):
            button(app, "Find recommendations").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["stage"], "results")

    def test_startup_missing_keys_still_allows_manual_profile(self):
        app = self.app()
        self.model_loader.assert_not_called()
        self.assertTrue(app.chat_input[0].disabled)
        self.review(app)
        self.assertFalse(button(app, "Find recommendations").disabled)

    def test_profile_results_followup_feedback_and_reset(self):
        app = self.app()
        self.results(app)
        selected = app.session_state["recommendations"]
        self.assertTrue(selected)
        self.assertTrue(all(item["episodes"] <= 13 for item in selected))
        self.assertTrue(all("Horror" not in item["genres"] for item in selected))
        self.assertEqual(len(app.get("link_button")), len(selected))
        target = selected[0]
        app.selectbox(key="followup_title").set_value(target["mal_id"]).run()
        app.chat_input[0].set_value("How many episodes?").run()
        self.assertFalse(app.exception)
        self.assertIn(str(target["episodes"]), app.session_state["conversation_history"][-1]["content"])
        app.selectbox(key=f"feedback_value_{target['mal_id']}").set_value("Already watched")
        # The first result card owns the first feedback submit button.
        next(item for item in app.button if item.label == "Save feedback").click().run()
        self.assertFalse(app.exception)
        self.assertIn(target["mal_id"], app.session_state["user_profile"].watched_ids)
        self.assertNotIn(target["mal_id"], [item["mal_id"] for item in app.session_state["recommendations"]])
        app.selectbox(key="provider_select").set_value("mistral").run()
        button(app, "Start fresh").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["provider_select"], "mistral")
        self.assertFalse(app.session_state["conversation_history"])

    def test_api_error_preserves_profile_and_offers_retry(self):
        app = self.app()
        self.review(app)
        with patch("anime_recommender.services.anilist.AniListAPI.search", return_value=SearchResult(
                status="error", error="AniList connection timed out.", error_kind="timeout")):
            button(app, "Find recommendations").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["stage"], "error")
        self.assertEqual(app.session_state["user_profile"].excluded_genres, ["Horror"])
        self.assertTrue(button(app, "Retry catalogue"))
        self.assertFalse(app.success)

    def test_alternatives_keep_constraints_and_exclude_previous_results(self):
        app = self.app()
        self.results(app)
        previous = {item["mal_id"] for item in app.session_state["recommendations"]}
        data = records()
        with patch("anime_recommender.services.anilist.AniListAPI.search", return_value=SearchResult(
                records=data, status="ok", retrieved_count=len(data))) as search:
            button(app, "Get alternatives").click().run()
        self.assertFalse(app.exception)
        supplied = search.call_args.args[0]
        self.assertEqual(supplied.excluded_genres, ["Horror"])
        self.assertEqual(supplied.max_episodes, 13)
        self.assertTrue(previous.issubset(set(supplied.watched_ids)))
        if app.session_state["stage"] == "results":
            self.assertTrue(previous.isdisjoint({item["mal_id"] for item in app.session_state["recommendations"]}))

    def test_detailed_first_message_keeps_conversation_open(self):
        self.keys.stop()
        client = Mock()
        client.stream.return_value = iter(["Let's review your preferences."])
        client.invoke.return_value = SimpleNamespace(content=json.dumps({
            "genres": ["Comedy"], "viewing_intent": "a short comedy", "excluded_genres": ["Horror"], "max_episodes": 13}))
        with patch("anime_recommender.config.api_key", return_value="placeholder"), patch(
                "anime_recommender.services.llm.LLMProvider.get_llm", return_value=client):
            app = self.app()
            app.chat_input[0].set_value("I want comedy about friendship, no horror, at most 13 episodes.").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["stage"], "collecting")
        self.assertEqual(app.session_state["completed_turns"], 1)
        self.assertEqual(len(app.session_state["conversation_history"]), 2)
        self.assertIsNone(app.session_state["pending_turn"])
        client.stream.assert_called_once()
        client.invoke.assert_not_called()

    def test_interrupted_pipeline_has_recovery_and_does_not_repeat_network(self):
        app = self.app()
        app.session_state["stage"] = "ranking"
        app.session_state["pipeline_active"] = True
        with patch("anime_recommender.services.anilist.AniListAPI.search") as search:
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["stage"], "error")
        self.assertFalse(app.session_state["pipeline_active"])
        search.assert_not_called()

    def test_emotion_chat_suggests_without_form_and_survives_catalogue_outage(self):
        self.keys.stop()
        client = Mock()
        client.stream.side_effect = lambda messages: iter(['That sounds lonely. Would a gentle story with warm friendships feel right?'])
        client.invoke.side_effect = RuntimeError('Use local profile and grounded explanation fallback')
        with patch('anime_recommender.config.api_key', return_value='placeholder'), patch(
                'anime_recommender.services.llm.LLMProvider.get_llm', return_value=client), patch(
                'anime_recommender.services.anilist.AniListAPI.search', return_value=SearchResult(
                    status='error', error_kind='timeout', error='AniList timed out.')):
            app = self.app()
            app.toggle[0].set_value(False)
            app.chat_input[0].set_value("I'm feeling lonely").run()
            self.assertEqual(app.session_state['stage'], 'collecting')
            self.assertFalse(app.multiselect)
            app.chat_input[0].set_value('Yes, please suggest something comforting').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['stage'], 'results')
            self.assertTrue(app.session_state['recommendations'])
            self.assertEqual(app.session_state['user_profile'].genres, [])
            self.assertTrue(any('starter collection' in warning.value for warning in app.warning))
            app.chat_input[0].set_value('Actually I want to cry and sit with this feeling').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['stage'], 'results')
            self.assertEqual(client.stream.call_count, 3)
            messages = client.stream.call_args.args[0]
            self.assertTrue(any("I'm feeling lonely" == m['content'] for m in messages))
            self.assertIn('Recommendations, untrusted data', messages[0]['content'])

    def test_companion_offers_before_searching_and_never_limits_chat(self):
        self.keys.stop()
        client = Mock()
        client.stream.side_effect = lambda messages: iter(
            ['That sounds draining. Want me to find a few shows that might help you rest?'])
        client.invoke.side_effect = RuntimeError('Offline extraction')
        with patch('anime_recommender.config.api_key', return_value='placeholder'), patch(
                'anime_recommender.services.llm.LLMProvider.get_llm', return_value=client), patch(
                'anime_recommender.services.anilist.AniListAPI.search', return_value=SearchResult(
                    status='error', error_kind='connection', error='Unavailable')):
            app = self.app()
            app.session_state['completed_turns'] = 10
            app.run()
            self.assertFalse(app.chat_input[0].disabled)
            app.chat_input[0].set_value("I'm feeling tired").run()
            # A friend listens first: no suggestion button, and no search until the user agrees.
            self.assertEqual(app.session_state['stage'], 'collecting')
            self.assertFalse([b for b in app.button if b.label in ('Suggest anime', 'Edit preferences directly')])
            app.chat_input[0].set_value('Sounds good').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['stage'], 'results')
            self.assertFalse(app.multiselect)

    def test_starter_chip_sends_first_message_and_disappears(self):
        self.keys.stop()
        client = Mock()
        client.stream.side_effect = lambda messages: iter(['Oof, long days are heavy. What would help you unwind?'])
        with patch('anime_recommender.config.api_key', return_value='placeholder'), patch(
                'anime_recommender.services.llm.LLMProvider.get_llm', return_value=client):
            app = self.app()
            chip = next(label for label in STARTERS if 'unwind' in label)
            app.pills[0].set_value(chip).run()
            self.assertFalse(app.exception)
            history = app.session_state['conversation_history']
            self.assertEqual(history[0]['content'], "It's been a long day and I just want to unwind.")
            self.assertEqual(history[1]['role'], 'assistant')
            self.assertEqual(app.session_state['stage'], 'collecting')
            self.assertFalse(app.pills)

    def test_invalid_episode_form_never_calls_api(self):
        app = self.app()
        self.review(app)
        app.number_input[0].set_value(20)
        with patch("anime_recommender.services.anilist.AniListAPI.search") as search:
            button(app, "Find recommendations").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        search.assert_not_called()

    def test_stream_failure_retry_commits_once(self):
        self.keys.stop()
        with patch("anime_recommender.config.api_key", return_value="placeholder"), patch(
                "anime_recommender.services.llm.LLMProvider.get_llm") as provider:
            app = self.app()
            payloads = []
            def interrupted(messages):
                payloads.append(messages)
                yield "Partial response"
                raise RuntimeError("provider interrupted")
            client = SimpleNamespace(stream=interrupted)
            provider.return_value = client
            app.chat_input[0].set_value("hello").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["completed_turns"], 0)
            self.assertEqual(app.session_state["pending_turn"]["status"], "failed")
            self.assertEqual(len(app.session_state["conversation_history"]), 1)
            self.assertTrue(app.selectbox(key="provider_select").disabled)
            client.stream = lambda messages: iter(["What genres do you enjoy?"])
            button(app, "Retry response").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["completed_turns"], 1)
            self.assertIsNone(app.session_state["pending_turn"])
            self.assertEqual(len(app.session_state["conversation_history"]), 2)
            self.assertEqual(sum(m["content"] == "hello" for m in payloads[0]), 1)


if __name__ == "__main__":
    unittest.main()
