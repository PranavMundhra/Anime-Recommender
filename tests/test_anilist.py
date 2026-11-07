"""Deterministic catalogue reliability tests; no live network or API keys."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from anime_recommender.services.anilist import AniListAPI, media_to_record
from anime_recommender.services.catalogue import normalize_record, record_is_eligible


def anime(anime_id, *, tags=(), **kwargs):
    """An AniList Media object; anime_id is the MyAnimeList ID."""
    return {"id": anime_id + 1000, "idMal": anime_id,
            "title": {"romaji": f"Title {anime_id}", "english": None, "native": None},
            "synonyms": [], "description": "A story.", "averageScore": 80, "episodes": 12,
            "isAdult": False, "genres": ["Action"], "siteUrl": f"https://anilist.co/anime/{anime_id + 1000}",
            "coverImage": {"large": "https://img.example/cover.jpg"},
            "tags": [{"name": name, "rank": 90, "category": category, "isMediaSpoiler": False,
                      "isGeneralSpoiler": False} for name, category in tags], **kwargs}


def response(media=None, *, next_page=False, status=200):
    result = Mock(status_code=status, headers={})
    result.json.return_value = {"data": {"Page": {"pageInfo": {"hasNextPage": next_page},
                                                  "media": media or []}}}
    return result


def variables(call):
    return call.kwargs["json"]["variables"]


class AniListTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = Mock()
        self.client = AniListAPI(session=self.session, cache_dir=self.directory.name,
                                 retries=0, request_interval=0)
        AniListAPI._next_request = 0

    def test_multiple_pages_deduplicate_and_stop_at_budget(self):
        self.session.post.side_effect = [response([anime(x) for x in range(1, 51)], next_page=True),
                                         response([anime(50)] + [anime(x) for x in range(51, 101)], next_page=True)]
        result = self.client.search({"preferred_genres": ["Action"]}, budget=100)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.retrieved_count, 100)
        self.assertEqual(len({x["mal_id"] for x in result.records}), 100)
        self.assertEqual(self.session.post.call_count, 2)
        for number, call in enumerate(self.session.post.call_args_list, 1):
            self.assertEqual(variables(call)["perPage"], 50)
            self.assertEqual(variables(call)["page"], number)
            self.assertEqual(variables(call)["genreIn"], ["Action"])
            self.assertIn("timeout", call.kwargs)

    def test_final_page_reports_actual_count(self):
        self.session.post.return_value = response([anime(1), anime(2)])
        result = self.client.search({}, budget=50)
        self.assertEqual(result.retrieved_count, 2)
        self.assertEqual(self.session.post.call_count, 1)

    def test_zero_matches_are_not_api_error(self):
        self.session.post.return_value = response([])
        result = self.client.search({})
        self.assertEqual(result.status, "empty")
        self.assertIsNone(result.error)

    def test_timeout_connection_rate_limit_invalid_and_http_are_distinct(self):
        for failure, expected in [(requests.Timeout(), "timeout"),
                                  (requests.ConnectionError(), "connection"),
                                  (response(status=429), "rate_limit"),
                                  (response(status=503), "upstream"),
                                  (response(status=400), "http")]:
            with self.subTest(expected=expected):
                self.session.post.reset_mock(side_effect=True)
                if isinstance(failure, Exception):
                    self.session.post.side_effect = failure
                else:
                    self.session.post.return_value = failure
                result = self.client.search({}, refresh=True)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.error_kind, expected)
                self.assertEqual(result.records, [])
        self.session.post.side_effect = None
        broken = response()
        broken.json.side_effect = ValueError("not JSON")
        self.session.post.return_value = broken
        self.assertEqual(self.client.search({}, refresh=True).error_kind, "invalid_response")

    def test_graphql_errors_without_data_are_invalid_responses(self):
        failed = response()
        failed.json.return_value = {"data": None, "errors": [{"message": "Internal error"}]}
        self.session.post.return_value = failed
        self.assertEqual(self.client.search({}, refresh=True).error_kind, "invalid_response")

    def test_transient_retry_and_no_retry_for_bad_filters(self):
        self.client.retries = 1
        self.session.post.side_effect = [response(status=429), response([anime(1)])]
        with patch("anime_recommender.services.anilist.time.sleep") as sleeper:
            result = self.client.search({})
        self.assertEqual(result.status, "ok")
        self.assertEqual(self.session.post.call_count, 2)
        self.assertEqual(sleeper.call_count, 1)
        self.session.post.reset_mock(side_effect=True)
        self.session.post.return_value = response(status=400)
        self.assertEqual(self.client.search({}, refresh=True).error_kind, "http")
        self.assertEqual(self.session.post.call_count, 1)

    def test_partial_page_failure_retains_successful_candidates(self):
        self.session.post.side_effect = [response([anime(1)], next_page=True), requests.Timeout()]
        result = self.client.search({}, budget=50)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.error_kind, "timeout")
        self.assertEqual(result.retrieved_count, 1)
        self.assertEqual(result.records[0]["mal_id"], 1)

    def test_cached_data_preserves_provenance_and_refilters_locally(self):
        self.session.post.return_value = response([anime(1), anime(2)])
        first = self.client.search({})
        second = self.client.search({"watched_ids": [1]})
        self.assertTrue(second.cache_hit)
        self.assertEqual(self.session.post.call_count, 1)
        self.assertEqual([x["mal_id"] for x in second.records], [2])
        self.assertEqual(first.records[1]["retrieved_at"], second.records[0]["retrieved_at"])
        self.assertEqual(second.records[0]["source_url"], "https://anilist.co/anime/1002")
        self.assertNotIn("watched_ids", "".join(x.read_text() for x in Path(self.directory.name).glob("*.json")))

    def test_cache_refresh_expiration_and_corruption(self):
        self.session.post.return_value = response([anime(1)])
        self.client.search({})
        self.client.search({}, refresh=True)
        self.assertEqual(self.session.post.call_count, 2)
        cache_path = next(Path(self.directory.name).glob("*.json"))
        envelope = json.loads(cache_path.read_text())
        envelope["created_at"] = 0
        cache_path.write_text(json.dumps(envelope))
        self.client.search({})
        self.assertEqual(self.session.post.call_count, 3)
        cache_path.write_text("broken JSON")
        self.client.search({})
        self.assertEqual(self.session.post.call_count, 4)

    def test_unknown_preference_never_defaults_to_action(self):
        self.session.post.return_value = response([anime(1)])
        result = self.client.search({"preferred_genres": ["obscure genre"]})
        self.assertNotIn("genreIn", variables(self.session.post.call_args))
        self.assertIn("Unrecognized", result.warnings[0])

    def test_unknown_exclusion_requires_clarification_without_api_call(self):
        result = self.client.search({"excluded_genres": ["unknown safety attribute"]})
        self.assertEqual(result.error_kind, "constraints")
        self.session.post.assert_not_called()

    def test_filters_map_to_anilist_genres_tags_and_bounds(self):
        self.session.post.return_value = response([anime(1)])
        self.client.search({"preferred_genres": ["Suspense", "Iyashikei", "Boys Love"],
                            "excluded_genres": ["Shonen", "Erotica", "Gourmet"],
                            "min_score": 7.5, "min_episodes": 10, "max_episodes": 24})
        sent = variables(self.session.post.call_args)
        self.assertEqual(sent["genreIn"], ["Thriller"])
        self.assertEqual(sent["tagIn"], ["Boys' Love", "Iyashikei"])
        self.assertEqual(sent["tagNotIn"], ["Food", "Shounen"])
        self.assertNotIn("genreNotIn", sent)
        self.assertIs(sent["isAdult"], False)
        self.assertEqual(sent["minScore"], 74)
        self.assertEqual((sent["episodesAbove"], sent["episodesBelow"]), (9, 25))

    def test_mood_focus_supplies_one_broad_genre_hint(self):
        self.session.post.return_value = response([anime(1)])
        self.client.search({"recommendation_focus": "a gentle comforting story", "excluded_genres": []})
        self.assertEqual(variables(self.session.post.call_args)["genreIn"], ["Slice of Life"])

    def test_themes_demographics_and_aliases_are_enforced(self):
        self.session.post.return_value = response([
            anime(1, genres=["Psychological"]),
            anime(2, tags=[("Shounen", "Demographic")]),
            anime(3),
        ])
        result = self.client.search({"preferred_genres": ["Sci Fi"], "excluded_genres": ["psychological", "shonen"]})
        self.assertEqual([x["mal_id"] for x in result.records], [3])
        sent = variables(self.session.post.call_args)
        self.assertEqual(sent["genreIn"], ["Sci-Fi"])
        self.assertEqual(sent["genreNotIn"], ["Psychological"])
        self.assertEqual(sent["tagNotIn"], ["Shounen"])

    def test_watched_alias_episode_bounds_and_score(self):
        records = [anime(1, title={"romaji": "Cowboy Bebop", "english": None, "native": None}),
                   anime(2, episodes=None), anime(3, episodes=100), anime(4, averageScore=60), anime(5)]
        self.session.post.return_value = response(records)
        result = self.client.search({"watched_titles": ["cowboy-bebop"], "min_episodes": 10,
                                     "max_episodes": 24, "min_score": 7})
        self.assertEqual([x["mal_id"] for x in result.records], [5])
        self.assertEqual(result.retrieved_count, 5)

    def test_missing_metadata_and_missing_mal_ids(self):
        self.session.post.return_value = response([
            {}, anime(7, idMal=None),
            {"idMal": 1, "description": None, "genres": None, "averageScore": None, "tags": None}])
        result = self.client.search({"min_score": None})
        self.assertEqual(result.retrieved_count, 1)
        record = result.records[0]
        self.assertEqual(record["synopsis"], "")
        self.assertEqual(record["genres"], [])
        self.assertIsNone(record["episodes"])
        self.assertIsNone(record["score"])
        self.assertEqual(record["images"], {})
        self.assertEqual(record["source_url"], "https://myanimelist.net/anime/1")
        self.assertTrue(any("without a MyAnimeList ID" in item for item in result.warnings))

    def test_media_conversion_cleans_text_skips_spoilers_and_weak_tags(self):
        media = anime(1, title={"romaji": "Romaji", "english": "English", "native": "Native"},
                      description="First line.<br><br>Second &amp; <i>third</i>.")
        media["tags"] = [{"name": "Iyashikei", "rank": 90, "category": "Theme"},
                         {"name": "Weak", "rank": 20, "category": "Theme"},
                         {"name": "Twist", "rank": 95, "category": "Theme", "isMediaSpoiler": True},
                         {"name": "Seinen", "rank": 30, "category": "Demographic"}]
        record = media_to_record(media)
        self.assertEqual(record["title"], "English")
        self.assertIn("Romaji", record["title_synonyms"])
        self.assertEqual(record["synopsis"], "First line.\n\nSecond & third.")
        self.assertEqual([t["name"] for t in record["themes"]], ["Iyashikei"])
        self.assertEqual([t["name"] for t in record["demographics"]], ["Seinen"])
        self.assertEqual(record["score"], 8.0)
        self.assertEqual(record["images"]["jpg"]["image_url"], "https://img.example/cover.jpg")

    def test_page_and_time_limits_are_bounded(self):
        self.client.max_pages = 2
        self.session.post.side_effect = [response([anime(1)], next_page=True), response([anime(2)], next_page=True)]
        result = self.client.search({})
        self.assertEqual(self.session.post.call_count, 2)
        self.assertTrue(any("page limit" in item for item in result.warnings))
        self.session.post.reset_mock()
        self.client.cache_ttl = 0  # Isolate the deadline from outage cache recovery.
        with patch("anime_recommender.services.anilist.time.monotonic", side_effect=[0, 100]):
            result = self.client.search({}, refresh=True)
        self.assertEqual(result.error_kind, "timeout")
        self.session.post.assert_not_called()

    def test_details_provenance_and_cache(self):
        self.session.post.return_value = response()
        self.session.post.return_value.json.return_value = {"data": {"Media": anime(1)}}
        first = self.client.details(1)
        self.assertEqual(first, self.client.details(1))
        self.assertEqual(first["mal_id"], 1)
        self.assertEqual(variables(self.session.post.call_args)["idMal"], 1)
        self.assertEqual(self.session.post.call_count, 1)

    def test_invalid_shape_is_not_cached_and_can_be_retried(self):
        broken = response()
        broken.json.return_value = {"data": {"Page": {"unexpected": "shape"}}}
        self.session.post.side_effect = [broken, response([anime(1)])]
        self.assertEqual(self.client.search({}).error_kind, "invalid_response")
        self.assertEqual(list(Path(self.directory.name).glob("*.json")), [])
        self.assertEqual(self.client.search({}).status, "ok")


class CatalogueRecordTests(unittest.TestCase):
    def test_malformed_optional_metadata_stays_usable(self):
        record = normalize_record({"mal_id": 1, "score": 8, "genres": 123, "titles": None, "title_synonyms": 1,
                                   "images": {"jpg": None}, "synopsis": None})
        self.assertEqual(record["genres"], [])
        self.assertEqual(record["titles"], [])
        self.assertEqual(record["images"]["jpg"], {})
        self.assertTrue(record_is_eligible(record, {"watched_titles": ["other"]}))

    def test_pre_display_recheck_accepts_engine_normalized_records(self):
        record = {"mal_id": 1, "title": "Title", "aliases": ["Watched alias"],
                  "genres": ["Action"], "themes": ["Psychological"],
                  "demographics": ["Shounen"], "score": 8, "episodes": 12}
        self.assertFalse(record_is_eligible(record, {"excluded_genres": ["psychological"]}))
        self.assertFalse(record_is_eligible(record, {"excluded_genres": ["shonen"]}))
        self.assertFalse(record_is_eligible(record, {"watched_titles": ["watched-alias"]}))
        self.assertFalse(record_is_eligible(record, {"watched_ids": [1]}))
        self.assertTrue(record_is_eligible(record, {}))

    def test_anilist_label_names_map_to_canonical_exclusions(self):
        record = {"mal_id": 1, "genres": [{"name": "Thriller"}], "themes": [{"name": "Boys' Love"}],
                  "demographics": [], "score": 8, "episodes": 12}
        self.assertFalse(record_is_eligible(record, {"excluded_genres": ["Suspense"]}))
        self.assertFalse(record_is_eligible(record, {"excluded_genres": ["Boys Love"]}))


if __name__ == "__main__":
    unittest.main()
