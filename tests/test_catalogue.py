import tempfile
import json
from pathlib import Path
import time
import unittest
from unittest.mock import Mock

import requests

from anime_recommender.profile import UserProfile, local_profile
from anime_recommender.recommendations import recommend
from anime_recommender.services.catalogue import resilient_search, starter_records
from anime_recommender.services.anilist import AniListAPI
from anime_recommender.services.catalogue import SearchResult
from tests.test_anilist import anime, response


class CatalogueRecoveryTests(unittest.TestCase):
    def test_expired_cache_survives_outage_with_original_date(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Mock()
            client = AniListAPI(session=session, cache_dir=directory, retries=0, request_interval=0)
            session.post.return_value = response([anime(1)])
            first = client.search(UserProfile())
            cached_path = next(Path(directory).glob('*.json'))
            envelope = json.loads(cached_path.read_text())
            envelope['created_at'] = time.time() - 2 * 86400
            cached_path.write_text(json.dumps(envelope))
            session.post.side_effect = requests.Timeout()
            result = client.search(UserProfile(watched_ids=[2]))
            self.assertTrue(result.records)
            self.assertTrue(result.cache_hit)
            self.assertEqual(result.records[0]['retrieved_at'], first.records[0]['retrieved_at'])
            self.assertTrue(any('previously saved' in warning for warning in result.warnings))

    def test_cache_older_than_seven_days_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Mock()
            client = AniListAPI(session=session, cache_dir=directory, retries=0, request_interval=0)
            session.post.return_value = response([anime(1)])
            client.search(UserProfile())
            cached_path = next(Path(directory).glob('*.json'))
            envelope = json.loads(cached_path.read_text())
            envelope['created_at'] = time.time() - 8 * 86400
            cached_path.write_text(json.dumps(envelope))
            session.post.side_effect = requests.Timeout()
            result = client.search(UserProfile())
            self.assertEqual(result.error_kind, 'timeout')
            self.assertFalse(result.records)

    def test_first_run_outage_supplies_honest_editorial_suggestions(self):
        client = Mock()
        client.search.return_value = SearchResult(status='error', error_kind='timeout', error='AniList timed out.')
        profile = local_profile([{'role': 'user', 'content': "I'm feeling lonely, please suggest something comforting"}])
        result = resilient_search(client, profile)
        self.assertEqual(result.status, 'fallback')
        ranked = recommend(profile, result.records, cache_dir=None, limit=3)
        self.assertTrue(ranked.items)
        self.assertTrue({22789, 4081, 34798} & {r['mal_id'] for r in ranked.items})
        for record in ranked.items:
            self.assertIsNone(record['score'])
            self.assertIsNone(record['episodes'])
            self.assertIsNone(record['retrieved_at'])
            self.assertEqual(record['catalogue_origin'], 'editorial starter collection')
        self.assertTrue(any('starter collection' in warning for warning in result.warnings))

    def test_fallback_preserves_exclusions_watched_and_unknown_bounds(self):
        profile = UserProfile(excluded_genres=['Comedy'], watched_ids=[4081])
        records = starter_records(profile)
        self.assertTrue(records)
        self.assertNotIn(4081, [r['mal_id'] for r in records])
        self.assertTrue(all('Comedy' not in [g['name'] for g in r['genres']] for r in records))
        self.assertFalse(starter_records(UserProfile(max_episodes=13)))
        self.assertFalse(starter_records(UserProfile(min_score=7)))

    def test_empty_search_or_invalid_constraint_does_not_trigger_fallback(self):
        for original in (SearchResult(status='empty'), SearchResult(status='error', error_kind='constraints')):
            client = Mock()
            client.search.return_value = original
            self.assertIs(resilient_search(client, UserProfile()), original)
