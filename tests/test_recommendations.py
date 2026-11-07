import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from anime_recommender.recommendations import eligible, follow_up, normalize_record, recommend


RECORDS = json.loads((Path(__file__).parent / "fixtures/catalogue.json").read_text())['records']


class FakeModel:
    model_name = "test-embedding-v1"

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return [[1.0, 1.0 if "Action" in text else 0.0, float(len(text) % 3)] for text in texts]


class FakeLLM:
    def __init__(self, malicious=False):
        self.calls = 0
        self.malicious = malicious

    def invoke(self, messages):
        self.calls += 1
        data = json.loads(messages[-1]["content"])
        return SimpleNamespace(content=json.dumps([{"id":r["id"], "reason":"A perfectly safe anime with 3 episodes." if self.malicious else r["options"][-1]} for r in data['candidates']]))


class RecommendationTests(unittest.TestCase):
    def test_hard_constraints_and_watched_aliases(self):
        profile = {"excluded_genres":["action"], "watched_titles":["NATSUME’S BOOK OF FRIENDS"], "max_episodes":26, "min_score":7}
        result = recommend(profile, RECORDS)
        ids = {r['mal_id'] for r in result.items}
        self.assertFalse(ids & {1,6,20,4081,1535,900001})
        self.assertTrue(ids)

    def test_unknown_episodes_never_pass_explicit_bounds(self):
        missing = normalize_record(RECORDS[-1])
        self.assertFalse(eligible(missing, {"max_episodes":24}))
        self.assertTrue(eligible(missing, {}))

    def test_theme_demographic_and_watched_id_exclusions(self):
        record = normalize_record(dict(RECORDS[0], demographics=[{'name':'Seinen'}]))
        for profile in ({'excluded_genres':['Space']}, {'excluded_genres':['Seinen']}, {'watched_ids':[1]}):
            self.assertFalse(eligible(record, profile))
        self.assertTrue(eligible(record, {'watched_titles':['Cowboy']}))  # exact alias matching

    def test_missing_fields_and_duplicates(self):
        result = recommend({}, [RECORDS[-1], RECORDS[-1]])
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0]['synopsis'], '')
        self.assertIsNone(result.items[0]['episodes'])
        self.assertEqual(result.diagnostics['missing_synopsis_count'], 1)
        self.assertEqual(result.diagnostics['skipped_duplicates'], 1)

    def test_embeddings_batched_cached_and_content_invalidated(self):
        model = FakeModel()
        with tempfile.TemporaryDirectory() as path:
            first = recommend({}, RECORDS[:3], embedding_model=model, cache_dir=path)
            self.assertEqual([len(batch) for batch in model.calls], [3,1])
            second = recommend({}, RECORDS[:3], embedding_model=model, cache_dir=path)
            self.assertEqual(second.diagnostics['embedding_cache_hits'], 3)
            self.assertEqual(len(model.calls), 3)  # second invocation only query
            changed = [dict(RECORDS[0], synopsis="Changed content")]
            third = recommend({}, changed, embedding_model=model, cache_dir=path)
            self.assertEqual(third.diagnostics['embedding_cache_hits'], 0)
            self.assertEqual(first.diagnostics['ranking_method'], 'semantic')
            expired = recommend({}, changed, embedding_model=model, cache_dir=path, cache_ttl=0)
            self.assertEqual(expired.diagnostics['embedding_cache_hits'], 0)

    def test_embedding_failure_is_observable_lexical_fallback(self):
        class Broken:
            def encode(self, texts):
                raise RuntimeError('private API key must not appear')
        result = recommend({"genres":["Comedy"]}, RECORDS, embedding_model=Broken(), cache_dir=None)
        self.assertEqual(result.diagnostics['ranking_method'], 'lexical')
        self.assertEqual(result.items[0]['mal_id'], 22789)
        self.assertIsNone(result.items[0]['semantic_similarity'])
        self.assertNotIn('private', str(result.diagnostics))
        self.assertEqual(result.diagnostics['embedding_failure_count'], len(RECORDS))

    def test_one_batched_grounded_explanation_call(self):
        llm = FakeLLM()
        result = recommend({"genres":["Action"]}, RECORDS, llm=llm)
        self.assertEqual(llm.calls, 1)
        for item in result.items:
            self.assertLessEqual(len(item['reason'].split()), 25)
            self.assertTrue(item['source_url'])
            self.assertTrue(item['evidence'])

    def test_untrusted_reasons_are_rejected(self):
        result = recommend({}, RECORDS, llm=FakeLLM(malicious=True))
        self.assertEqual(result.diagnostics['rejected_explanations'], len(result.items))
        self.assertNotIn('perfectly', str(result.items))

    def test_provider_failure_returns_valid_reasons(self):
        class Broken:
            def invoke(self, messages):
                raise RuntimeError('secret')
        result = recommend({}, RECORDS, llm=Broken())
        self.assertTrue(result.items)
        self.assertNotIn('secret', str(result))

    def test_feedback_and_refinement_keep_exclusions(self):
        feedback = {'1':{'value':'Already watched'}, '6':{'value':'Not for me'}}
        result = recommend({"genres":["Action"],"watched_ids":[20]}, RECORDS, feedback=feedback)
        self.assertFalse({1,6,20} & {r['mal_id'] for r in result.items})
        follow = follow_up('What else has similar themes?', result.items)
        self.assertEqual(follow.intent, 'refine')

    def test_relevant_feedback_boosts_alternative_when_original_absent(self):
        records = [RECORDS[1], RECORDS[5]]  # original Cowboy Bebop ID1 absent
        feedback = {'1':{'value':'Relevant', 'genres':['Action','Sci-Fi']}}
        result = recommend({}, records, feedback=feedback)
        self.assertEqual(result.items[0]['mal_id'], 6)
        self.assertAlmostEqual(result.items[0]['ranking_score'], 0.08)
        constrained = recommend({'excluded_genres':['Action']}, records, feedback=feedback)
        self.assertEqual([r['mal_id'] for r in constrained.items], [22789])

    def test_followup_uses_retrieved_facts(self):
        item = recommend({}, [RECORDS[0]]).items[0]
        self.assertIn('26 episodes', follow_up('How many episodes?', [item]).answer)
        self.assertIn(item['source_url'], follow_up('Why this one?', [item]).answer)
        missing = recommend({}, [RECORDS[-1]]).items[0]
        self.assertIn('does not list', follow_up('how many episodes', [missing]).answer)
        ambiguous = follow_up('why this one', recommend({}, RECORDS).items)
        self.assertIn('Select', ambiguous.answer)

    def test_empty_result_is_recoverable(self):
        result = recommend({"max_episodes":1}, RECORDS)
        self.assertFalse(result.items)
        self.assertTrue(result.diagnostics['warnings'])

    def test_no_invented_source_for_missing_id(self):
        result = recommend({}, [{'title':'Untitled source-free record'}])
        self.assertFalse(result.items)
        self.assertEqual(result.diagnostics['skipped_missing_source_or_id'], 1)

    def test_offline_evaluation_release_targets(self):
        from anime_recommender.evaluation import run_evaluation
        report = run_evaluation()
        self.assertTrue(report['targets_passed'])
        self.assertEqual(report['aggregate']['v2']['hard_constraint_violations'], 0)
        self.assertGreater(report['aggregate']['v1']['hard_constraint_violations'], 0)


if __name__ == '__main__':
    unittest.main()
