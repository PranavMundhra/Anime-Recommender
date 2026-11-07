"""Offline V1/V2 regression evaluation; no provider, network or model downloads."""
from __future__ import annotations

import ast
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from anime_recommender.recommendations import _cosine, eligible, normalize_record, normalize_title, recommend


ROOT = Path(__file__).resolve().parent.parent
TARGETS = {"v2_hard_constraint_violations": 0, "v2_source_coverage": 1.0,
           "v2_evidence_coverage": 1.0, "v2_max_explanation_calls_per_run": 1,
           "v2_repeat_catalogue_vectors_encoded": 0}


class Vector(list):
    def reshape(self, *args):
        return [self]


class CountingModel:
    """Deterministic keyword vectors; evaluates wiring, not learned relevance."""
    model_name = "offline-keyword-evaluation-v1"
    keywords = ("action", "adventure", "comedy", "slice of life", "sports", "personal growth", "supernatural", "psychological")

    def __init__(self):
        self.batches = 0
        self.texts_encoded = 0

    def encode(self, texts):
        self.batches += 1
        self.texts_encoded += 1 if isinstance(texts, str) else len(texts)
        def vector(text):
            normalized = normalize_title(text)
            return Vector([float(normalized.count(word)) for word in self.keywords] + [0.1])
        return vector(texts) if isinstance(texts, str) else [vector(text) for text in texts]


class CountingLLM:
    def __init__(self):
        self.calls = 0
        self.prompt_words = 0
        self.response_words = 0

    def invoke(self, messages):
        self.calls += 1
        content = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content
        self.prompt_words += sum(len((m['content'] if isinstance(m, dict) else m.content).split()) for m in messages)
        if content.startswith('{'):
            candidates = json.loads(content)['candidates']
            response = json.dumps([{"id":r['id'], "reason":r['options'][-1]} for r in candidates])
        else:
            response = "Consider this title based on its retrieved catalogue metadata."
        self.response_words += len(response.split())
        return SimpleNamespace(content=response)


def _baseline(records, profile, model, llm, path):
    """Execute actual saved V1 ranking functions with offline dependencies injected.

    Snapshot retrieval supplies the same full records to both versions, so pagination
    and live network are deliberately outside this comparison.
    """
    names = {'generate_user_embedding', 'generate_anime_embedding', 'generate_recommendations', 'generate_recommendation_reason'}
    source = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    requests = {'count':0}
    def fetch(*args, **kwargs):
        requests['count'] += 1
        return records
    namespace = {'load_embedding_model': lambda: model, 'init_llm': lambda provider: llm,
                 'HumanMessage': lambda content: SimpleNamespace(content=content),
                 'cosine_similarity': lambda a, b: [[_cosine(a[0], b[0])]],
                 'JikanAPI': SimpleNamespace(search_anime_by_genres=fetch),
                 'st': SimpleNamespace(spinner=lambda *args: nullcontext(), error=lambda *args: None)}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    p = {'emotions':[], 'genres':[], 'themes':[], 'description':'', **profile}
    output = namespace['generate_recommendations'](p)
    indexed = {r.get('title_english') or r['title']: r for r in records}
    items = [{**normalize_record(indexed[r['title_english']]), 'reason':r['reason'],
              'source_url':r.get('source_url', '')} for r in output]
    return items, requests['count']


def run_evaluation(fixture_path=None, baseline_path=None):
    fixture_path = Path(fixture_path or ROOT / 'tests/fixtures/catalogue.json')
    baseline_path = Path(baseline_path or ROOT / 'tests/fixtures/v1_baseline.py')
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    records = fixture['records']
    report = {'method': 'Actual saved V1 functions and V2 on identical hand-authored records with deterministic keyword embedding and mocked provider. No live API/model/human scores.',
              'fixture_description': fixture['description'], 'fixture_version':fixture['fixture_version'],
              'targets_defined_before_execution': TARGETS, 'profiles':[],
              'limitations':['Keyword vectors do not measure human relevance or production semantic quality.',
                             'Mock elapsed time is local computation only; first visible message/token latency and billed token/cost require live UI/provider measurement.',
                             'Retrieval coverage uses the complete fixture input, not live catalogue pagination performance.',
                             'V1 snapshot includes earlier reliability fixes and is the pre-V2 repository baseline.',
                             'Prompt/response word counts are instrumentation, not tokenizer counts or billed API costs.']}
    with tempfile.TemporaryDirectory() as cache:
        for case in fixture['profiles']:
            p, expected = case['profile'], set(case['expected_ids'])
            row = {'name':case['name'], 'expected_fixture_ids':sorted(expected)}
            for version in ('v1','v2'):
                model, llm = CountingModel(), CountingLLM()
                def run():
                    if version == 'v1':
                        return _baseline(records, p, model, llm, baseline_path)
                    result = recommend(p, records, llm=llm, embedding_model=model, cache_dir=Path(cache) / case['name'])
                    return result.items, 0
                start = time.perf_counter()
                items, external_requests = run()
                duration = time.perf_counter() - start
                encoded_before, calls_before = model.texts_encoded, llm.calls
                run()  # Repeat run measures catalogue-vector cache reuse.
                ids = {r['mal_id'] for r in items}
                row[version] = {'selected_ids': [r['mal_id'] for r in items],
                                'hard_constraint_violations':sum(not eligible(r,p) for r in items),
                                'source_coverage':sum(bool(r['source_url']) for r in items)/len(items) if items else 1.0,
                                'evidence_coverage':sum(bool(r.get('evidence')) for r in items)/len(items) if items else 1.0,
                                'fixture_expected_recall_at_5':len(expected & ids)/len(expected) if expected else (1.0 if not items else 0.0),
                                'input_retrieval_recall':len(expected & {r['mal_id'] for r in records}) / len(expected) if expected else 1.0,
                                'distinct_genre_sets':len({tuple(sorted(r['genres'])) for r in items}),
                                'local_mock_elapsed_seconds':duration,
                                'explanation_calls':calls_before,
                                'embedding_texts_first_run':encoded_before,
                                'repeat_catalogue_vectors_encoded':max(0, model.texts_encoded-encoded_before-1),
                                'mock_retrieval_requests_first_run':external_requests,
                                'prompt_words_two_runs':llm.prompt_words,'response_words_two_runs':llm.response_words}
            report['profiles'].append(row)
    rows = report['profiles']
    report['aggregate'] = {v:{'hard_constraint_violations':sum(r[v]['hard_constraint_violations'] for r in rows),
                             'max_explanation_calls':max(r[v]['explanation_calls'] for r in rows),
                             'repeat_catalogue_vectors_encoded':sum(r[v]['repeat_catalogue_vectors_encoded'] for r in rows),
                             'mean_fixture_expected_recall_at_5':sum(r[v]['fixture_expected_recall_at_5'] for r in rows)/len(rows)} for v in ('v1','v2')}
    report['targets_passed'] = all(r['v2']['hard_constraint_violations']==0 and r['v2']['source_coverage']==1 and r['v2']['evidence_coverage']==1 and r['v2']['explanation_calls']<=1 and r['v2']['repeat_catalogue_vectors_encoded']==0 for r in rows)
    return report
