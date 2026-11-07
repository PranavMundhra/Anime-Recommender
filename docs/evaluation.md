# V2 offline evaluation

Run date: 14 October 2026. Environment: Python 3.11.5, Windows, Streamlit 1.65.0.

The [machine-readable report](evaluation-results.json) compares actual saved pre-V2 ranking functions with V2 on identical records. The baseline already contains the fixes made before this V2 work; it is preserved in `tests/fixtures/v1_baseline.py`.

The catalogue fixture is hand-authored and deliberately includes missing metadata. It is not a scraped live catalogue snapshot or human relevance dataset. Seven profiles cover brief requests, detailed preferences, watched aliases, exclusions, conflicting bounds, obscure themes, and missing information. Deterministic keyword vectors and mocked LLM outputs allow network-free reproduction.

## Release targets and measured results

Targets are declared before execution in `anime_recommender/evaluation.py`: zero hard-constraint violations, complete source/evidence coverage for returned cards, at most one explanation call per recommendation run, and zero repeated catalogue vector encodes for an unchanged query with a valid cache.

| Measure | Pre-V2 baseline | V2 |
| --- | ---: | ---: |
| Hard-constraint violations across seven profiles | 20 | 0 |
| Maximum explanation calls per run | 5 | 1 |
| Repeated catalogue vectors encoded across seven repeated queries | 63 | 0 |
| Mean expected-fixture-title recall at five | 0.762 | 1.000 |

All declared software targets pass. Source and evidence coverage are checked for every returned card; empty results are counted separately rather than treated as invented recommendations.

Recall is a fixture-specific software proxy. It does not establish human-rated relevance. The comparison supplies the complete fixture to both versions, so it does not measure live Jikan retrieval coverage, outages, pagination latency, or catalogue popularity bias. The separate Jikan unit tests cover request construction and recovery with mocks.

## Unmeasured release evidence

Production time to visible user message, first provider token, and final results requires a running browser and live provider. Mock local timings in the JSON are computation timings, not production response speeds. Prompt/response word counts are not billed tokenizer usage or API costs.

Human relevance, factual explanations on a representative live catalogue, preference-fit diversity, streamed-message ordering, and model availability for the owner's account require live review. Startup/profile screenshots and a real desktop/mobile review are now available in `docs/browser-review.md`; no provider or catalogue request was submitted during that review. A future human rubric should score relevance and evidence support from 1–5, record every explicit-constraint violation, and compare the same confirmed profiles and catalogue snapshot. A zero-violation target remains mandatory; human scores and production latency/cost should be reported only after measurement.

Proposed live release targets, not measured results: mean human preference-fit score at least 4/5, zero unsupported factual explanation claims in the reviewed sample, P95 time to first visible provider text at most five seconds, and P95 warm-cache final recommendations at most ten seconds. Record cold model-download times separately. Record actual billed tokens/cost without substituting word counts; the owner should set a billing budget before the live evaluation.

Live Jikan checks timed out even after approved network access. Gemini/Mistral calls were not made. No measured production speed-up is claimed.

Reproduce:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --output docs/evaluation-results.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
