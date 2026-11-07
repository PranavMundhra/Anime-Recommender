# V2 implementation and verification

Updated: 15 October 2026. The conversation now follows the user's emotional context and desired story experience. Suggestions start when the user accepts the companion's offer or asks in chat (the Suggest anime button was removed on 7 November 2026); profile review is optional. The six-turn cutoff and fixed question phases have been removed. Follow-up conversation uses history and supplied recommendation facts when an AI key is present. Catalogue failures recover through labeled saved pages or a small editorial starter collection while preserving hard constraints. Public-network success, authenticated provider output, human relevance, and production latency/cost remain unverified release evidence. Earlier screenshots predate this revision.

## Corrections

| Requirement | Implementation | Verification |
| --- | --- | --- |
| C01 User message visible before generation | Chat submit callback commits the user message; replay occurs before the stream begins | Conversation transaction tests and actual AppTest; real browser timing pending |
| C02 No duplicate user message | One authoritative history payload | Conversation and AppTest payload assertions |
| C03 Safe streaming commit | Partial output stays in pending state; completion commits once; interrupted/truncated responses offer retry | Failed streams, cancellation, retry, SDK mixed-content parser tests |
| C04 Rerun/overlap protection | Unique turn IDs, disabled input/provider controls, successful-turn counter, interrupted pipeline recovery | Overlap, rerun, AppTest interruption/retry checks |
| C05 Honest safe fallbacks | Local profile draft labels; validated neutral reason templates; no unbound fallback data | Profile parsing/schema failure and provider failure tests |
| C06 Bounded calls and empty results | Timeout/retry/deadline budgets; distinct errors; error stage preserves history and filters | HTTP timeout/429/400/invalid shape, zero result, partial retrieval, UI retry tests |
| C07 Pagination and actual counts | At most 25/page; ID deduplication; candidate/page/time budgets; actual retrieved count | Multiple/final/duplicate/partial pages and bounded-work tests |
| C08 Valid profile and constraints | Allowed moods, confidence bounds, list validation, explicit exclusions/watch/episode/rating fields | Schema and explicit-constraint preservation tests |
| C09 Honest scores and reasons | Semantic similarity numeric label; catalogue ratings separate; supported reasons at most 25 words | Ranking/reason validation tests and card rendering |
| C10 Scoped reset | Conversation/profile/feedback/pending/results reset; provider and unrelated settings remain | Pure state reset and AppTest reset checks |
| C11 Metadata and ranking failures | Normalized nullable fields; fallback ranking with missing/skipped/failure counts and warnings | Missing metadata, duplicate, invalid IDs, and failed embeddings tests |
| C12 Reproducible configuration | Pinned direct dependencies and full installed snapshot, documented keys/models/timeouts, optional live diagnostic | pip check, actual offline SDK constructors/parser, full local suite; fresh CI and live provider validation pending |

## Features

| Requirement | Implementation | Verification |
| --- | --- | --- |
| F01 Adaptive conversation | Mood and desired story experience guide dialogue; no required genres, review, or turn cutoff | Emotion-only chat, direct suggestions, continued chat after results, and outage AppTest checks |
| F02 Editable profile | Batched form with watched aliases, genres/exclusions, episode/rating limits; explicit confirmation before search | Form validation, filtering, source card, and feedback AppTests |
| F03 Cached knowledge | Public metadata/provenance cache and content/model keyed catalogue vector cache with TTL/refresh | Cache reuse/expiry/corruption/content invalidation tests |
| F04 Retrieval/filter/rank | Known filters, exact watched aliases/IDs, strict missing-metadata rules, semantic ranking with explicit lexical fallback | Constraint fixtures and AniList/engine tests |
| F05 Grounded explanations | One AI batch chooses supported sentences; metadata evidence separate from inferred preference fit; sources per card | Invalid prose/IDs, provider failure, word bounds and source coverage tests |
| F06 Follow-ups/refinement | Retrieved-record episode/genre/theme/why answers; alternatives preserve exclusions without repeating initial chat | Follow-up, refinement tests and actual UI interaction |
| F07 Feedback | Relevant/Not for me/Already watched plus reason; session-only adjustment and ID exclusion | Feedback filtering and AppTest watched exclusion |
| F08 Progress/recovery | Collect, analyse, retrieve, rank, explain, present/error states; visible history and editing/retry | AppTest error/interruption/retry/result checks |

## Implemented defaults

- Conversation does not require readiness or profile review. Extraction happens when suggestions are requested or optional review is opened. Emotional context and desired experience are separate from explicit genre preferences; assistant suggestions count only when the user accepts them.
- Coverage is the public AniList anime catalogue (switched from Jikan on 7 November 2026 after the Jikan API host became unreachable), with up to 50 unique candidates per search keyed by MyAnimeList ID and a 24-hour local cache. Refresh never promises to bypass AniList's own cache. Avant Garde, Award Winning and Team Sports have no AniList equivalent and are no longer offered as filters.
- Supported exclusions are known catalogue genre/theme/demographic labels, exact watched aliases/IDs, episode bounds, and optional rating bounds. Unknown episode metadata fails explicit limits; unknown hard exclusions require clarification.
- Gemini and Mistral clients are supported; model names and deadlines are configurable. Existing installed Gemini integration is used because installing a newer SDK was not approved. Only offline construction/parser compatibility has been verified.
- Feedback stays in the Streamlit session. No accounts, feedback database, fine-tuning, trained personalization, or cross-session learning were added.
- Offline numerical targets and comparison methods are defined in the evaluation module and report. Production human ratings, latency, tokens/cost, and visual ordering require live measurement.

## Repository deliverables and outstanding evidence

README, safe environment/secrets templates, ignore rules, pinned dependencies, unit tests, actual Streamlit AppTest integration checks, optional public-API diagnostic, CI configuration, a saved baseline, fixed evaluation profiles, and the evaluation report are present.

No GitHub remote is configured, so repository URLs and hosted CI results cannot be supplied. The owner approved starting the local server on port 8501. Real startup/profile screenshots and a browser report are present; the app has not been published. Authenticated live verification, live recommendation screenshots, and streamed-message timing remain pending.
