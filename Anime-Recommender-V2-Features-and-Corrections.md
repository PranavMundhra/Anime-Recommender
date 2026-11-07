# Anime Recommender System V2 — Features and Corrections

Version: 0.1  
Prepared: 14 October 2026  
Status: V2 application implemented locally; deterministic checks pass. See `docs/v2-implementation.md` for requirement coverage and outstanding live/browser/human validation.  
Baseline: supplied `Pasted text(2).txt` and the discussion of chat behaviour and RAG

## 1. V2 objective

Deliver a responsive conversational recommender that collects useful preferences, retrieves appropriate anime, and explains recommendations using identifiable source information.

The first priority is reliable conversation and recommendation behaviour. The second is stronger retrieval and grounded generation. New capabilities below are proposals for V2, not claims that they already exist.

## 2. Existing capabilities to retain

- Streamlit chat interface, history, profile summary, and recommendation cards.
- Gemini/Mistral provider selection through an abstraction.
- Six stage-specific conversation prompts.
- JSON profile extraction and Pydantic parsing.
- Default analysis profiles on failures.
- Jikan genre-filtered candidate retrieval.
- `all-MiniLM-L6-v2` embeddings and cosine-similarity ranking.
- Selection of up to five recommendations.
- LLM-generated reasons using each selected anime's retrieved metadata.

The explanation step already uses retrieved metadata as generation context: a basic retrieval-augmented generation pattern. V2 strengthens its retrieval, grounding, sources, and evaluation rather than adding a RAG label alone.

## 3. Corrections

Priority definitions: **P0** = core reliability; **P1** = recommendation correctness and evidence; **P2** = refinement. These priorities are proposed sequencing, not a deadline commitment.

### C01 — Display the user message before assistant generation [P0]

**Reported symptom:** The user message and assistant response appear together. The supplied code uses a blocking LLM call; the exact visual timing needs reproduction in the running app.

**Correction:** Persist and render the submitted message first, show a visible assistant processing state, and then display incoming response chunks. Use an explicit pending-turn state if a rerun is necessary to achieve the required ordering.

**Acceptance:** With an intentionally delayed response, the user message is visible during the wait; a processing indicator appears; the assistant answer subsequently streams or visibly arrives after that state.

### C02 — Remove duplicated user messages in model requests [P0]

**Observed:** The UI stores the latest user message in history before calling `chat_with_llm()`. That helper reconstructs history and appends the same user message again.

**Correction:** Choose one authoritative history-building path. Each submitted message appears exactly once in the model request.

**Acceptance:** Inspect the request payload over multiple turns; no turn contains two copies of the latest user message.

### C03 — Stream and commit assistant responses safely [P0]

**Observed:** `llm.invoke()` returns the complete response before display.

**Correction:** Use provider-supported streaming and Streamlit's stream display. Keep partially displayed output separate from committed conversation history. Mark interrupted responses as incomplete and offer retry; do not store them as successful answers.

**Acceptance:** Successful output is saved once; rerendering does not repeat generation; a stream failure produces a recoverable state without duplicate messages. A nonstreaming provider still shows a visible waiting state.

### C04 — Protect turns against reruns and overlapping submissions [P0]

**Correction:** Track a turn identifier and processing state. Prevent additional submissions and provider changes during an active turn. Advance the successful conversation count only after completion; an API failure must not silently consume a successful turn.

**Acceptance:** Rerunning or interacting during generation does not duplicate the call or advance the conversation twice. Retry refers to the pending turn.

### C05 — Fix error paths and distinguish defaults from personalization [P0]

**Observed:** `generate_recommendation_reason()` assigns `genres` after provider initialization, but its exception handler references `genres`. Initialization failure can therefore cause a second exception. Some fallback wording also promises a perfect match without evidence.

**Correction:** Prepare fallback data before risky operations. Use neutral, metadata-supported fallback text. Mark default-profile recommendations as defaults and allow users to correct their profile.

**Acceptance:** Provider initialization, JSON parsing, schema validation, and explanation failures each return a usable, honestly labelled state without a secondary exception.

### C06 — Bound external calls and handle empty results [P0]

**Observed:** Jikan requests lack explicit timeouts. Non-success responses can become empty lists. Recommendation generation can end with an empty result while the UI still enters the completed state.

**Correction:** Add request deadlines, explicit response handling, and bounded retries for appropriate transient failures. Distinguish API failure from a successful search with no matches. Offer retry or explicit filter relaxation, preserving the conversation.

**Acceptance:** A timeout, rate-limit response, invalid response, and genuine zero-match search show distinct recovery messages. Never silently relax explicit exclusions.

### C07 — Verify candidate limits and implement pagination [P1]

**Observed:** The code requests a limit of 50 in one request; retrieval of 50 actual candidates is not established.

**Correction:** Verify the current Jikan contract, paginate within documented limits, deduplicate by anime ID, and report the actual retrieved count. Stop at a configurable budget.

**Acceptance:** Tests cover multiple pages, duplicates, the final page, partial retrieval, and empty results. UI and project claims use actual counts.

### C08 — Enforce profile fields and explicit constraints [P1]

**Observed:** `mood_level` is a string and `confidence` is a float with descriptive text; allowed moods and confidence bounds are not enforced by those descriptions.

**Correction:** Validate allowed values, bounds, and field shapes; normalize genre labels. Separate preferences from hard exclusions. Treat missing information as unknown rather than an invented preference.

**Acceptance:** Invalid fields are rejected or repaired visibly. A malformed profile cannot override an explicit user exclusion. The LLM's confidence is never presented as calibrated certainty.

### C09 — Correct misleading score and explanation presentation [P1]

**Observed:** Cosine similarity is displayed as a percentage “Match Score.” It is not a probability that the user will enjoy the title. The explanation prompt requests 25 words, while truncation allows 30.

**Correction:** Label the value “Semantic similarity” or omit it from the main card. Distinguish catalogue ratings from personalized ranking. Apply a consistent explanation-length policy and remove absolute claims such as “perfectly matches.”

**Acceptance:** No enjoyment-probability interpretation is implied. Explanations remain grammatical and distinguish supported facts from preference-based suggestions.

### C10 — Reset only the current recommendation session [P1]

**Observed:** Starting a new session deletes every Streamlit session-state key.

**Correction:** Reset conversation, pending turn, profile, and results intentionally. Preserve unrelated application settings, including provider choice where appropriate.

**Acceptance:** A fresh session has no old recommendations or pending work; retained preferences/settings match the documented behaviour.

### C11 — Make similarity failures and metadata gaps observable [P1]

**Observed:** Candidate embedding failures are silently skipped. Missing metadata can be inconsistent across candidate records.

**Correction:** Normalize missing values; record aggregate skipped-candidate counts and failure categories. Show a concise message when evidence is insufficient, without exposing secrets or full conversation logs.

**Acceptance:** Missing synopsis, images, episodes, and genres do not crash ranking or rendering. Fewer than five eligible candidates produces fewer than five honest recommendations.

### C12 — Establish a reproducible environment [P0]

**Correction:** Verify provider/model availability, compatible LangChain and Streamlit packages, and required credentials. Pin a tested environment. Keep provider model names configurable and document dependencies and setup.

**Acceptance:** A fresh installation can start the app and run mocked tests. Missing configuration produces an actionable message. Live API checks are separately labelled.

## 4. V2 features

### F01 — Adaptive preference collection [P1]

Replace unconditional six-turn completion with preference readiness and a configurable turn cap.

- Collect viewing intent, liked/disliked genres or themes, watched titles, and relevant constraints.
- Make emotional disclosure optional; the user can describe the viewing experience they want instead.
- Ask targeted questions for missing information and avoid repeating answered questions.
- Offer “Recommend now” after a usable profile exists.
- At the turn cap, show unresolved fields and let the user proceed or edit.

**Acceptance:** A sufficiently detailed first message can shorten the conversation; a vague response triggers a relevant follow-up. Required fields and the cap must be chosen before implementation.

### F02 — Editable profile and exclusions [P1]

Before retrieval, show the extracted profile and let users correct it. Support already-watched titles, excluded genres, desired length, and other constraints only where reliable metadata is available.

**Acceptance:** Profile edits change retrieval. Explicit exclusions are checked again before displaying results. Do not assert that an unknown tone/content attribute is safe or suitable.

### F03 — Cached anime knowledge base [P1]

Maintain normalized records containing anime ID, title, synopsis, genres, themes, episode count, catalogue rating, source URL, and retrieval date. Cache reusable embeddings and metadata; keep update policy configurable.

Use one title-level record initially. Chunking is optional for longer, separately sourced material. Choose storage based on catalogue size; a dedicated vector database is not mandatory.

**Acceptance:** Repeated queries reuse eligible cached data, records retain provenance, and refresh does not create duplicate titles. Retrieved catalogue content is treated as data, not instructions to the assistant.

### F04 — Retrieval, filtering, and ranking [P1]

Use the confirmed user profile to retrieve candidates, enforce structured constraints, and rank eligible titles semantically. Exact title matching should assist already-watched exclusion. Preserve catalogue coverage rather than silently defaulting every unknown genre to action.

The existing minimum rating of 7.0 becomes a configurable catalogue-quality preference. It must not be mistaken for personalized relevance.

**Acceptance:** Known exclusions never appear in results; unknown genres prompt clarification or transparent handling. Incompatible constraints produce an explanation and optional relaxation rather than invented matches.

### F05 — Grounded recommendation explanations [P1]

Provide the generator with selected records and the user profile. Ask it to explain the match using that evidence, avoiding plot claims, tone guarantees, or facts absent from the source.

- Include a source link on each card.
- Keep factual evidence separate from inferred preference alignment.
- Use a neutral fallback when generation fails.
- Avoid unnecessary spoilers in explanations.

**Acceptance:** Every recommendation has an identifiable record/source. Sample explanations are checked for unsupported statements. Sparse source data produces a qualified explanation.

### F06 — Follow-up questions and recommendation refinement [P2]

Support questions such as “Why this one?”, “How many episodes?”, and “What else has similar themes?” Retrieve relevant records before answering factual questions. Users can request different results without repeating the entire initial conversation.

**Acceptance:** The system resolves the referenced title, answers from retrieved evidence, and states when information is unavailable. Refinement preserves earlier exclusions unless the user changes them.

### F07 — Feedback [P2]

Offer “Relevant”, “Not for me”, and “Already watched”. Allow an optional reason. Use feedback to adjust the current session; cross-session learning or persistence is a later explicit decision.

**Acceptance:** Already-watched feedback excludes that title from subsequent results. No claim of a trained personalization model is made without implementing and evaluating one.

### F08 — Clear progress and recovery UI [P0/P1]

Show separate states for collecting preferences, analysing the profile, retrieving candidates, ranking, generating explanations, and presenting results. Keep chat history visible after recommendation generation. Display source links and useful metadata; gracefully omit unavailable fields.

**Acceptance:** The user can identify the active stage, correct their profile, recover from failure, and understand why fewer results are shown. The interface does not claim completed recommendations while still processing or after failure.

## 5. Proposed processing flow

1. Receive and display a user message once.
2. Stream the assistant response and commit the completed turn.
3. Update the preference profile and decide whether a follow-up is needed.
4. Let the user review or edit the ready profile.
5. Retrieve candidate records and enforce constraints.
6. Rank candidates and select up to five supported results.
7. Generate grounded reasons using retrieved evidence.
8. Present recommendations, source links, and refinement options.

Keep conversation, profile extraction, retrieval, ranking, and explanation generation as distinct components so each can be tested independently.

## 6. Evaluation and tests

### Reliability tests

- Latest user message appears once in the model request.
- User message remains visible while a delayed response is pending.
- Interrupted streams, retries, reruns, and duplicate submissions preserve correct history.
- Provider initialization and all fallback branches do not produce secondary exceptions.
- Invalid JSON/schema, missing metadata, zero results, timeout, and rate limiting have recovery paths.
- Pagination and deduplication produce the expected candidate count.
- Constraints and watched-title exclusion hold after retrieval and refinement.
- Reset clears only the documented session fields.

### Recommendation and RAG evaluation

Build a fixed set of representative profiles: brief inputs, detailed preferences, exclusions, conflicting constraints, obscure titles, missing metadata, and follow-up questions.

Compare V1 with V2 on the same profiles and a documented catalogue snapshot:

| Measure | What to examine |
| --- | --- |
| Relevance | Human-rated fit of the top recommendations |
| Constraint adherence | Violations of explicit exclusions/limits |
| Groundedness | Factual explanation claims supported by retrieved records |
| Retrieval quality | Whether suitable candidates appear in the retrieved set |
| Diversity | Redundant recommendations versus useful alternatives |
| Latency | Time to visible user message, first assistant token, and final results |
| Cost | LLM calls/tokens and external requests per session |
| Failure recovery | Whether failures preserve a usable session |

Define scoring rubrics and numerical release targets before running the comparison. Do not publish unmeasured improvements or treat similarity/self-reported confidence as quality evaluation.

## 7. Implementation sequence

| Phase | Scope | Exit condition |
| --- | --- | --- |
| A — Conversation and reliability | C01–C06, C12, essential F08 states | Correct message order, no duplication, bounded calls, safe recovery |
| B — Recommendation and RAG | C07–C11, F01–F05 | Editable profile, constraint-aware retrieval, grounded explanations, evidence links |
| C — Refinement and evaluation | F06–F07, comparative evaluation | Follow-ups tested and V1/V2 comparison documented |

Feedback persistence, accounts, agentic browsing, fine-tuning, and large infrastructure are outside the initial V2 requirement. Add them only for a demonstrated product need.

## 8. Repository and portfolio deliverables

- README with architecture, current scope, and supported providers.
- Tested setup instructions and pinned dependencies.
- Example environment configuration containing placeholders only.
- Unit tests and mocked integration tests; optional live checks documented separately.
- Screenshots or a short recording showing message ordering and recommendations.
- A small evaluation report with methods, results, and limitations.
- Direct repository and source links for the portfolio case study.

Update the website and resume only after the corresponding V2 capabilities are implemented and verified. Until then, describe them as planned.

## 9. Decisions to confirm before implementation

- [x] Readiness: one viewing intent, preferred genre, or theme; maximum six successful turns, configurable.
- [x] Public Jikan catalogue: 50 unique candidate budget; 24-hour cache and explicit refresh.
- [x] Known catalogue labels, watched aliases/IDs, episode bounds, and optional minimum rating; unknown bounds fail closed.
- [x] Gemini/Mistral configurable model names; offline SDK compatibility tested, live account access pending.
- [x] Feedback remains session-only.
- [x] Numerical software targets and reproducible evaluation defined in `docs/evaluation.md`; human/live metrics remain unmeasured.

## 10. Source and verification notes

- Supplied `Pasted text(2).txt`: static inspection of the current implementation; no live model/API execution.
- Owner feedback: message-display issue and interest in RAG.
- [Streamlit streaming](https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream).
- [Streamlit conversational-app guidance](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps).
- [LangChain retrieval and RAG guidance](https://docs.langchain.com/oss/python/deepagents/retrieval).

Provider versions and the Jikan API contract must be checked against current documentation during implementation. This specification does not promise compatibility with the older configuration in the supplied code.

## Change log

| Date | Change |
| --- | --- |
| 2026-10-14 | Initial V2 features, corrections, acceptance criteria, and implementation sequence. |
| 2026-10-14 | Implemented V2 modules, native Streamlit UI, bounded retrieval, safe streaming/retry, editable profiles, cached ranking, grounded explanations, follow-ups, feedback, and offline evaluation. Live/browser validation remains outstanding. |
