# Anime Recommender V2

A conversational Streamlit anime companion that listens to how you feel and finds stories for the experience you need. Genre knowledge and preference forms are optional.

## Setup

The local checks use Python 3.11 on Windows. Create the environment and install the pinned direct dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.lock.txt` records the entire tested local environment, including previously installed packages. Use it when reproducing that exact environment. The direct requirements file is the supported app setup; transitive versions may vary unless you use the full snapshot.

Copy `.env.example` to `.env` if it does not already exist. Add `GOOGLE_API_KEY` or `MISTRAL_API_KEY` for the provider you select. Streamlit deployment secrets are supported through `.streamlit/secrets.toml` or the hosting service's secrets UI. Existing environment variables take precedence over `.env`; Streamlit secrets supply keys when the environment value is blank.

The model names are configurable with `GEMINI_MODEL` and `MISTRAL_MODEL`. Retired Gemini 1.5 names are no longer hard-coded. Model access depends on your account; the test suite checks client construction without making billable provider calls. Restart the app after changing keys or model settings so cached clients reload.

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py
```

## Using the app

1. Talk about your day or what is on your mind, or tap a conversation starter such as *Long day, need to unwind*. The companion listens like a friend: it acknowledges what you said and gently explores whether you want comfort, distraction, motivation, or a story that sits with the feeling. Sharing feelings is optional.
2. Keep talking for as long as you like. The companion does not push suggestions in its first reply; once it has a sense of what you need, it offers to find a few shows.
3. Say yes to that offer ("sounds good", "go for it") or ask directly ("can you suggest something comforting?"). The app derives a tentative story focus from the conversation and searches. No genre selection or form is required.
4. Continue the conversation after suggestions: ask about a title, explain why a suggestion feels wrong, or request a different direction. With an AI key, these messages use the conversational model with the chat history and supplied recommendation facts.
5. Use **Choose my own filters** in the sidebar (or **Edit preferences** under results) only when you want explicit control over exclusions, watched titles, episode limits, or minimum ratings. Unknown episode counts do not satisfy explicit limits. Feedback affects this session; it does not train a model or persist across sessions.

Without an AI key, the manual profile editor, catalogue search, lexical/semantic ranking, neutral explanations, and catalogue-based follow-ups remain available. Failed conversations can be retried without duplicating the user message or consuming a successful turn. Search failures keep the profile and history intact; filters are relaxed only through explicit edits.

## Architecture

```text
main.py                         Native Streamlit UI and pipeline coordination
anime_recommender/
  config.py                     Environment/secrets and bounded settings
  conversation.py               Pending turns, streaming, retry, scoped reset
  profile.py                    Validated profiles and extraction safety net
  recommendations.py            Filtering, cached embeddings, ranking, explanations
  evaluation.py                 Offline V1/V2 comparison
  ui.py                         Forms, feedback, and interaction callbacks
  services/
    llm.py                      Gemini integration and native Mistral adapter
    anilist.py                  AniList GraphQL client: retry, pagination, public catalogue cache
    catalogue.py                Shared record shape, hard-constraint filter, outage recovery
tests/                          Unit and actual Streamlit AppTest checks
scripts/check_catalogue.py      Optional live AniList connectivity diagnostic
docs/v2-implementation.md       Requirement mapping and validation limits
docs/evaluation.md              Comparison methods and results
.claude/agents/                 structure-keeper: repo tidiness agent (see below)
```

No separate API server or local dataset folder is required. The anime catalogue comes from the public AniList GraphQL API, which requires no key. The sentence model `all-MiniLM-L6-v2` loads lazily and downloads on first semantic search; use the semantic-ranking toggle to avoid that download when needed. If embedding loading or encoding fails, the UI identifies lexical word-overlap ranking.

Metadata and catalogue vectors are cached in ignored `.cache/` files. User conversations and personal profile vectors are not persisted there. Candidate encoding is batched and reused by content/model hash; preference vectors remain per request. Refresh forces new catalogue requests, but AniList may still serve its own upstream cache.

Explanations use one batched AI call to choose validated, metadata-supported sentences for up to five results. The service rejects arbitrary factual prose, unknown IDs, and unsupported reason text; failure produces neutral evidence-based templates. Semantic similarity is a numeric ranking measure, not an enjoyment probability.

## Configuration

Defaults are in `.env.example`: 50 unique candidate budget, 24-hour public-data cache, bounded catalogue deadlines/retries (`CATALOGUE_TIMEOUT`, `CATALOGUE_RETRIES`), and a 30-second provider timeout. Live catalogue retrieval has an 18-second deadline before recovery. Conversation has no six-turn cutoff; the old `MAX_CONVERSATION_TURNS` setting is no longer used. Search pages contain at most 50 results and are deduplicated by MyAnimeList ID; AniList titles without a MAL ID are skipped and reported. The displayed candidate count is the actual retrieved count before local filtering. Fewer eligible titles means fewer cards.

The desired story experience drives matching. Feeling sad does not itself become a request for sad shows. Genres remain optional; when none are provided, the story focus supplies a broad candidate-retrieval hint. Interpretation is tentative, and users can correct it through conversation. The companion does not diagnose conditions or promise that anime will treat them.

Known genre, theme, and demographic exclusions are enforced both after retrieval and immediately before display. Unknown preferred genre labels are reported and handled transparently, never changed to Action. Unknown hard exclusions require clarification rather than being silently ignored. Genre, theme and demographic labels are mapped to AniList genres or tags (for example Suspense → Thriller, Shonen → Shounen); excluding Hentai or Erotica also removes adult-flagged titles. Tags marked as spoilers are never used.

`.env`, `.streamlit/secrets.toml`, environments, local model caches, and generated cache reports are ignored by Git. Only placeholder templates are committed.

## Tests and evaluation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/evaluate.py --output docs/evaluation-results.json
```

Tests mock HTTP, billable providers, and model downloads while exercising actual Streamlit widgets, profiles, transactions, retries, pagination, caches, constraints, follow-ups, and feedback. Offline SDK constructor/parser checks use the installed provider libraries. GitHub Actions is configured for Windows/Python 3.11; the workflow itself has not run in GitHub because no remote is configured.

The comparison uses identical hand-authored metadata and deterministic keyword vectors. It measures software wiring and constraints, not real user relevance or production model quality. See the evaluation report for targets, results, and unmeasured metrics.

## Catalogue troubleshooting and live checks

```powershell
.\.venv\Scripts\python.exe scripts/check_catalogue.py
```

This requests one public catalogue record with retries disabled and a short deadline. It reports timeout, connection, rate-limit, HTTP, or response-shape failures separately. A connection timeout is a network/service issue; changing an API key will not fix it. Check whether `https://graphql.anilist.co` is reachable on your network and check proxy/firewall access if it does not. Do not bypass TLS verification.

During a transient outage, the app can reuse the exact requested catalogue page from disk for up to seven days, preserving its retrieval date and displaying a warning. If no usable page exists, a small built-in editorial starter collection keeps suggestions available. It is labeled as editorial, with original descriptions, broad genre/experience tags and MAL identity links; no live ratings, episode counts, posters or retrieval timestamps are invented. Explicit exclusions and watched titles remain enforced. If episode or rating constraints require unavailable metadata, the app preserves those constraints and offers retry instead of silently relaxing them. Empty searches and invalid constraints do not trigger this fallback. The diagnostic script disables caching so it measures actual connectivity.

The catalogue moved from Jikan to AniList on 7 November 2026 after the Jikan API host stopped accepting connections. Live AniList search, details and the diagnostic succeeded at that time. Gemini/Mistral responses, production latency/cost, streamed-message timing, and human recommendation ratings still require live validation.

Earlier startup/profile browser screenshots are historical and predate the emotion-focused conversation flow. See [browser review and screenshots](docs/browser-review.md). The revised flow is covered by Streamlit AppTest; authenticated provider wording still needs live validation.

## Keeping the repository tidy

`.claude/agents/structure-keeper.md` defines a Claude Code subagent that audits the folder layout, moves generated/duplicate/stale files to a backup instead of deleting them, keeps this README's architecture tree accurate, and reports (without applying) source-code reorganisations. Ask Claude Code to "run the structure-keeper agent" after a batch of changes or before committing.
