"""Bounded AniList GraphQL retrieval with normalized provenance and public-data disk caching.

AniList needs no API key. No credentials or user conversations enter this cache.
Responses are converted to the app's catalogue record shape, keyed by MyAnimeList ID,
so watched IDs, feedback and the starter collection stay comparable across sources.
"""

from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from uuid import uuid4

import requests

from anime_recommender.services.catalogue import (
    RECOVERABLE_ERRORS, CatalogueError, SearchResult, normalize_label, normalize_record,
    profile_dict, record_is_eligible)

_GENRES = ("Action", "Adventure", "Comedy", "Drama", "Ecchi", "Fantasy", "Hentai", "Horror",
           "Mahou Shoujo", "Mecha", "Music", "Mystery", "Psychological", "Romance", "Sci-Fi",
           "Slice of Life", "Sports", "Supernatural")
_TAGS = ("School", "Historical", "Military", "Parody", "Detective", "Samurai", "Space", "Vampire",
         "Martial Arts", "Super Power", "Mythology", "Isekai", "Iyashikei", "Survival",
         "Seinen", "Josei", "Kids")
# Canonical app label -> (AniList filter kind, AniList name).
FILTERS = {
    **{normalize_label(name): ("genre", name) for name in _GENRES},
    **{normalize_label(name): ("tag", name) for name in _TAGS},
    "suspense": ("genre", "Thriller"), "shonen": ("tag", "Shounen"), "shojo": ("tag", "Shoujo"),
    "boys love": ("tag", "Boys' Love"), "girls love": ("tag", "Yuri"), "gourmet": ("tag", "Food"),
    "racing": ("tag", "Cars"), "harem": ("tag", "Female Harem"),
    "adult cast": ("tag", "Primarily Adult Cast"), "workplace": ("tag", "Work"),
}
# Excluding any of these also removes every adult-flagged title.
ADULT_LABELS = frozenset({"hentai", "erotica"})
# Tags below this relevance rank are too weak to describe a title's themes.
THEME_MIN_RANK = 60

_MEDIA_FIELDS = """
    id idMal title { romaji english native } synonyms description(asHtml: false)
    episodes averageScore isAdult genres siteUrl coverImage { large }
    tags { name rank category isMediaSpoiler isGeneralSpoiler }
"""
SEARCH_QUERY = """
query ($page: Int, $perPage: Int, $genreIn: [String], $genreNotIn: [String],
       $tagIn: [String], $tagNotIn: [String], $minScore: Int, $isAdult: Boolean,
       $episodesAbove: Int, $episodesBelow: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(type: ANIME, sort: [SCORE_DESC, POPULARITY_DESC], genre_in: $genreIn,
          genre_not_in: $genreNotIn, tag_in: $tagIn, tag_not_in: $tagNotIn,
          averageScore_greater: $minScore, isAdult: $isAdult,
          episodes_greater: $episodesAbove, episodes_lesser: $episodesBelow) {""" + _MEDIA_FIELDS + """}
  }
}"""
DETAILS_QUERY = """
query ($idMal: Int) {
  Media(idMal: $idMal, type: ANIME) {""" + _MEDIA_FIELDS + """}
}"""


def _payload_error(query: str, payload: Any) -> str | None:
    """Describe why an AniList payload has the wrong shape, or None when it is usable."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return "AniList returned an unexpected response shape. Retry this search."
    if query == SEARCH_QUERY:
        page = data.get("Page")
        if (not isinstance(page, dict) or not isinstance(page.get("media"), list)
                or not isinstance(page.get("pageInfo"), dict)):
            return "AniList returned an invalid catalogue page. Retry this search."
    elif not isinstance(data.get("Media"), dict):
        return "AniList returned invalid title metadata. Retry this search."
    return None


def _clean_text(value: Any) -> str:
    text = re.sub(r"<br\s*/?>", "\n", str(value or ""), flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def media_to_record(media: Any, retrieved_at: str | None = None) -> dict:
    """Convert one AniList Media object to a catalogue record; ValueError if unusable."""
    if not isinstance(media, dict):
        raise ValueError("AniList media entry is not an object.")
    titles = media.get("title") if isinstance(media.get("title"), dict) else {}
    tags = [t for t in media.get("tags") or [] if isinstance(t, dict) and isinstance(t.get("name"), str)
            and not t.get("isMediaSpoiler") and not t.get("isGeneralSpoiler")]
    score = media.get("averageScore")
    image = (media.get("coverImage") or {}).get("large") if isinstance(media.get("coverImage"), dict) else None
    anilist_id = media.get("id")
    site = media.get("siteUrl") if isinstance(media.get("siteUrl"), str) else ""
    if not site.startswith("https://anilist.co/") and isinstance(anilist_id, int):
        site = f"https://anilist.co/anime/{anilist_id}"
    return normalize_record({
        "mal_id": media.get("idMal"),
        "anilist_id": anilist_id,
        "title": titles.get("english") or titles.get("romaji") or titles.get("native"),
        "title_english": titles.get("english"),
        "title_japanese": titles.get("native"),
        "title_synonyms": [s for s in [titles.get("romaji"), *(media.get("synonyms") or [])]
                           if isinstance(s, str) and s],
        "synopsis": _clean_text(media.get("description")),
        "genres": [{"name": g} for g in media.get("genres") or [] if isinstance(g, str)],
        "themes": [{"name": t["name"]} for t in tags
                   if t.get("category") != "Demographic" and (t.get("rank") or 0) >= THEME_MIN_RANK],
        "demographics": [{"name": t["name"]} for t in tags if t.get("category") == "Demographic"],
        "episodes": media.get("episodes"),
        "score": score / 10 if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
        "images": {"jpg": {"image_url": image}} if isinstance(image, str) else {},
        "source_url": site,
    }, retrieved_at)


class AniListAPI:
    URL = "https://graphql.anilist.co"
    _rate_lock = threading.Lock()
    _next_request = 0.0

    def __init__(self, *, url: str | None = None, timeout: tuple[float, float] = (4, 10),
                 retries: int = 1, cache_dir: str | Path | None = None,
                 cache_ttl: float = 86400, max_pages: int = 4,
                 deadline_seconds: float = 40, session: Any = None,
                 request_interval: float = 0.7):
        self.url = url or self.URL
        self.timeout = timeout
        self.retries = min(max(int(retries), 0), 3)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[2] / ".cache" / "catalogue"
        self.cache_ttl = max(float(cache_ttl), 0)
        self.max_pages = min(max(int(max_pages), 1), 20)
        self.deadline_seconds = max(float(deadline_seconds), 1)
        self.request_interval = max(float(request_interval), 0)
        self.session = session or requests.Session()
        if session is None:
            self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json",
                                         "User-Agent": "AnimeRecommender/2 (AniList public catalogue)"})

    def _cache_key(self, query: str, variables: dict) -> str:
        return hashlib.sha256(json.dumps([self.url, query, variables], sort_keys=True).encode()).hexdigest()

    def _cache_read(self, key: str, *, allow_stale: bool = False) -> dict | None:
        if self.cache_ttl == 0:
            return None
        try:
            envelope = json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
            age = time.time() - float(envelope["created_at"])
            lifetime = max(self.cache_ttl, 7 * 86400) if allow_stale else self.cache_ttl
            if 0 <= age < lifetime and isinstance(envelope.get("payload"), dict) and isinstance(envelope.get("retrieved_at"), str):
                return envelope
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def _cache_write(self, key: str, envelope: dict) -> None:
        if self.cache_ttl == 0:
            return
        temporary = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_dir / f"{key}.{uuid4().hex}.tmp"
            temporary.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, self.cache_dir / f"{key}.json")
        except OSError:
            # Unwritable cache must not turn successful retrieval into failure.
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _request(self, query: str, variables: dict, refresh: bool, deadline: float) -> tuple[dict, str, bool, bool]:
        try:
            payload, retrieved_at, hit = self._request_live(query, variables, refresh, deadline)
            return payload, retrieved_at, hit, False
        except CatalogueError as error:
            cached = self._cache_read(self._cache_key(query, variables), allow_stale=True)
            if error.kind in RECOVERABLE_ERRORS and cached and _payload_error(query, cached["payload"]) is None:
                return cached["payload"], cached["retrieved_at"], True, True
            raise

    def _request_live(self, query: str, variables: dict, refresh: bool, deadline: float) -> tuple[dict, str, bool]:
        key = self._cache_key(query, variables)
        cached = None if refresh else self._cache_read(key)
        if cached and _payload_error(query, cached["payload"]) is None:
            return cached["payload"], cached["retrieved_at"], True
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CatalogueError("timeout", "AniList search reached its time limit. Retry or reduce the candidate budget.")
            with self._rate_lock:
                delay = max(type(self)._next_request - time.monotonic(), 0)
                if delay >= remaining:
                    raise CatalogueError("timeout", "AniList search reached its time limit while waiting to request data.")
                if delay:
                    time.sleep(delay)
                type(self)._next_request = time.monotonic() + self.request_interval
            remaining = deadline - time.monotonic()
            response = None
            try:
                connect, read = self.timeout
                timeout = (max(.001, min(connect, remaining / 2)), max(.001, min(read, remaining / 2)))
                response = self.session.post(self.url, json={"query": query, "variables": variables}, timeout=timeout)
                if response.status_code == 429:
                    error = CatalogueError("rate_limit", "AniList is rate limited. Wait briefly and retry; your preferences are preserved.")
                elif response.status_code in (500, 502, 503, 504):
                    error = CatalogueError("upstream", "AniList is temporarily unavailable. Retry this search.")
                elif response.status_code != 200:
                    raise CatalogueError("http", f"AniList returned HTTP {response.status_code}. Check the catalogue filters and retry.")
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise CatalogueError("invalid_response", "AniList returned invalid JSON. Retry this search.") from exc
                    problem = _payload_error(query, payload)
                    if problem:
                        raise CatalogueError("invalid_response", problem)
                    retrieved_at = datetime.now(timezone.utc).isoformat()
                    self._cache_write(key, {"created_at": time.time(), "retrieved_at": retrieved_at, "payload": payload})
                    return payload, retrieved_at, False
            except requests.Timeout:
                error = CatalogueError("timeout", "AniList connection timed out. Check connectivity and retry this search.")
            except requests.ConnectionError:
                error = CatalogueError("connection", "Cannot connect to AniList. Check network, proxy, or firewall settings and retry.")
            except requests.RequestException:
                error = CatalogueError("connection", "AniList request failed. Check connectivity and retry this search.")
            if attempt == self.retries:
                raise error
            try:
                retry_after = float(response.headers.get("Retry-After", 0)) if response is not None else 0
            except (ValueError, TypeError):
                retry_after = 0
            delay = max(.5 * (2 ** attempt), min(retry_after, 5))
            if delay >= deadline - time.monotonic():
                raise error
            time.sleep(delay)
        raise CatalogueError("connection", "AniList search failed. Retry this search.")

    def search(self, profile: Any, budget: int = 50, refresh: bool = False) -> SearchResult:
        """Retrieve up to budget unique candidates and enforce confirmed constraints.

        retrieved_count is unique source candidates before local filtering; len(records)
        is eligible count. Independent page and elapsed-time budgets bound pagination.
        """
        profile = profile_dict(profile)
        budget = min(max(int(budget), 1), 500)
        warnings = []
        preferred = [normalize_label(x) for x in profile.get("preferred_genres", profile.get("genres", []))]
        excluded = [normalize_label(x) for x in profile.get("excluded_genres", [])]
        unknown = sorted({x for x in preferred if x not in FILTERS})
        if unknown:
            warnings.append("Unrecognized preferences were not sent as genre filters: " + ", ".join(unknown) + ". Ranking still uses your profile.")
        unknown_exclusions = sorted({x for x in excluded if x not in FILTERS and x not in ADULT_LABELS})
        if unknown_exclusions:
            return SearchResult(status="error", warnings=warnings, error_kind="constraints",
                                error="Unknown excluded genres need clarification before retrieval: " + ", ".join(unknown_exclusions))
        wanted = sorted({x for x in preferred if x in FILTERS and x not in excluded})
        # Mood is not a genre preference. A single broad retrieval hint supplies
        # suitable candidates; the desired experience then drives local ranking.
        if not wanted and profile.get("recommendation_focus"):
            focus = profile["recommendation_focus"].casefold()
            hints = [("gentle|comfort|calm|warmth", "slice of life"),
                     ("funny|comedy|lighthearted", "comedy"),
                     ("perseverance|motivating", "sports"),
                     ("reflective|emotional|loss", "drama"),
                     ("adventure|discovery|fantasy", "adventure")]
            for pattern, label in hints:
                if label not in excluded and re.search(pattern, focus):
                    wanted = [label]
                    break
        variables = {"perPage": min(budget, 50)}
        for labels, suffix in ((wanted, "In"), (excluded, "NotIn")):
            for kind in ("genre", "tag"):
                names = sorted({FILTERS[x][1] for x in labels if x in FILTERS and FILTERS[x][0] == kind})
                if names:
                    variables[kind + suffix] = names
        if ADULT_LABELS & set(excluded):
            variables["isAdult"] = False
        minimum = profile.get("min_score", 7.0)
        if minimum is not None:
            # AniList scores are 0-100 and the filter is strictly greater-than.
            variables["minScore"] = math.ceil(minimum * 10) - 1
        # Episode bounds are strict on AniList too; local filtering still rejects unknown counts.
        if profile.get("min_episodes") is not None:
            variables["episodesAbove"] = int(profile["min_episodes"]) - 1
        if profile.get("max_episodes") is not None:
            variables["episodesBelow"] = int(profile["max_episodes"]) + 1
        records = {}
        cache_hits = 0
        requested_pages = 0
        deadline = time.monotonic() + self.deadline_seconds
        try:
            for page in range(1, self.max_pages + 1):
                payload, retrieved_at, hit, stale = self._request(SEARCH_QUERY, {**variables, "page": page}, refresh, deadline)
                if stale:
                    warnings.append("AniList is unavailable. Showing previously saved catalogue data (up to seven days old); retrieval dates are preserved.")
                requested_pages += 1
                cache_hits += int(hit)
                data = payload["data"]["Page"]["media"]
                skipped = 0
                for media in data:
                    try:
                        record = media_to_record(media, retrieved_at)
                    except ValueError:
                        skipped += 1
                        continue
                    records.setdefault(record["mal_id"], record)
                    if len(records) >= budget:
                        break
                if skipped:
                    warnings.append(f"Page {page}: skipped {skipped} catalogue records without a MyAnimeList ID or with malformed data.")
                if len(records) >= budget or not data or not payload["data"]["Page"]["pageInfo"].get("hasNextPage", False):
                    break
            else:
                warnings.append(f"Stopped after the configured {self.max_pages}-page limit.")
        except CatalogueError as exc:
            failure = exc
        else:
            failure = None
        eligible = [x for x in records.values() if record_is_eligible(x, profile)]
        if failure:
            return SearchResult(eligible, "partial" if eligible else "error", warnings,
                                str(failure), failure.kind, len(records), False)
        if len(eligible) < len(records):
            warnings.append(f"Removed {len(records) - len(eligible)} candidates using your exclusions or catalogue constraints.")
        if not eligible:
            warnings.append("No eligible titles matched. Adjust filters explicitly or retry; exclusions were preserved.")
        return SearchResult(records=eligible, status="ok" if eligible else "empty", warnings=warnings,
                            retrieved_count=len(records), cache_hit=requested_pages > 0 and cache_hits == requested_pages)

    def details(self, mal_id: int, refresh: bool = False) -> dict:
        payload, retrieved_at, _, _ = self._request(DETAILS_QUERY, {"idMal": int(mal_id)}, refresh,
                                                    time.monotonic() + self.deadline_seconds)
        try:
            return media_to_record(payload["data"]["Media"], retrieved_at)
        except ValueError as exc:
            raise CatalogueError("invalid_response", "AniList returned invalid title metadata.") from exc
