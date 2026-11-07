"""Source-neutral catalogue records, hard-constraint filtering and outage recovery.

Every catalogue client normalizes its responses to one record shape keyed by
MyAnimeList ID, so ranking, display and feedback never depend on the source.

The small built-in starter collection is editorial, not an API response or rating
snapshot. Descriptions and experience tags are our own recommendation notes. MAL links
identify the titles; no live rating, poster, episode count or retrieval timestamp is claimed.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re
from typing import Any
from urllib.parse import urlparse

# Canonical label spellings. Source-specific names (e.g. AniList's "Thriller" or
# "Boys' Love") map onto the labels users see and profiles store.
ALIASES = {"sci fi": "sci-fi", "science fiction": "sci-fi", "scifi": "sci-fi",
           "shounen": "shonen", "shoujo": "shojo", "thriller": "suspense",
           "police": "detective", "slice-of-life": "slice of life",
           "boys' love": "boys love", "yuri": "girls love", "food": "gourmet",
           "cars": "racing", "female harem": "harem", "male harem": "harem",
           "primarily adult cast": "adult cast", "work": "workplace"}

# Failures where a saved page or the editorial starter collection may stand in.
RECOVERABLE_ERRORS = frozenset({"connection", "timeout", "upstream", "rate_limit", "invalid_response"})


def normalize_label(value: str) -> str:
    value = " ".join(str(value).strip().casefold().split())
    return ALIASES.get(value, value)


def normalize_title(value: str) -> str:
    return re.sub(r"[^\w]+", "", str(value).casefold())


@dataclass
class SearchResult:
    records: list[dict] = field(default_factory=list)
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None
    retrieved_count: int = 0
    cache_hit: bool = False


class CatalogueError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def normalize_record(raw: dict, retrieved_at: str | None = None) -> dict:
    """Preserve source fields and normalize gaps without inventing metadata."""
    record = dict(raw)
    anime_id = record.get("mal_id")
    if isinstance(anime_id, bool) or not isinstance(anime_id, int) or anime_id <= 0:
        raise ValueError("Anime record has no valid MAL ID.")
    record["title"] = str(record.get("title") or record.get("title_english") or f"Anime {anime_id}")
    record["synopsis"] = str(record.get("synopsis") or "")
    for key in ("genres", "themes", "demographics"):
        items = record.get(key)
        record[key] = [item for item in (items if isinstance(items, list) else [])
                       if isinstance(item, dict) and isinstance(item.get("name"), str)]
    record["images"] = record.get("images") if isinstance(record.get("images"), dict) else {}
    for kind in ("jpg", "webp"):
        if kind in record["images"] and not isinstance(record["images"][kind], dict):
            record["images"][kind] = {}
    for key in ("titles", "title_synonyms"):
        if not isinstance(record.get(key), list):
            record[key] = []
    episodes = record.get("episodes")
    record["episodes"] = episodes if isinstance(episodes, int) and not isinstance(episodes, bool) and episodes > 0 else None
    score = record.get("score")
    record["score"] = score if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 10 else None
    source = str(record.get("source_url") or "")
    if urlparse(source).scheme != "https":
        source = f"https://myanimelist.net/anime/{anime_id}"
    record["source_url"] = record["url"] = source
    record["retrieved_at"] = retrieved_at or datetime.now(timezone.utc).isoformat()
    return record


def profile_dict(profile: Any) -> dict:
    if isinstance(profile, dict):
        return profile
    if hasattr(profile, "model_dump"):
        return profile.model_dump()
    if hasattr(profile, "dict"):
        return profile.dict()
    raise TypeError("Profile must be a mapping or a validated profile model.")


def record_is_eligible(record: dict, profile: Any) -> bool:
    """Apply hard exclusions to all known genre categories and title aliases."""
    profile = profile_dict(profile)
    excluded = {normalize_label(x) for x in profile.get("excluded_genres", [])}
    labels = {normalize_label(x.get("name") if isinstance(x, dict) else x)
              for key in ("genres", "themes", "demographics") for x in record.get(key, [])
              if (isinstance(x, dict) and x.get("name")) or isinstance(x, str)}
    if labels & excluded or str(record.get("mal_id")) in {str(x) for x in profile.get("watched_ids", [])}:
        return False
    watched = {normalize_title(x) for x in profile.get("watched_titles", []) if str(x).strip()}
    titles = [record.get(key) for key in ("title", "title_english", "title_japanese")]
    titles += list(record.get("title_synonyms") or [])
    titles += list(record.get("aliases") or [])
    titles += [x.get("title") for x in (record.get("titles") or []) if isinstance(x, dict)]
    if watched & {normalize_title(x) for x in titles if x}:
        return False
    for field_name, comparison in (("min_episodes", lambda x, n: x >= n),
                                   ("max_episodes", lambda x, n: x <= n)):
        bound = profile.get(field_name)
        if bound is not None and (record.get("episodes") is None or not comparison(record["episodes"], bound)):
            return False
    minimum = profile.get("min_score", 7.0)
    if minimum is not None and (record.get("score") is None or record["score"] < minimum):
        return False
    return True


# id, title, broad editorial genres, original editorial description, experience tags
STARTER_TITLES = (
    (22789, 'Barakamon', ['Comedy', 'Slice of Life'],
     'A calligrapher starts again in an island community.',
     ['gentle', 'comforting', 'warmth', 'friendship', 'personal growth', 'hopeful']),
    (4081, "Natsume's Book of Friends", ['Supernatural', 'Slice of Life'],
     'A young man builds connections with people and spirits.',
     ['gentle', 'connection', 'supportive friendship', 'reflective', 'loneliness', 'warmth']),
    (34798, 'Laid-Back Camp', ['Slice of Life'],
     'Camping and time outdoors bring a group of friends together.',
     ['calm', 'gentle', 'comforting', 'friendship', 'warmth', 'relaxing']),
    (50265, 'Spy x Family', ['Action', 'Comedy'],
     'A spy forms an unconventional family while keeping his mission secret.',
     ['funny', 'playful', 'lighthearted', 'connection', 'adventure']),
    (20583, 'Haikyu!!', ['Sports'],
     'A school volleyball team learns to grow together through practice and competition.',
     ['motivating', 'uplifting', 'perseverance', 'supportive friendship', 'personal growth']),
    (22135, 'Ping Pong the Animation', ['Sports', 'Drama'],
     'Table tennis players confront competition, ambition and their own expectations.',
     ['reflective', 'personal growth', 'perseverance', 'emotional']),
    (31646, 'March Comes in Like a Lion', ['Drama'],
     'A young shogi player navigates isolation, competition and relationships.',
     ['emotional', 'reflective', 'connection', 'personal growth']),
    (1, 'Cowboy Bebop', ['Action', 'Sci-Fi'],
     'Bounty hunters travel through space while their pasts catch up with them.',
     ['adventure', 'absorbing', 'action', 'reflective']),
    (52991, 'Frieren: Beyond Journey\'s End', ['Adventure', 'Fantasy', 'Drama'],
     'An elf revisits the meaning of time and companionship after a long journey.',
     ['reflective', 'loss', 'connection', 'discovery', 'fantasy', 'hopeful']),
)


def starter_records(profile):
    records = []
    for anime_id, title, genres, synopsis, tags in STARTER_TITLES:
        record = dict(mal_id=anime_id, title=title, genres=[{'name': g} for g in genres],
                      themes=[], demographics=[], editorial_tags=tags, synopsis=synopsis,
                      score=None, episodes=None, retrieved_at=None, images={},
                      source_url=f'https://myanimelist.net/anime/{anime_id}',
                      catalogue_origin='editorial starter collection')
        if record_is_eligible(record, profile):
            records.append(record)
    return records


def resilient_search(client, profile, **kwargs):
    result = client.search(profile, **kwargs)
    # Empty results and invalid constraints are not connectivity failures.
    if result.records or result.error_kind not in RECOVERABLE_ERRORS:
        return result
    records = starter_records(profile)
    if not records:
        result.warnings.append('The offline starter collection cannot satisfy your constraints. They have been preserved; retry the live catalogue.')
        return result
    return SearchResult(records=records, status='fallback', error_kind=result.error_kind,
                        error=result.error, retrieved_count=len(records), warnings=result.warnings + [
                            'The live catalogue is unavailable. These suggestions come from a small editorial starter collection, '
                            'not a live search. Ratings, episode counts and posters are unavailable.'])
