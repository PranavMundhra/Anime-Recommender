"""Optional live AniList check; prints no credentials or personal preferences."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anime_recommender.services.anilist import AniListAPI


def main():
    # A diagnostic must test the network, never report cache recovery as success.
    result = AniListAPI(retries=0, max_pages=1, deadline_seconds=15, cache_ttl=0).search({}, budget=1, refresh=True)
    print(json.dumps({"status": result.status, "error_kind": result.error_kind,
                      "message": result.error, "retrieved_count": result.retrieved_count,
                      "first_title": result.records[0]["title"] if result.records else None,
                      "warnings": result.warnings}, indent=2))
    return 0 if result.records else 1


if __name__ == "__main__":
    raise SystemExit(main())
