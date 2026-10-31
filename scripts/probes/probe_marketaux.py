"""Read-only Step 4a probe for the Marketaux News API.

Purpose:
- verify API access without exposing the API token;
- test historical-news availability;
- inspect returned article metadata and timestamps;
- assess whether Marketaux can support the proposed normalized schema;
- perform no sentiment scoring;
- write nothing to the project cache or filesystem.

This is investigation code only, not production ingestion code.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://api.marketaux.com/v1/news/all"

API_TOKEN_ENV = "MARKETAUX_API_TOKEN"

TIMEOUT_SECONDS = 30

# Keep the initial investigation deliberately small.
DEFAULT_LIMIT = 3


def require_token() -> str:
    token = os.getenv(API_TOKEN_ENV)

    if not token:
        raise RuntimeError(
            f"{API_TOKEN_ENV} is not configured"
        )

    return token


def request_json(
    params: dict[str, str],
) -> dict:
    token = require_token()

    # The token is added only at request time.
    # It is never printed.
    request_params = {
        **params,
        "api_token": token,
    }

    url = (
        f"{BASE_URL}?"
        f"{urlencode(request_params)}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": (
                "multi-asset-market-ml-"
                "step4a-probe/1.0"
            )
        },
    )

    try:
        with urlopen(
            request,
            timeout=TIMEOUT_SECONDS,
        ) as response:
            status = response.status
            payload = response.read().decode(
                "utf-8"
            )

    except HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        # Never print the requested URL because
        # it contains the API token.
        raise RuntimeError(
            f"HTTP {exc.code}: {body[:1000]}"
        ) from exc

    except (
        URLError,
        TimeoutError,
    ) as exc:
        raise RuntimeError(
            f"Network error: {exc}"
        ) from exc

    if status != 200:
        raise RuntimeError(
            f"Marketaux returned HTTP {status}"
        )

    try:
        return json.loads(payload)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Marketaux returned invalid JSON"
        ) from exc


def safe_article_summary(
    article: dict,
) -> dict:
    """Return only fields useful for Step 4a investigation."""

    entities = article.get("entities")

    if not isinstance(entities, list):
        entities = []

    entity_symbols = []

    for entity in entities[:10]:
        if not isinstance(entity, dict):
            continue

        symbol = entity.get("symbol")

        if symbol:
            entity_symbols.append(symbol)

    return {
        "uuid": article.get("uuid"),
        "title": article.get("title"),
        "description_present": bool(
            article.get("description")
        ),
        "snippet_present": bool(
            article.get("snippet")
        ),
        "url_present": bool(
            article.get("url")
        ),
        "language": article.get("language"),
        "published_at": article.get(
            "published_at"
        ),
        "source": article.get("source"),
        "entity_symbols": entity_symbols,
    }


def probe_historical_date(
    *,
    search: str,
    published_on: str,
) -> None:
    params = {
        "search": search,
        "published_on": published_on,
        "language": "en",
        "limit": str(DEFAULT_LIMIT),
    }

    print("=" * 72)
    print("MARKETAUX HISTORICAL PROBE")
    print("SEARCH:", search)
    print("PUBLISHED_ON:", published_on)
    print("LIMIT:", DEFAULT_LIMIT)

    retrieved_at = datetime.now(
        timezone.utc
    )

    print(
        "RETRIEVED_AT_UTC:",
        retrieved_at.isoformat(),
    )

    try:
        payload = request_json(params)

    except Exception as exc:
        print(
            "RESULT: ERROR"
        )
        print(
            "ERROR:",
            f"{type(exc).__name__}: {exc}",
        )
        return

    print("RESULT: OK")

    meta = payload.get("meta")

    if not isinstance(meta, dict):
        meta = {}

    data = payload.get("data")

    if not isinstance(data, list):
        data = []

    print(
        "META_FOUND:",
        meta.get("found"),
    )

    print(
        "META_RETURNED:",
        meta.get("returned"),
    )

    print(
        "META_LIMIT:",
        meta.get("limit"),
    )

    print(
        "ARTICLES_RECEIVED:",
        len(data),
    )

    if not data:
        print(
            "No articles returned."
        )
        return

    print("ARTICLE FIELD NAMES:")

    field_names = sorted(
        {
            key
            for article in data
            if isinstance(article, dict)
            for key in article.keys()
        }
    )

    print(
        ", ".join(field_names)
    )

    print("ARTICLE SUMMARIES:")

    for article in data:
        if not isinstance(article, dict):
            continue

        print(
            json.dumps(
                safe_article_summary(article),
                ensure_ascii=False,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Marketaux "
            "historical coverage probe."
        )
    )

    parser.add_argument(
        "--search",
        default="Apple Inc",
        help=(
            "Search expression for the "
            "bounded investigation."
        ),
    )

    parser.add_argument(
        "--date",
        default="2015-02-20",
        help=(
            "Historical publication date "
            "in YYYY-MM-DD format."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        "MARKETAUX STEP 4A "
        "READ-ONLY PROBE"
    )

    print(
        "API token configured:",
        "YES"
        if os.getenv(API_TOKEN_ENV)
        else "NO",
    )

    print(
        "Writes project/cache data: NO"
    )

    probe_historical_date(
        search=args.search,
        published_on=args.date,
    )


if __name__ == "__main__":
    main()
