"""Read-only Step 4a probe for the GDELT DOC API.

This script:
- writes nothing to the project cache;
- performs bounded queries only;
- prints raw coverage evidence;
- does not perform sentiment scoring;
- does not modify production configuration.

It is an investigation utility, not production ingestion code.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Deliberately small, high-specificity queries for the initial probe.
QUERIES = {
    "AAPL": '"Apple Inc"',
    "MSFT": '"Microsoft Corporation"',
    "BTC_USD": 'bitcoin',
    "EURUSD": '"EUR/USD"',
    "GOLD": '"gold price"',
}

TIMEOUT_SECONDS = 30


def request_json(params: dict[str, str]) -> dict:
    url = f"{BASE_URL}?{urlencode(params)}"

    request = Request(
        url,
        headers={
            "User-Agent": (
                "multi-asset-market-ml-step4a-probe/1.0"
            )
        },
    )

    with urlopen(
        request,
        timeout=TIMEOUT_SECONDS,
    ) as response:
        status = response.status
        payload = response.read().decode("utf-8")

    if status != 200:
        raise RuntimeError(
            f"GDELT returned HTTP {status}"
        )

    return json.loads(payload)


def probe_recent_articles(
    asset: str,
    query: str,
) -> None:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=7)

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": "25",
        "sort": "datedesc",
        "startdatetime": start.strftime(
            "%Y%m%d%H%M%S"
        ),
        "enddatetime": now.strftime(
            "%Y%m%d%H%M%S"
        ),
    }

    print("=" * 72)
    print(f"ASSET: {asset}")
    print(f"QUERY: {query}")
    print(
        "WINDOW:",
        start.isoformat(),
        "->",
        now.isoformat(),
    )

    try:
        payload = request_json(params)
    except Exception as exc:
        print(
            "RESULT: ERROR",
            f"{type(exc).__name__}: {exc}",
        )
        return

    articles = payload.get("articles", [])

    print("RESULT: OK")
    print("ARTICLES RETURNED:", len(articles))

    if not articles:
        print("EARLIEST RETURNED: none")
        print("LATEST RETURNED: none")
        return

    dates = [
        article.get("seendate")
        for article in articles
        if article.get("seendate")
    ]

    print(
        "EARLIEST RETURNED:",
        min(dates) if dates else "unknown",
    )
    print(
        "LATEST RETURNED:",
        max(dates) if dates else "unknown",
    )

    print("SAMPLE:")
    for article in articles[:3]:
        print(
            json.dumps(
                {
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "domain": article.get("domain"),
                    "language": article.get(
                        "language"
                    ),
                    "sourcecountry": article.get(
                        "sourcecountry"
                    ),
                    "seendate": article.get(
                        "seendate"
                    ),
                },
                ensure_ascii=False,
            )
        )


def probe_historical_window(
    asset: str,
    query: str,
    *,
    start: datetime,
    end: datetime,
) -> None:
    """Test whether DOC API exposes a specific old window."""

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": "10",
        "sort": "datedesc",
        "startdatetime": start.strftime(
            "%Y%m%d%H%M%S"
        ),
        "enddatetime": end.strftime(
            "%Y%m%d%H%M%S"
        ),
    }

    print("=" * 72)
    print("HISTORICAL WINDOW TEST")
    print(f"ASSET: {asset}")
    print(f"QUERY: {query}")
    print(
        "REQUESTED WINDOW:",
        start.isoformat(),
        "->",
        end.isoformat(),
    )

    try:
        payload = request_json(params)
    except Exception as exc:
        print(
            "RESULT: ERROR",
            f"{type(exc).__name__}: {exc}",
        )
        return

    articles = payload.get("articles", [])

    print("RESULT: OK")
    print("ARTICLES RETURNED:", len(articles))

    for article in articles[:3]:
        print(
            json.dumps(
                {
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "domain": article.get("domain"),
                    "seendate": article.get(
                        "seendate"
                    ),
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    print("GDELT STEP 4A READ-ONLY PROBE")
    print(
        "retrieved_at_utc:",
        datetime.now(timezone.utc).isoformat(),
    )

    print("\nRECENT COVERAGE PROBE\n")

    for asset, query in QUERIES.items():
        probe_recent_articles(
            asset,
            query,
        )

    # One deliberately old request proves whether the
    # convenient DOC API itself can service our 2015
    # historical requirement. This is distinct from
    # GDELT's GKG archive/BigQuery availability.
    probe_historical_window(
        "AAPL",
        QUERIES["AAPL"],
        start=datetime(
            2015,
            3,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2015,
            3,
            8,
            23,
            59,
            59,
            tzinfo=timezone.utc,
        ),
    )


if __name__ == "__main__":
    main()