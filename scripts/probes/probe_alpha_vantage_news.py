"""Read-only Step 4a probe for Alpha Vantage NEWS_SENTIMENT.

Purpose:
- test actual NEWS_SENTIMENT entitlement;
- inspect historical access if permitted;
- inspect returned article/text/timestamp fields;
- expose no API secrets;
- perform no sentiment scoring;
- write nothing to project cache/config.

Investigation code only.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = (
    "https://www.alphavantage.co/query"
)

API_KEY_ENV = "ALPHA_VANTAGE_API_KEY"
TIMEOUT_SECONDS = 30


def require_key() -> str:
    key = os.getenv(API_KEY_ENV)

    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not configured"
        )

    return key


def request_json(
    params: dict[str, str],
) -> dict:
    key = require_key()

    request_params = {
        **params,
        "apikey": key,
    }

    url = (
        f"{BASE_URL}?"
        f"{urlencode(request_params)}"
    )

    # Do not print URL: it contains the API key.
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
            f"HTTP {status}"
        )

    return json.loads(payload)


def main() -> None:
    print(
        "ALPHA VANTAGE NEWS_SENTIMENT "
        "STEP 4A PROBE"
    )

    print(
        "API key configured:",
        "YES"
        if os.getenv(API_KEY_ENV)
        else "NO",
    )

    print(
        "Writes project/cache data: NO"
    )

    # One bounded historical equity request.
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "AAPL",
        "time_from": "20150220T0000",
        "time_to": "20150220T2359",
        "sort": "EARLIEST",
        "limit": "3",
    }

    try:
        payload = request_json(
            params
        )

    except Exception as exc:
        print("RESULT: ERROR")
        print(
            "ERROR:",
            f"{type(exc).__name__}: {exc}",
        )
        return

    print("RESULT: OK")

    # These provider messages are especially important:
    # premium entitlement, rate limit, invalid request, etc.
    for field in (
        "Information",
        "Note",
        "Error Message",
    ):
        if field in payload:
            print(
                f"{field.upper()}:",
                payload[field],
            )

    print(
        "TOP-LEVEL FIELDS:",
        sorted(payload.keys()),
    )

    print(
        "ITEMS:",
        payload.get("items"),
    )

    feed = payload.get("feed")

    if not isinstance(feed, list):
        feed = []

    print(
        "FEED RECORDS:",
        len(feed),
    )

    for article in feed[:3]:
        if not isinstance(
            article,
            dict,
        ):
            continue

        ticker_sentiment = article.get(
            "ticker_sentiment"
        )

        if not isinstance(
            ticker_sentiment,
            list,
        ):
            ticker_sentiment = []

        print(
            json.dumps(
                {
                    "title": article.get(
                        "title"
                    ),
                    "url_present": bool(
                        article.get("url")
                    ),
                    "time_published": (
                        article.get(
                            "time_published"
                        )
                    ),
                    "authors": article.get(
                        "authors"
                    ),
                    "summary_present": bool(
                        article.get("summary")
                    ),
                    "source": article.get(
                        "source"
                    ),
                    "category_within_source": (
                        article.get(
                            "category_within_source"
                        )
                    ),
                    "topics_present": bool(
                        article.get("topics")
                    ),
                    "ticker_sentiment_count": (
                        len(ticker_sentiment)
                    ),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()