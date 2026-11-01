from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import Any

import pytest

from market_prediction.providers.news.base import (
    NewsQuery,
)
from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
    NewsProviderUnavailableError,
    NewsQuotaExceededError,
)
from market_prediction.providers.news.marketaux import (
    MarketauxNewsProvider,
)


UTC = timezone.utc


def make_query() -> NewsQuery:
    return NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )


def start_at() -> datetime:
    return datetime(
        2025,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )


def end_at() -> datetime:
    return datetime(
        2025,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

def retrieval_time() -> datetime:
    return datetime(
        2025,
        10,
        31,
        1,
        30,
        tzinfo=UTC,
    )

def article_payload(
    *,
    uuid: str = "article-1",
    published_at: str = "2025-10-30T16:00:01Z",
) -> dict[str, object]:
    return {
        "uuid": uuid,
        "title": "Apple reports quarterly results",
        "description": (
            "Apple reported its latest quarterly "
            "financial results."
        ),
        "url": (
            f"https://example.com/{uuid}"
        ),
        "source": "Example News",
        "language": "en",
        "published_at": published_at,
    }


class FakeTransport:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: object | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = (
            payload
            if payload is not None
            else {
                "meta": {
                    "found": 1,
                    "returned": 1,
                    "limit": 50,
                    "page": 1,
                },
                "data": [
                    article_payload()
                ],
            }
        )
        self.calls: list[
            tuple[
                str,
                dict[str, str],
                float,
            ]
        ] = []

    def get_json(
        self,
        *,
        url: str,
        params: dict[str, str],
        timeout_seconds: float,
    ) -> tuple[int, object]:
        self.calls.append(
            (
                url,
                dict(params),
                timeout_seconds,
            )
        )

        return (
            self.status_code,
            self.payload,
        )

class FakeClock:
    def __init__(
        self,
        value: datetime,
    ) -> None:
        self.value = value
        self.calls = 0

    def now_utc(self) -> datetime:
        self.calls += 1
        return self.value

class SequencedFakeTransport:
    def __init__(
        self,
        responses: list[tuple[int, object]],
    ) -> None:
        self.responses = list(responses)
        self.calls: list[
            tuple[str, dict[str, str], float]
        ] = []

    def get_json(
        self,
        *,
        url: str,
        params: dict[str, str],
        timeout_seconds: float,
    ) -> tuple[int, object]:
        self.calls.append(
            (
                url,
                dict(params),
                timeout_seconds,
            )
        )

        if not self.responses:
            raise AssertionError(
                "Unexpected Marketaux request"
            )

        return self.responses.pop(0)

def marketaux_page(
    *,
    found: int,
    page: int,
    limit: int,
    items: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "meta": {
            "found": found,
            "returned": len(items),
            "limit": limit,
            "page": page,
        },
        "data": items,
    }


def make_provider(
    transport: FakeTransport,
    *,
    clock: FakeClock | None = None,
) -> MarketauxNewsProvider:
    if clock is None:
        clock = FakeClock(
            retrieval_time()
        )

    return MarketauxNewsProvider(
        base_url="https://api.marketaux.com/v1",
        api_token="secret-token",
        timeout_seconds=10.0,
        transport=transport,
        clock=clock,
    )


def test_provider_fetches_and_normalizes_articles() -> None:
    transport = FakeTransport()

    provider = make_provider(
        transport
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert len(articles) == 1

    article = articles[0]

    assert article.provider == "marketaux"
    assert article.provider_item_id == "article-1"
    assert article.matched_assets == ["AAPL"]
    assert article.query_id == "aapl_company"


def test_provider_constructs_one_http_request() -> None:
    transport = FakeTransport()

    provider = make_provider(
        transport
    )

    provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert len(transport.calls) == 1

    url, params, timeout = (
        transport.calls[0]
    )

    assert url == (
        "https://api.marketaux.com/v1/news/all"
    )
    assert params["api_token"] == "secret-token"
    assert params["search"] == (
        '"AAPL" | "Apple Inc"'
    )
    assert timeout == 10.0


def test_provider_returns_empty_list_for_empty_response() -> None:
    transport = FakeTransport(
        payload={
            "meta": {
                "found": 0,
                "returned": 0,
                "limit": 50,
                "page": 1,
            },
            "data": [],
        }
    )

    provider = make_provider(
        transport
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert articles == []


def test_provider_enforces_exact_half_open_window() -> None:
    transport = FakeTransport(
        payload={
            "meta": {
                "found": 4,
                "returned": 4,
                "limit": 50,
                "page": 1,
            },
            "data": [
                article_payload(
                    uuid="before",
                    published_at=(
                        "2025-10-29T23:59:59Z"
                    ),
                ),
                article_payload(
                    uuid="at-start",
                    published_at=(
                        "2025-10-30T00:00:00Z"
                    ),
                ),
                article_payload(
                    uuid="inside",
                    published_at=(
                        "2025-10-30T12:00:00Z"
                    ),
                ),
                article_payload(
                    uuid="at-end",
                    published_at=(
                        "2025-10-31T00:00:00Z"
                    ),
                ),
            ],
        }
    )

    provider = make_provider(
        transport
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert [
        article.provider_item_id
        for article in articles
    ] == [
        "at-start",
        "inside",
    ]


def test_provider_preserves_provider_order() -> None:
    transport = FakeTransport(
        payload={
            "meta": {
                "found": 2,
                "returned": 2,
                "limit": 50,
                "page": 1,
            },
            "data": [
                article_payload(
                    uuid="second-by-time",
                    published_at=(
                        "2025-10-30T18:00:00Z"
                    ),
                ),
                article_payload(
                    uuid="first-by-time",
                    published_at=(
                        "2025-10-30T10:00:00Z"
                    ),
                ),
            ],
        }
    )

    provider = make_provider(
        transport
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert [
        article.provider_item_id
        for article in articles
    ] == [
        "second-by-time",
        "first-by-time",
    ]


def test_provider_maps_429_to_quota_error() -> None:
    provider = make_provider(
        FakeTransport(
            status_code=429,
            payload={
                "error": {
                    "message": "Rate limit"
                }
            },
        )
    )

    with pytest.raises(
        NewsQuotaExceededError,
    ):
        provider.fetch_articles(
            query=make_query(),
            start_at_utc=start_at(),
            end_at_utc=end_at(),
        )


def test_provider_maps_server_error() -> None:
    provider = make_provider(
        FakeTransport(
            status_code=503,
            payload={
                "error": {
                    "message": "Unavailable"
                }
            },
        )
    )

    with pytest.raises(
        NewsProviderUnavailableError,
    ):
        provider.fetch_articles(
            query=make_query(),
            start_at_utc=start_at(),
            end_at_utc=end_at(),
        )


def test_provider_rejects_invalid_article_payload() -> None:
    provider = make_provider(
        FakeTransport(
            payload={
                "data": [
                    {
                        "uuid": "article-1"
                    }
                ]
            }
        )
    )

    with pytest.raises(
        InvalidNewsProviderDataError,
    ):
        provider.fetch_articles(
            query=make_query(),
            start_at_utc=start_at(),
            end_at_utc=end_at(),
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0.0,
        -1.0,
    ],
)
def test_provider_rejects_non_positive_timeout(
    timeout_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="timeout_seconds",
    ):
        MarketauxNewsProvider(
            base_url="https://api.marketaux.com/v1",
            api_token="secret-token",
            timeout_seconds=timeout_seconds,
            transport=FakeTransport(),
            clock=FakeClock(
                retrieval_time()
            ),
        )


def test_provider_rejects_missing_transport() -> None:
    with pytest.raises(
        TypeError,
        match="transport",
    ):
        MarketauxNewsProvider(
            base_url="https://api.marketaux.com/v1",
            api_token="secret-token",
            timeout_seconds=10.0,
            transport=None,  # type: ignore[arg-type]
            clock=FakeClock(
                retrieval_time()
            ),
        )

def test_provider_records_actual_retrieval_time() -> None:
    clock = FakeClock(
        retrieval_time()
    )

    provider = make_provider(
        FakeTransport(),
        clock=clock,
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert len(articles) == 1

    assert (
        articles[0].retrieved_at_utc
        == retrieval_time()
    )

    assert (
        articles[0].provider_seen_at_utc
        == retrieval_time()
    )

    assert (
        articles[0].retrieved_at_utc
        != end_at()
    )


def test_provider_reads_clock_once_per_response() -> None:
    clock = FakeClock(
        retrieval_time()
    )

    provider = make_provider(
        FakeTransport(
            payload={
                "meta": {
                    "found": 2,
                    "returned": 2,
                    "limit": 50,
                    "page": 1,
                },
                "data": [
                    article_payload(
                        uuid="article-1",
                    ),
                    article_payload(
                        uuid="article-2",
                    ),
                ],
            }
        ),
        clock=clock,
    )

    provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert clock.calls == 1


def test_provider_rejects_non_utc_clock_value() -> None:
    non_utc = timezone(
        timedelta(hours=9)
    )

    clock = FakeClock(
        datetime(
            2025,
            10,
            31,
            10,
            30,
            tzinfo=non_utc,
        )
    )

    provider = make_provider(
        FakeTransport(),
        clock=clock,
    )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="retrieved_at_utc",
    ):
        provider.fetch_articles(
            query=make_query(),
            start_at_utc=start_at(),
            end_at_utc=end_at(),
        )

def test_provider_fetches_multiple_pages_in_order() -> None:
    transport = SequencedFakeTransport(
        responses=[
            (
                200,
                marketaux_page(
                    found=3,
                    page=1,
                    limit=2,
                    items=[
                        article_payload(
                            uuid="article-1",
                        ),
                        article_payload(
                            uuid="article-2",
                        ),
                    ],
                ),
            ),
            (
                200,
                marketaux_page(
                    found=3,
                    page=2,
                    limit=2,
                    items=[
                        article_payload(
                            uuid="article-3",
                        ),
                    ],
                ),
            ),
        ]
    )

    provider = make_provider(
        transport
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert [
        article.provider_item_id
        for article in articles
    ] == [
        "article-1",
        "article-2",
        "article-3",
    ]

    assert len(transport.calls) == 2

    assert transport.calls[0][1]["page"] == "1"
    assert transport.calls[1][1]["page"] == "2"

def test_provider_stops_after_short_page() -> None:
    transport = SequencedFakeTransport(
        responses=[
            (
                200,
                marketaux_page(
                    found=10,
                    page=1,
                    limit=50,
                    items=[
                        article_payload(
                            uuid="article-1",
                        ),
                    ],
                ),
            ),
        ]
    )

    provider = make_provider(
        transport
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert [
        article.provider_item_id
        for article in articles
    ] == [
        "article-1",
    ]

    assert len(transport.calls) == 1

class SequencedFakeClock:
    def __init__(
        self,
        values: list[datetime],
    ) -> None:
        self.values = list(values)
        self.calls = 0

    def now_utc(self) -> datetime:
        self.calls += 1

        if not self.values:
            raise AssertionError(
                "Unexpected clock read"
            )

        return self.values.pop(0)

def test_provider_reads_clock_once_per_page() -> None:
    first_retrieval = retrieval_time()
    second_retrieval = (
        first_retrieval
        + timedelta(seconds=1)
    )

    clock = SequencedFakeClock(
        [
            first_retrieval,
            second_retrieval,
        ]
    )

    transport = SequencedFakeTransport(
        responses=[
            (
                200,
                marketaux_page(
                    found=3,
                    page=1,
                    limit=2,
                    items=[
                        article_payload(
                            uuid="article-1",
                        ),
                        article_payload(
                            uuid="article-2",
                        ),
                    ],
                ),
            ),
            (
                200,
                marketaux_page(
                    found=3,
                    page=2,
                    limit=2,
                    items=[
                        article_payload(
                            uuid="article-3",
                        ),
                    ],
                ),
            ),
        ]
    )

    provider = make_provider(
        transport,
        clock=clock,
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert clock.calls == 2

    assert articles[0].retrieved_at_utc == (
        first_retrieval
    )
    assert articles[1].retrieved_at_utc == (
        first_retrieval
    )
    assert articles[2].retrieved_at_utc == (
        second_retrieval
    )

def test_provider_propagates_quota_error_on_later_page() -> None:
    transport = SequencedFakeTransport(
        responses=[
            (
                200,
                marketaux_page(
                    found=3,
                    page=1,
                    limit=2,
                    items=[
                        article_payload(
                            uuid="article-1",
                        ),
                        article_payload(
                            uuid="article-2",
                        ),
                    ],
                ),
            ),
            (
                429,
                {
                    "error": {
                        "message": "Quota exceeded",
                    }
                },
            ),
        ]
    )

    provider = make_provider(
        transport
    )

    with pytest.raises(
        NewsQuotaExceededError,
    ):
        provider.fetch_articles(
            query=make_query(),
            start_at_utc=start_at(),
            end_at_utc=end_at(),
        )

    assert len(transport.calls) == 2
    assert transport.calls[0][1]["page"] == "1"
    assert transport.calls[1][1]["page"] == "2"

def test_provider_rejects_mismatched_response_page() -> None:
    transport = SequencedFakeTransport(
        responses=[
            (
                200,
                marketaux_page(
                    found=2,
                    page=2,
                    limit=2,
                    items=[
                        article_payload(
                            uuid="article-1",
                        ),
                        article_payload(
                            uuid="article-2",
                        ),
                    ],
                ),
            ),
        ]
    )

    provider = make_provider(
        transport
    )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="page",
    ):
        provider.fetch_articles(
            query=make_query(),
            start_at_utc=start_at(),
            end_at_utc=end_at(),
        )

    assert len(transport.calls) == 1
    assert transport.calls[0][1]["page"] == "1"

def test_provider_enforces_half_open_window_across_pages() -> None:
    transport = SequencedFakeTransport(
        responses=[
            (
                200,
                marketaux_page(
                    found=3,
                    page=1,
                    limit=2,
                    items=[
                        article_payload(
                            uuid="at-start",
                            published_at=(
                                "2025-10-30T00:00:00Z"
                            ),
                        ),
                        article_payload(
                            uuid="inside",
                            published_at=(
                                "2025-10-30T12:00:00Z"
                            ),
                        ),
                    ],
                ),
            ),
            (
                200,
                marketaux_page(
                    found=3,
                    page=2,
                    limit=2,
                    items=[
                        article_payload(
                            uuid="at-end",
                            published_at=(
                                "2025-10-31T00:00:00Z"
                            ),
                        ),
                    ],
                ),
            ),
        ]
    )

    provider = make_provider(
        transport
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert [
        article.provider_item_id
        for article in articles
    ] == [
        "at-start",
        "inside",
    ]

    assert len(transport.calls) == 2

def test_provider_assigns_duplicate_groups_by_url() -> None:
    transport = FakeTransport(
        payload={
            "meta": {
                "found": 2,
                "returned": 2,
                "limit": 50,
                "page": 1,
            },
            "data": [
                {
                    **article_payload(
                        uuid="article-1",
                    ),
                    "url": (
                        "https://example.com/story"
                    ),
                },
                {
                    **article_payload(
                        uuid="article-2",
                    ),
                    "url": (
                        "https://EXAMPLE.com/"
                        "story#section"
                    ),
                },
            ],
        }
    )

    provider = make_provider(
        transport
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert len(articles) == 2

    assert (
        articles[0].article_id
        != articles[1].article_id
    )

    assert (
        articles[0].duplicate_group_id
        is not None
    )

    assert (
        articles[0].duplicate_group_id
        == articles[1].duplicate_group_id
    )

def test_provider_assigns_different_groups_to_different_urls() -> None:
    transport = FakeTransport(
        payload={
            "meta": {
                "found": 2,
                "returned": 2,
                "limit": 50,
                "page": 1,
            },
            "data": [
                {
                    **article_payload(
                        uuid="article-1",
                    ),
                    "url": (
                        "https://example.com/story-1"
                    ),
                },
                {
                    **article_payload(
                        uuid="article-2",
                    ),
                    "url": (
                        "https://example.com/story-2"
                    ),
                },
            ],
        }
    )

    provider = make_provider(
        transport
    )

    articles = provider.fetch_articles(
        query=make_query(),
        start_at_utc=start_at(),
        end_at_utc=end_at(),
    )

    assert (
        articles[0].duplicate_group_id
        != articles[1].duplicate_group_id
    )