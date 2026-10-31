"""Tests for the provider-neutral market-news contract."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)
from market_prediction.providers.news.base import (
    NewsProvider,
    NewsQuery,
    validate_news_query_window,
)
from tests.contracts.news_provider_contract import (
    assert_news_provider_contract,
)
from tests.fakes.fake_news_provider import (
    FakeNewsProvider,
)


UTC = timezone.utc


def make_article() -> ArticleRecord:
    published = datetime(
        2026,
        10,
        30,
        12,
        0,
        tzinfo=UTC,
    )

    return ArticleRecord(
        article_id="a" * 64,
        matched_assets=["AAPL"],
        query_id="aapl_company",
        title="Apple reports quarterly earnings",
        snippet="Revenue increased.",
        full_text=None,
        canonical_url=(
            "https://example.com/apple-earnings"
        ),
        publisher="Example News",
        published_at_utc=published,
        published_at_precision=(
            PublishedAtPrecision.SECOND
        ),
        original_timezone="UTC",
        provider_seen_at_utc=(
            published + timedelta(minutes=1)
        ),
        retrieved_at_utc=(
            published + timedelta(minutes=2)
        ),
        provider="fake",
        provider_item_id="article-123",
        language="en",
        content_hash="b" * 64,
        duplicate_group_id=None,
        raw_payload_reference=None,
        schema_version="1",
        timestamp_source=(
            TimestampSource.PROVIDER_PUBLICATION
        ),
        text_license_class=(
            TextLicenseClass.PROVIDER_METADATA
        ),
    )


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


def test_news_query_is_immutable() -> None:
    query = make_query()

    with pytest.raises(AttributeError):
        query.asset = "MSFT"


def test_news_query_rejects_empty_asset() -> None:
    with pytest.raises(ValueError):
        NewsQuery(
            asset="",
            query_id="aapl_company",
            strict_terms=("AAPL",),
            context_terms=(),
        )


def test_news_query_rejects_empty_query_id() -> None:
    with pytest.raises(ValueError):
        NewsQuery(
            asset="AAPL",
            query_id="",
            strict_terms=("AAPL",),
            context_terms=(),
        )


def test_news_query_requires_strict_terms() -> None:
    with pytest.raises(ValueError):
        NewsQuery(
            asset="AAPL",
            query_id="aapl_company",
            strict_terms=(),
            context_terms=(),
        )


def test_news_query_rejects_empty_terms() -> None:
    with pytest.raises(ValueError):
        NewsQuery(
            asset="AAPL",
            query_id="aapl_company",
            strict_terms=(
                "AAPL",
                "",
            ),
            context_terms=(),
        )


def test_fake_provider_satisfies_contract() -> None:
    article = make_article()

    provider = FakeNewsProvider(
        [article]
    )

    start = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    articles = assert_news_provider_contract(
        provider,
        query=make_query(),
        start_at_utc=start,
        end_at_utc=end,
    )

    assert articles == [article]

    assert provider.calls == [
        (
            make_query(),
            start,
            end,
        )
    ]


def test_news_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        NewsProvider()


def test_provider_contract_allows_empty_result() -> None:
    provider = FakeNewsProvider([])

    articles = assert_news_provider_contract(
        provider,
        query=make_query(),
        start_at_utc=datetime(
            2026,
            10,
            30,
            tzinfo=UTC,
        ),
        end_at_utc=datetime(
            2026,
            10,
            31,
            tzinfo=UTC,
        ),
    )

    assert articles == []

from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
    NewsProviderError,
    NewsProviderUnavailableError,
    NewsQuotaExceededError,
    UnsupportedNewsQueryError,
)


def test_news_provider_exceptions_share_base_class() -> None:
    assert issubclass(
        NewsProviderUnavailableError,
        NewsProviderError,
    )

    assert issubclass(
        NewsQuotaExceededError,
        NewsProviderError,
    )

    assert issubclass(
        UnsupportedNewsQueryError,
        NewsProviderError,
    )

    assert issubclass(
        InvalidNewsProviderDataError,
        NewsProviderError,
    )

def test_query_window_accepts_utc_range() -> None:
    start = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    validate_news_query_window(
        start_at_utc=start,
        end_at_utc=end,
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "start_at_utc",
        "end_at_utc",
    ],
)
def test_query_window_rejects_naive_timestamp(
    field_name: str,
) -> None:
    start = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    values = {
        "start_at_utc": start,
        "end_at_utc": end,
    }

    values[field_name] = values[
        field_name
    ].replace(tzinfo=None)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        validate_news_query_window(**values)


@pytest.mark.parametrize(
    "field_name",
    [
        "start_at_utc",
        "end_at_utc",
    ],
)
def test_query_window_rejects_non_utc_timestamp(
    field_name: str,
) -> None:
    start = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    non_utc = timezone(
        timedelta(hours=9)
    )

    values = {
        "start_at_utc": start,
        "end_at_utc": end,
    }

    values[field_name] = values[
        field_name
    ].astimezone(non_utc)

    with pytest.raises(
        ValueError,
        match="must be UTC",
    ):
        validate_news_query_window(**values)


def test_query_window_rejects_equal_boundaries() -> None:
    timestamp = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    with pytest.raises(
        ValueError,
        match="must be earlier",
    ):
        validate_news_query_window(
            start_at_utc=timestamp,
            end_at_utc=timestamp,
        )


def test_query_window_rejects_reversed_range() -> None:
    start = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    end = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    with pytest.raises(
        ValueError,
        match="must be earlier",
    ):
        validate_news_query_window(
            start_at_utc=start,
            end_at_utc=end,
        )