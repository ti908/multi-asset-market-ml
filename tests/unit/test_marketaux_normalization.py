from __future__ import annotations

from datetime import datetime, timezone

import pytest

from market_prediction.news.fingerprint import (
    article_identity,
    content_fingerprint,
)
from market_prediction.news.models import (
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)
from market_prediction.providers.news.base import (
    NewsQuery,
)
from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
)
from market_prediction.providers.news.marketaux import (
    normalize_marketaux_article,
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


def make_payload(
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "uuid": "marketaux-article-123",
        "title": "Apple reports quarterly results",
        "description": (
            "Apple reported its latest quarterly "
            "financial results."
        ),
        "url": (
            "https://example.com/"
            "apple-quarterly-results"
        ),
        "source": "Example News",
        "language": "en",
        "published_at": (
            "2025-10-30T16:00:01.000000Z"
        ),
    }

    payload.update(overrides)

    return payload


def retrieved_at() -> datetime:
    return datetime(
        2025,
        10,
        30,
        17,
        0,
        tzinfo=UTC,
    )


def test_normalize_marketaux_article_maps_identity() -> None:
    article = normalize_marketaux_article(
        payload=make_payload(),
        query=make_query(),
        retrieved_at_utc=retrieved_at(),
    )

    assert article.article_id == article_identity(
        provider="marketaux",
        provider_item_id="marketaux-article-123",
    )
    assert article.provider == "marketaux"
    assert (
        article.provider_item_id
        == "marketaux-article-123"
    )


def test_normalize_marketaux_article_maps_query() -> None:
    article = normalize_marketaux_article(
        payload=make_payload(),
        query=make_query(),
        retrieved_at_utc=retrieved_at(),
    )

    assert article.matched_assets == ["AAPL"]
    assert article.query_id == "aapl_company"


def test_normalize_marketaux_article_maps_text() -> None:
    article = normalize_marketaux_article(
        payload=make_payload(),
        query=make_query(),
        retrieved_at_utc=retrieved_at(),
    )

    assert article.title == (
        "Apple reports quarterly results"
    )
    assert article.snippet == (
        "Apple reported its latest quarterly "
        "financial results."
    )
    assert article.full_text is None


def test_normalize_marketaux_article_maps_source_metadata() -> None:
    article = normalize_marketaux_article(
        payload=make_payload(),
        query=make_query(),
        retrieved_at_utc=retrieved_at(),
    )

    assert article.canonical_url == (
        "https://example.com/"
        "apple-quarterly-results"
    )
    assert article.publisher == "Example News"
    assert article.language == "en"


def test_normalize_marketaux_article_parses_utc_timestamp() -> None:
    article = normalize_marketaux_article(
        payload=make_payload(),
        query=make_query(),
        retrieved_at_utc=retrieved_at(),
    )

    assert article.published_at_utc == datetime(
        2025,
        10,
        30,
        16,
        0,
        1,
        tzinfo=UTC,
    )
    assert (
        article.published_at_precision
        is PublishedAtPrecision.SECOND
    )
    assert (
        article.timestamp_source
        is TimestampSource.PROVIDER_PUBLICATION
    )


def test_normalize_marketaux_article_uses_retrieval_as_provider_seen() -> None:
    retrieved = retrieved_at()

    article = normalize_marketaux_article(
        payload=make_payload(),
        query=make_query(),
        retrieved_at_utc=retrieved,
    )

    assert article.provider_seen_at_utc == retrieved
    assert article.retrieved_at_utc == retrieved


def test_normalize_marketaux_article_sets_content_fingerprint() -> None:
    payload = make_payload()

    article = normalize_marketaux_article(
        payload=payload,
        query=make_query(),
        retrieved_at_utc=retrieved_at(),
    )

    assert article.content_hash == content_fingerprint(
        title=payload["title"],
        snippet=payload["description"],
        full_text=None,
    )


def test_normalize_marketaux_article_sets_license_class() -> None:
    article = normalize_marketaux_article(
        payload=make_payload(),
        query=make_query(),
        retrieved_at_utc=retrieved_at(),
    )

    assert (
        article.text_license_class
        is TextLicenseClass.PROVIDER_METADATA
    )


def test_normalize_marketaux_article_does_not_claim_duplicate_group() -> None:
    article = normalize_marketaux_article(
        payload=make_payload(),
        query=make_query(),
        retrieved_at_utc=retrieved_at(),
    )

    assert article.duplicate_group_id is None


@pytest.mark.parametrize(
    "field",
    [
        "uuid",
        "title",
        "url",
        "source",
        "language",
        "published_at",
    ],
)
def test_normalize_marketaux_article_rejects_missing_required_field(
    field: str,
) -> None:
    payload = make_payload()
    del payload[field]

    with pytest.raises(
        InvalidNewsProviderDataError,
        match=field,
    ):
        normalize_marketaux_article(
            payload=payload,
            query=make_query(),
            retrieved_at_utc=retrieved_at(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "uuid",
        "title",
        "url",
        "source",
        "language",
        "published_at",
    ],
)
def test_normalize_marketaux_article_rejects_blank_required_field(
    field: str,
) -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match=field,
    ):
        normalize_marketaux_article(
            payload=make_payload(
                **{field: "   "}
            ),
            query=make_query(),
            retrieved_at_utc=retrieved_at(),
        )


def test_normalize_marketaux_article_accepts_null_description() -> None:
    article = normalize_marketaux_article(
        payload=make_payload(
            description=None
        ),
        query=make_query(),
        retrieved_at_utc=retrieved_at(),
    )

    assert article.snippet is None


def test_normalize_marketaux_article_rejects_non_string_description() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="description",
    ):
        normalize_marketaux_article(
            payload=make_payload(
                description=123
            ),
            query=make_query(),
            retrieved_at_utc=retrieved_at(),
        )


def test_normalize_marketaux_article_rejects_naive_published_at() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="published_at",
    ):
        normalize_marketaux_article(
            payload=make_payload(
                published_at="2025-10-30T16:00:01"
            ),
            query=make_query(),
            retrieved_at_utc=retrieved_at(),
        )


def test_normalize_marketaux_article_rejects_non_utc_published_at() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="published_at",
    ):
        normalize_marketaux_article(
            payload=make_payload(
                published_at=(
                    "2025-10-31T01:00:01+09:00"
                )
            ),
            query=make_query(),
            retrieved_at_utc=retrieved_at(),
        )


def test_normalize_marketaux_article_rejects_future_publication() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="after",
    ):
        normalize_marketaux_article(
            payload=make_payload(
                published_at=(
                    "2025-10-30T18:00:00Z"
                )
            ),
            query=make_query(),
            retrieved_at_utc=retrieved_at(),
        )


def test_normalize_marketaux_article_rejects_non_mapping_payload() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="mapping",
    ):
        normalize_marketaux_article(
            payload=None,  # type: ignore[arg-type]
            query=make_query(),
            retrieved_at_utc=retrieved_at(),
        )