"""Tests for the canonical Step 4 news article contract."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TimestampSource,
)


def valid_article_payload() -> dict:
    """Return one valid canonical news article payload."""

    return {
        "article_id": "article-001",
        "matched_assets": ["AAPL"],
        "query_id": "aapl_company",
        "title": "Apple reports quarterly earnings",
        "snippet": "Apple reported quarterly financial results.",
        "full_text": None,
        "canonical_url": "https://example.com/apple-earnings",
        "publisher": "Example News",
        "published_at_utc": datetime(
            2026,
            10,
            30,
            16,
            0,
            tzinfo=timezone.utc,
        ),
        "published_at_precision": "second",
        "original_timezone": "UTC",
        "provider_seen_at_utc": datetime(
            2026,
            10,
            30,
            16,
            1,
            tzinfo=timezone.utc,
        ),
        "retrieved_at_utc": datetime(
            2026,
            10,
            30,
            16,
            5,
            tzinfo=timezone.utc,
        ),
        "provider": "marketaux",
        "provider_item_id": "provider-item-001",
        "language": "en",
        "content_hash": (
            "a" * 64
        ),
        "duplicate_group_id": None,
        "raw_payload_reference": None,
        "schema_version": "1",
        "timestamp_source": "provider_publication",
        "text_license_class": "provider_metadata",
    }


def test_valid_article_record_loads() -> None:
    article = ArticleRecord.model_validate(
        valid_article_payload()
    )

    assert article.article_id == "article-001"
    assert article.matched_assets == ["AAPL"]
    assert article.provider == "marketaux"

    assert (
        article.published_at_precision
        is PublishedAtPrecision.SECOND
    )

    assert (
        article.timestamp_source
        is TimestampSource.PROVIDER_PUBLICATION
    )


def test_article_record_rejects_unknown_fields() -> None:
    payload = valid_article_payload()
    payload["unexpected"] = "not allowed"

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    [
        "article_id",
        "query_id",
        "title",
        "canonical_url",
        "publisher",
        "provider",
        "provider_item_id",
        "language",
        "content_hash",
        "schema_version",
    ],
)
def test_required_strings_reject_empty_values(
    field: str,
) -> None:
    payload = valid_article_payload()
    payload[field] = ""

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


def test_article_requires_at_least_one_matched_asset() -> None:
    payload = valid_article_payload()
    payload["matched_assets"] = []

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


def test_matched_assets_must_be_unique() -> None:
    payload = valid_article_payload()
    payload["matched_assets"] = [
        "AAPL",
        "AAPL",
    ]

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


def test_published_timestamp_must_be_timezone_aware() -> None:
    payload = valid_article_payload()
    payload["published_at_utc"] = datetime(
        2026,
        10,
        30,
        16,
        0,
    )

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


def test_provider_seen_timestamp_must_be_timezone_aware() -> None:
    payload = valid_article_payload()
    payload["provider_seen_at_utc"] = datetime(
        2026,
        10,
        30,
        16,
        1,
    )

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


def test_retrieved_timestamp_must_be_timezone_aware() -> None:
    payload = valid_article_payload()
    payload["retrieved_at_utc"] = datetime(
        2026,
        10,
        30,
        16,
        5,
    )

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    [
        "published_at_utc",
        "provider_seen_at_utc",
        "retrieved_at_utc",
    ],
)
def test_normalized_timestamps_accept_utc(
    field: str,
) -> None:
    payload = valid_article_payload()

    utc_values = {
        "published_at_utc": datetime(
            2026,
            10,
            30,
            16,
            0,
            tzinfo=timezone.utc,
        ),
        "provider_seen_at_utc": datetime(
            2026,
            10,
            30,
            16,
            1,
            tzinfo=timezone.utc,
        ),
        "retrieved_at_utc": datetime(
            2026,
            10,
            30,
            16,
            5,
            tzinfo=timezone.utc,
        ),
    }

    payload[field] = utc_values[field]

    article = ArticleRecord.model_validate(
        payload
    )

    timestamp = getattr(
        article,
        field,
    )

    assert timestamp.utcoffset() is not None
    assert timestamp.utcoffset().total_seconds() == 0

@pytest.mark.parametrize(
    "field",
    [
        "published_at_utc",
        "provider_seen_at_utc",
        "retrieved_at_utc",
    ],
)
def test_normalized_timestamp_rejects_non_utc_offset(
    field: str,
) -> None:
    from datetime import timedelta

    payload = valid_article_payload()

    payload[field] = datetime(
        2026,
        10,
        30,
        16,
        0,
        tzinfo=timezone(timedelta(hours=9)),
    )

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)

def test_retrieved_timestamp_cannot_precede_publication() -> None:
    payload = valid_article_payload()

    payload["retrieved_at_utc"] = datetime(
        2026,
        10,
        30,
        15,
        59,
        tzinfo=timezone.utc,
    )

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


def test_provider_seen_timestamp_cannot_precede_publication() -> None:
    payload = valid_article_payload()

    payload["provider_seen_at_utc"] = datetime(
        2026,
        10,
        30,
        15,
        59,
        tzinfo=timezone.utc,
    )

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


def test_retrieved_timestamp_cannot_precede_provider_seen() -> None:
    payload = valid_article_payload()

    payload["provider_seen_at_utc"] = datetime(
        2026,
        10,
        30,
        16,
        10,
        tzinfo=timezone.utc,
    )

    payload["retrieved_at_utc"] = datetime(
        2026,
        10,
        30,
        16,
        5,
        tzinfo=timezone.utc,
    )

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


def test_content_hash_must_be_sha256_hex() -> None:
    payload = valid_article_payload()
    payload["content_hash"] = "not-a-sha256"

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)


def test_optional_text_fields_may_be_absent() -> None:
    payload = valid_article_payload()

    payload["snippet"] = None
    payload["full_text"] = None
    payload["duplicate_group_id"] = None
    payload["raw_payload_reference"] = None

    article = ArticleRecord.model_validate(payload)

    assert article.snippet is None
    assert article.full_text is None


def test_title_is_required_even_without_article_body() -> None:
    payload = valid_article_payload()
    payload["title"] = ""

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)