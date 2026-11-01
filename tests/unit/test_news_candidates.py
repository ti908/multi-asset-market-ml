from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest
from pydantic import ValidationError

from market_prediction.news.candidates import NewsCandidate
from market_prediction.news.models import (
    PublishedAtPrecision,
    TimestampSource,
)


UTC = timezone.utc


def make_candidate(
    **overrides: object,
) -> NewsCandidate:
    values: dict[str, object] = {
        "candidate_id": (
            "gdelt:20150219000000-0"
        ),
        "asset": "AAPL",
        "query_id": "aapl_company",
        "canonical_url": (
            "https://example.com/apple-results"
        ),
        "publisher": "example.com",
        "published_at_utc": datetime(
            2015,
            2,
            19,
            0,
            0,
            tzinfo=UTC,
        ),
        "published_at_precision": (
            PublishedAtPrecision.SECOND
        ),
        "provider_seen_at_utc": datetime(
            2015,
            2,
            19,
            0,
            0,
            tzinfo=UTC,
        ),
        "retrieved_at_utc": datetime(
            2026,
            10,
            31,
            12,
            0,
            tzinfo=UTC,
        ),
        "provider": "gdelt",
        "provider_item_id": (
            "20150219000000-0"
        ),
        "timestamp_source": (
            TimestampSource.PROVIDER_PUBLICATION
        ),
    }

    values.update(overrides)

    return NewsCandidate.model_validate(
        values
    )


def test_candidate_accepts_valid_discovery_metadata() -> None:
    candidate = make_candidate()

    assert candidate.asset == "AAPL"
    assert candidate.query_id == "aapl_company"
    assert candidate.provider == "gdelt"
    assert (
        candidate.provider_item_id
        == "20150219000000-0"
    )


def test_candidate_does_not_require_article_title() -> None:
    candidate = make_candidate()

    assert not hasattr(
        candidate,
        "title",
    )


def test_candidate_does_not_require_article_text() -> None:
    candidate = make_candidate()

    assert not hasattr(
        candidate,
        "snippet",
    )
    assert not hasattr(
        candidate,
        "full_text",
    )


@pytest.mark.parametrize(
    "field",
    [
        "candidate_id",
        "asset",
        "query_id",
        "canonical_url",
        "publisher",
        "provider",
        "provider_item_id",
    ],
)
def test_candidate_rejects_blank_required_text(
    field: str,
) -> None:
    with pytest.raises(ValidationError):
        make_candidate(
            **{field: "   "}
        )


def test_candidate_rejects_naive_publication_timestamp() -> None:
    with pytest.raises(ValidationError):
        make_candidate(
            published_at_utc=datetime(
                2015,
                2,
                19,
                0,
                0,
            )
        )


def test_candidate_rejects_non_utc_publication_timestamp() -> None:
    non_utc = timezone(
        timedelta(hours=9)
    )

    with pytest.raises(ValidationError):
        make_candidate(
            published_at_utc=datetime(
                2015,
                2,
                19,
                9,
                0,
                tzinfo=non_utc,
            )
        )


def test_candidate_rejects_publication_after_provider_seen() -> None:
    with pytest.raises(ValidationError):
        make_candidate(
            published_at_utc=datetime(
                2015,
                2,
                20,
                tzinfo=UTC,
            ),
            provider_seen_at_utc=datetime(
                2015,
                2,
                19,
                tzinfo=UTC,
            ),
        )


def test_candidate_rejects_provider_seen_after_retrieval() -> None:
    with pytest.raises(ValidationError):
        make_candidate(
            provider_seen_at_utc=datetime(
                2026,
                11,
                1,
                tzinfo=UTC,
            ),
            retrieved_at_utc=datetime(
                2026,
                10,
                31,
                tzinfo=UTC,
            ),
        )


def test_candidate_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        make_candidate(
            unexpected_field="not-allowed"
        )