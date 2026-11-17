"""Tests for canonical article-level sentiment contracts."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from market_prediction.sentiment.models import (
    SentimentLabel,
    SentimentRecord,
)


UTC = timezone.utc


def make_record(
    **overrides: object,
) -> SentimentRecord:
    """Build a valid sentiment record for focused tests."""

    values = {
        "sentiment_id": (
            "sentiment-marketaux-article-1-AAPL"
        ),
        "article_id": "marketaux-article-1",
        "asset": "AAPL",
        "query_id": "aapl-primary",
        "published_at_utc": datetime(
            2026,
            10,
            30,
            12,
            0,
            tzinfo=UTC,
        ),
        "provider_seen_at_utc": datetime(
            2026,
            10,
            30,
            12,
            1,
            tzinfo=UTC,
        ),
        "model_name": "ProsusAI/finbert",
        "model_revision": (
            "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
        ),
        "scoring_method": "finbert",
        "positive_probability": 0.80,
        "negative_probability": 0.10,
        "neutral_probability": 0.10,
        "predicted_label": SentimentLabel.POSITIVE,
        "confidence": 0.80,
        "signed_score": 0.70,
        "scored_at_utc": datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        "input_text_hash": "a" * 64,
        "schema_version": "1",
    }

    values.update(overrides)

    return SentimentRecord(**values)


def test_valid_sentiment_record_is_accepted() -> None:
    record = make_record()

    assert record.article_id == "marketaux-article-1"
    assert record.asset == "AAPL"
    assert record.predicted_label is SentimentLabel.POSITIVE
    assert record.signed_score == pytest.approx(0.70)


@pytest.mark.parametrize(
    "field",
    [
        "sentiment_id",
        "article_id",
        "asset",
        "query_id",
        "model_name",
        "model_revision",
        "scoring_method",
        "schema_version",
    ],
)
def test_required_strings_cannot_be_empty(
    field: str,
) -> None:
    with pytest.raises(ValidationError):
        make_record(**{field: ""})


@pytest.mark.parametrize(
    "field",
    [
        "positive_probability",
        "negative_probability",
        "neutral_probability",
        "confidence",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01],
)
def test_probability_like_fields_must_be_unit_interval(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        make_record(**{field: value})


@pytest.mark.parametrize(
    "value",
    [-1.01, 1.01],
)
def test_signed_score_must_be_between_minus_one_and_one(
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        make_record(signed_score=value)


def test_probabilities_must_sum_to_one() -> None:
    with pytest.raises(
        ValidationError,
        match="probabilities must sum to 1",
    ):
        make_record(
            positive_probability=0.60,
            negative_probability=0.20,
            neutral_probability=0.10,
        )


def test_signed_score_must_equal_positive_minus_negative() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "signed_score must equal "
            "positive_probability - negative_probability"
        ),
    ):
        make_record(signed_score=0.25)


def test_predicted_label_must_match_highest_probability() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "predicted_label must match "
            "highest probability"
        ),
    ):
        make_record(
            predicted_label=SentimentLabel.NEGATIVE,
        )


def test_confidence_must_equal_highest_probability() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "confidence must equal "
            "highest probability"
        ),
    ):
        make_record(confidence=0.75)


@pytest.mark.parametrize(
    "field",
    [
        "published_at_utc",
        "provider_seen_at_utc",
        "scored_at_utc",
    ],
)
def test_normalized_timestamps_must_be_timezone_aware(
    field: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match="timezone-aware",
    ):
        make_record(
            **{
                field: datetime(
                    2026,
                    10,
                    30,
                    12,
                    0,
                )
            }
        )


@pytest.mark.parametrize(
    "field",
    [
        "published_at_utc",
        "provider_seen_at_utc",
        "scored_at_utc",
    ],
)
def test_normalized_timestamps_must_be_utc(
    field: str,
) -> None:
    non_utc = timezone.utc

    if field == "published_at_utc":
        non_utc = timezone(
            __import__("datetime").timedelta(hours=1)
        )
    else:
        non_utc = timezone(
            __import__("datetime").timedelta(hours=-5)
        )

    with pytest.raises(
        ValidationError,
        match="must be UTC",
    ):
        make_record(
            **{
                field: datetime(
                    2026,
                    10,
                    30,
                    12,
                    0,
                    tzinfo=non_utc,
                )
            }
        )


def test_provider_seen_cannot_precede_publication() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "provider_seen_at_utc cannot precede "
            "published_at_utc"
        ),
    ):
        make_record(
            provider_seen_at_utc=datetime(
                2026,
                10,
                30,
                11,
                59,
                tzinfo=UTC,
            )
        )


def test_scoring_cannot_precede_provider_seen_time() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "scored_at_utc cannot precede "
            "provider_seen_at_utc"
        ),
    ):
        make_record(
            scored_at_utc=datetime(
                2026,
                10,
                30,
                12,
                0,
                tzinfo=UTC,
            ),
            provider_seen_at_utc=datetime(
                2026,
                10,
                30,
                12,
                1,
                tzinfo=UTC,
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        "abc",
        "A" * 64,
        "g" * 64,
        "a" * 63,
        "a" * 65,
    ],
)
def test_input_text_hash_must_be_lowercase_sha256(
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        make_record(input_text_hash=value)


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        make_record(unexpected_field="not allowed")
