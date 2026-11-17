"""Tests for strict sentiment dataset validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_prediction.sentiment.models import (
    SentimentLabel,
    SentimentRecord,
)
from market_prediction.sentiment.validation import (
    SentimentStatus,
    validate_sentiments,
)


UTC = timezone.utc

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_REVISION = (
    "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
)


def make_sentiment(
    **overrides,
) -> SentimentRecord:
    """Return one valid sentiment observation."""

    published = datetime(
        2026,
        10,
        30,
        12,
        0,
        tzinfo=UTC,
    )

    values = {
        "sentiment_id": "a" * 64,
        "article_id": "b" * 64,
        "asset": "AAPL",
        "query_id": "aapl_company",
        "published_at_utc": published,
        "provider_seen_at_utc": (
            published + timedelta(minutes=1)
        ),
        "model_name": FINBERT_MODEL,
        "model_revision": FINBERT_REVISION,
        "scoring_method": "finbert",
        "positive_probability": 0.80,
        "neutral_probability": 0.15,
        "negative_probability": 0.05,
        "predicted_label": SentimentLabel.POSITIVE,
        "confidence": 0.80,
        "signed_score": 0.75,
        "scored_at_utc": (
            published + timedelta(minutes=2)
        ),
        "input_text_hash": "c" * 64,
        "schema_version": "1",
    }

    values.update(overrides)

    return SentimentRecord(**values)


def validate(
    sentiments: list[SentimentRecord],
    *,
    skipped_checks: list[str] | None = None,
):
    """Validate using the frozen Step 5 configuration."""

    return validate_sentiments(
        sentiments,
        allowed_assets={"AAPL", "MSFT"},
        allowed_query_ids={
            "aapl_company",
            "msft_company",
        },
        expected_model_name=FINBERT_MODEL,
        expected_model_revision=FINBERT_REVISION,
        expected_scoring_method="finbert",
        expected_schema_version="1",
        skipped_checks=skipped_checks,
    )


def test_valid_sentiments_pass() -> None:
    report = validate([make_sentiment()])

    assert report.status is SentimentStatus.PASSED
    assert report.sentiment_count == 1
    assert report.hard_failures == []
    assert report.soft_anomalies == []
    assert report.skipped_checks == []


def test_empty_dataset_passes() -> None:
    report = validate([])

    assert report.status is SentimentStatus.PASSED
    assert report.sentiment_count == 0
    assert report.hard_failures == []
    assert report.soft_anomalies == []


def test_skipped_check_blocks_dataset() -> None:
    report = validate(
        [make_sentiment()],
        skipped_checks=["model_revision"],
    )

    assert report.status is SentimentStatus.BLOCKED
    assert report.skipped_checks == [
        "model_revision"
    ]


def test_unknown_asset_is_hard_failure() -> None:
    record = make_sentiment(asset="UNKNOWN")

    report = validate([record])

    assert report.status is SentimentStatus.BLOCKED

    assert any(
        issue.rule == "unknown_asset"
        for issue in report.hard_failures
    )


def test_unknown_query_id_is_hard_failure() -> None:
    record = make_sentiment(
        query_id="unknown_query"
    )

    report = validate([record])

    assert report.status is SentimentStatus.BLOCKED

    assert any(
        issue.rule == "unknown_query_id"
        for issue in report.hard_failures
    )


def test_duplicate_sentiment_id_is_hard_failure() -> None:
    first = make_sentiment()

    second = make_sentiment(
        article_id="d" * 64,
        input_text_hash="e" * 64,
    )

    report = validate([first, second])

    assert report.status is SentimentStatus.BLOCKED

    assert any(
        issue.rule == "duplicate_sentiment_id"
        for issue in report.hard_failures
    )


def test_duplicate_logical_observation_is_hard_failure() -> None:
    first = make_sentiment(
        sentiment_id="a" * 64,
    )

    second = make_sentiment(
        sentiment_id="d" * 64,
    )

    report = validate([first, second])

    assert report.status is SentimentStatus.BLOCKED

    assert any(
        issue.rule == "duplicate_logical_observation"
        for issue in report.hard_failures
    )


def test_unexpected_model_name_is_hard_failure() -> None:
    record = make_sentiment(
        model_name="other/model"
    )

    report = validate([record])

    assert report.status is SentimentStatus.BLOCKED

    assert any(
        issue.rule == "unexpected_model_name"
        for issue in report.hard_failures
    )


def test_unexpected_model_revision_is_hard_failure() -> None:
    record = make_sentiment(
        model_revision="d" * 40
    )

    report = validate([record])

    assert report.status is SentimentStatus.BLOCKED

    assert any(
        issue.rule == "unexpected_model_revision"
        for issue in report.hard_failures
    )


def test_unexpected_scoring_method_is_hard_failure() -> None:
    record = make_sentiment(
        scoring_method="vader"
    )

    report = validate([record])

    assert report.status is SentimentStatus.BLOCKED

    assert any(
        issue.rule == "unexpected_scoring_method"
        for issue in report.hard_failures
    )


def test_unexpected_schema_version_is_hard_failure() -> None:
    record = make_sentiment(
        schema_version="2"
    )

    report = validate([record])

    assert report.status is SentimentStatus.BLOCKED

    assert any(
        issue.rule == "unexpected_schema_version"
        for issue in report.hard_failures
    )

def test_validation_report_binds_dataset_fingerprint() -> None:
    sentiments = [make_sentiment()]

    report = validate(sentiments)

    from market_prediction.sentiment.fingerprint import (
        sentiment_dataset_fingerprint,
    )

    assert report.dataset_fingerprint == (
        sentiment_dataset_fingerprint(
            sentiments
        )
    )


def test_empty_dataset_has_validation_fingerprint() -> None:
    sentiments: list[SentimentRecord] = []

    report = validate(sentiments)

    from market_prediction.sentiment.fingerprint import (
        sentiment_dataset_fingerprint,
    )

    assert report.status is SentimentStatus.PASSED
    assert report.dataset_fingerprint == (
        sentiment_dataset_fingerprint([])
    )