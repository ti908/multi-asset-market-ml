"""Tests for strict sentiment aggregation dataset validation."""

from __future__ import annotations

from datetime import datetime, timezone

from market_prediction.sentiment.aggregation import (
    SentimentAggregationRecord,
)
from market_prediction.sentiment.aggregation_validation import (
    SentimentAggregationStatus,
    validate_sentiment_aggregations,
)


UTC = timezone.utc


def make_aggregation(
    **overrides,
) -> SentimentAggregationRecord:
    """Return one valid aggregation record."""

    values = {
        "asset": "AAPL",
        "cutoff_at_utc": datetime(
            2026,
            11,
            17,
            12,
            0,
            tzinfo=UTC,
        ),
        "window_hours": 6,
        "article_count": 2,
        "signed_score_mean": 0.20,
        "signed_score_std": 0.30,
        "signed_score_min": -0.10,
        "signed_score_max": 0.50,
        "positive_probability_mean": 0.50,
        "neutral_probability_mean": 0.20,
        "negative_probability_mean": 0.30,
        "positive_label_share": 0.50,
        "neutral_label_share": 0.00,
        "negative_label_share": 0.50,
    }

    values.update(overrides)

    return SentimentAggregationRecord(**values)


def validate(
    records: list[SentimentAggregationRecord],
    *,
    skipped_checks: list[str] | None = None,
):
    """Validate using the frozen Step 5d configuration."""

    return validate_sentiment_aggregations(
        records,
        allowed_assets={"AAPL", "MSFT"},
        expected_windows={6, 24, 72},
        skipped_checks=skipped_checks,
    )


def test_valid_aggregation_dataset_passes() -> None:
    records = [
        make_aggregation(window_hours=6),
        make_aggregation(window_hours=24),
        make_aggregation(window_hours=72),
    ]

    report = validate(records)

    assert (
        report.status
        is SentimentAggregationStatus.PASSED
    )
    assert report.aggregation_count == 3
    assert report.hard_failures == []
    assert report.soft_anomalies == []
    assert report.skipped_checks == []


def test_empty_dataset_passes() -> None:
    report = validate([])

    assert (
        report.status
        is SentimentAggregationStatus.PASSED
    )
    assert report.aggregation_count == 0
    assert report.hard_failures == []
    assert report.soft_anomalies == []


def test_skipped_check_blocks_dataset() -> None:
    report = validate(
        [make_aggregation()],
        skipped_checks=["window_coverage"],
    )

    assert (
        report.status
        is SentimentAggregationStatus.BLOCKED
    )
    assert report.skipped_checks == [
        "window_coverage"
    ]


def test_unknown_asset_is_hard_failure() -> None:
    record = make_aggregation(asset="UNKNOWN")

    report = validate([record])

    assert (
        report.status
        is SentimentAggregationStatus.BLOCKED
    )

    assert any(
        issue.rule == "unknown_asset"
        for issue in report.hard_failures
    )


def test_unexpected_window_is_hard_failure() -> None:
    record = make_aggregation(window_hours=12)

    report = validate([record])

    assert (
        report.status
        is SentimentAggregationStatus.BLOCKED
    )

    assert any(
        issue.rule == "unexpected_window"
        for issue in report.hard_failures
    )


def test_duplicate_logical_aggregation_is_hard_failure() -> None:
    first = make_aggregation()
    second = make_aggregation()

    report = validate([first, second])

    assert (
        report.status
        is SentimentAggregationStatus.BLOCKED
    )

    assert any(
        issue.rule == "duplicate_logical_aggregation"
        for issue in report.hard_failures
    )


def test_same_asset_different_windows_are_not_duplicates() -> None:
    records = [
        make_aggregation(window_hours=6),
        make_aggregation(window_hours=24),
        make_aggregation(window_hours=72),
    ]

    report = validate(records)

    assert not any(
        issue.rule == "duplicate_logical_aggregation"
        for issue in report.hard_failures
    )


def test_same_asset_window_different_cutoffs_are_not_duplicates() -> None:
    first = make_aggregation()

    second = make_aggregation(
        cutoff_at_utc=datetime(
            2026,
            11,
            17,
            13,
            0,
            tzinfo=UTC,
        )
    )

    report = validate([first, second])

    assert not any(
        issue.rule == "duplicate_logical_aggregation"
        for issue in report.hard_failures
    )


def test_complete_cutoff_requires_all_expected_windows() -> None:
    records = [
        make_aggregation(window_hours=6),
        make_aggregation(window_hours=24),
    ]

    report = validate(records)

    assert (
        report.status
        is SentimentAggregationStatus.BLOCKED
    )

    assert any(
        issue.rule == "missing_expected_window"
        for issue in report.hard_failures
    )


def test_complete_cutoff_with_all_windows_passes() -> None:
    records = [
        make_aggregation(window_hours=6),
        make_aggregation(window_hours=24),
        make_aggregation(window_hours=72),
    ]

    report = validate(records)

    assert (
        report.status
        is SentimentAggregationStatus.PASSED
    )


def test_window_coverage_is_checked_per_asset() -> None:
    records = [
        make_aggregation(
            asset="AAPL",
            window_hours=6,
        ),
        make_aggregation(
            asset="AAPL",
            window_hours=24,
        ),
        make_aggregation(
            asset="AAPL",
            window_hours=72,
        ),
        make_aggregation(
            asset="MSFT",
            window_hours=6,
        ),
    ]

    report = validate(records)

    assert (
        report.status
        is SentimentAggregationStatus.BLOCKED
    )

    missing = [
        issue
        for issue in report.hard_failures
        if issue.rule == "missing_expected_window"
    ]

    assert len(missing) == 2


def test_window_coverage_is_checked_per_cutoff() -> None:
    first_cutoff = datetime(
        2026,
        11,
        17,
        12,
        tzinfo=UTC,
    )
    second_cutoff = datetime(
        2026,
        11,
        17,
        13,
        tzinfo=UTC,
    )

    records = [
        make_aggregation(
            cutoff_at_utc=first_cutoff,
            window_hours=6,
        ),
        make_aggregation(
            cutoff_at_utc=first_cutoff,
            window_hours=24,
        ),
        make_aggregation(
            cutoff_at_utc=first_cutoff,
            window_hours=72,
        ),
        make_aggregation(
            cutoff_at_utc=second_cutoff,
            window_hours=6,
        ),
    ]

    report = validate(records)

    missing = [
        issue
        for issue in report.hard_failures
        if issue.rule == "missing_expected_window"
    ]

    assert len(missing) == 2


def test_zero_article_windows_are_valid_records() -> None:
    empty_values = {
        "article_count": 0,
        "signed_score_mean": None,
        "signed_score_std": None,
        "signed_score_min": None,
        "signed_score_max": None,
        "positive_probability_mean": None,
        "neutral_probability_mean": None,
        "negative_probability_mean": None,
        "positive_label_share": None,
        "neutral_label_share": None,
        "negative_label_share": None,
    }

    records = [
        make_aggregation(
            window_hours=6,
            **empty_values,
        ),
        make_aggregation(
            window_hours=24,
            **empty_values,
        ),
        make_aggregation(
            window_hours=72,
            **empty_values,
        ),
    ]

    report = validate(records)

    assert (
        report.status
        is SentimentAggregationStatus.PASSED
    )