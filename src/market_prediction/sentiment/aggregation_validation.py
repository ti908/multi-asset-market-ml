"""Strict validation for sentiment aggregation datasets.

This module validates already-constructed SentimentAggregationRecord
objects.

Record-level numerical and missing-value invariants belong to the
SentimentAggregationRecord contract. This module validates dataset-level
properties such as allowed assets, configured windows, logical
uniqueness, and complete window coverage.

It performs no sentiment scoring, aggregation, storage, price alignment,
or missing-value imputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from market_prediction.sentiment.aggregation import (
    SentimentAggregationRecord,
)


class SentimentAggregationStatus(str, Enum):
    """Validation status for an aggregation dataset."""

    PASSED = "PASSED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class SentimentAggregationIssue:
    """One validation issue in an aggregation dataset."""

    asset: str | None
    cutoff_at_utc: datetime | None
    window_hours: int | None
    field: str | None
    rule: str
    severity: str
    message: str


@dataclass
class SentimentAggregationValidationReport:
    """Validation result for one aggregation dataset."""

    aggregation_count: int
    hard_failures: list[SentimentAggregationIssue] = field(
        default_factory=list
    )
    soft_anomalies: list[SentimentAggregationIssue] = field(
        default_factory=list
    )
    skipped_checks: list[str] = field(
        default_factory=list
    )

    @property
    def status(self) -> SentimentAggregationStatus:
        """Return BLOCKED when validation failed or is incomplete."""

        if self.hard_failures or self.skipped_checks:
            return SentimentAggregationStatus.BLOCKED

        return SentimentAggregationStatus.PASSED


def validate_sentiment_aggregations(
    records: list[SentimentAggregationRecord],
    *,
    allowed_assets: set[str],
    expected_windows: set[int],
    skipped_checks: list[str] | None = None,
) -> SentimentAggregationValidationReport:
    """Validate canonical sentiment aggregation observations.

    Empty datasets are valid because there may be no requested
    asset/cutoff observations to validate.

    Once an asset/cutoff pair is represented, every configured temporal
    window must be present, including windows containing zero articles.

    Any hard failure or skipped validation check blocks the dataset.
    """

    report = SentimentAggregationValidationReport(
        aggregation_count=len(records),
        skipped_checks=list(skipped_checks or []),
    )

    logical_observations: set[
        tuple[str, datetime, int]
    ] = set()

    observed_windows: dict[
        tuple[str, datetime],
        set[int],
    ] = {}

    for record in records:
        if record.asset not in allowed_assets:
            report.hard_failures.append(
                SentimentAggregationIssue(
                    asset=record.asset,
                    cutoff_at_utc=record.cutoff_at_utc,
                    window_hours=record.window_hours,
                    field="asset",
                    rule="unknown_asset",
                    severity="hard",
                    message=(
                        "Sentiment aggregation references "
                        f"unknown asset '{record.asset}'."
                    ),
                )
            )

        if record.window_hours not in expected_windows:
            report.hard_failures.append(
                SentimentAggregationIssue(
                    asset=record.asset,
                    cutoff_at_utc=record.cutoff_at_utc,
                    window_hours=record.window_hours,
                    field="window_hours",
                    rule="unexpected_window",
                    severity="hard",
                    message=(
                        "Sentiment aggregation window is not "
                        "one of the configured windows."
                    ),
                )
            )

        logical_key = (
            record.asset,
            record.cutoff_at_utc,
            record.window_hours,
        )

        if logical_key in logical_observations:
            report.hard_failures.append(
                SentimentAggregationIssue(
                    asset=record.asset,
                    cutoff_at_utc=record.cutoff_at_utc,
                    window_hours=record.window_hours,
                    field=None,
                    rule="duplicate_logical_aggregation",
                    severity="hard",
                    message=(
                        "Logical sentiment aggregation "
                        "appears more than once."
                    ),
                )
            )
        else:
            logical_observations.add(logical_key)

        coverage_key = (
            record.asset,
            record.cutoff_at_utc,
        )

        observed_windows.setdefault(
            coverage_key,
            set(),
        ).add(record.window_hours)

    for (
        asset,
        cutoff_at_utc,
    ), windows in observed_windows.items():
        missing_windows = expected_windows - windows

        for window_hours in sorted(missing_windows):
            report.hard_failures.append(
                SentimentAggregationIssue(
                    asset=asset,
                    cutoff_at_utc=cutoff_at_utc,
                    window_hours=window_hours,
                    field="window_hours",
                    rule="missing_expected_window",
                    severity="hard",
                    message=(
                        "Sentiment aggregation dataset is "
                        f"missing expected {window_hours}h "
                        "window for this asset and cutoff."
                    ),
                )
            )

    return report
