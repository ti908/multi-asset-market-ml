"""Strict validation contracts for canonical sentiment datasets.

This module validates already-normalized SentimentRecord objects.

It deliberately contains no article fetching, sentiment scoring, storage,
market-data alignment, temporal aggregation, or network checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from market_prediction.sentiment.models import (
    SentimentRecord,
)


class SentimentStatus(str, Enum):
    """Validation status for a canonical sentiment dataset."""

    PASSED = "PASSED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class SentimentIssue:
    """One validation issue discovered in a sentiment dataset."""

    sentiment_id: str | None
    field: str | None
    rule: str
    severity: str
    message: str


@dataclass
class SentimentValidationReport:
    """Validation result for one canonical sentiment dataset."""

    sentiment_count: int
    hard_failures: list[SentimentIssue] = field(
        default_factory=list
    )
    soft_anomalies: list[SentimentIssue] = field(
        default_factory=list
    )
    skipped_checks: list[str] = field(
        default_factory=list
    )

    @property
    def status(self) -> SentimentStatus:
        """Return BLOCKED when validation is incomplete or failed."""

        if self.hard_failures or self.skipped_checks:
            return SentimentStatus.BLOCKED

        return SentimentStatus.PASSED


def validate_sentiments(
    sentiments: list[SentimentRecord],
    *,
    allowed_assets: set[str],
    allowed_query_ids: set[str],
    expected_model_name: str,
    expected_model_revision: str,
    expected_scoring_method: str,
    expected_schema_version: str,
    skipped_checks: list[str] | None = None,
) -> SentimentValidationReport:
    """Validate canonical sentiment observations.

    Empty datasets are valid. Zero sentiment observations means that no
    source articles were available for scoring; it must not be converted
    into an artificial neutral sentiment observation.

    Any hard failure or skipped validation check blocks the dataset.
    """

    report = SentimentValidationReport(
        sentiment_count=len(sentiments),
        skipped_checks=list(
            skipped_checks or []
        ),
    )

    sentiment_ids: set[str] = set()

    logical_observations: set[
        tuple[str, str, str, str, str]
    ] = set()

    for sentiment in sentiments:
        if sentiment.asset not in allowed_assets:
            report.hard_failures.append(
                SentimentIssue(
                    sentiment_id=sentiment.sentiment_id,
                    field="asset",
                    rule="unknown_asset",
                    severity="hard",
                    message=(
                        "Sentiment observation references "
                        f"unknown asset '{sentiment.asset}'."
                    ),
                )
            )

        if sentiment.query_id not in allowed_query_ids:
            report.hard_failures.append(
                SentimentIssue(
                    sentiment_id=sentiment.sentiment_id,
                    field="query_id",
                    rule="unknown_query_id",
                    severity="hard",
                    message=(
                        "Sentiment observation references "
                        f"unknown query_id "
                        f"'{sentiment.query_id}'."
                    ),
                )
            )

        if sentiment.sentiment_id in sentiment_ids:
            report.hard_failures.append(
                SentimentIssue(
                    sentiment_id=sentiment.sentiment_id,
                    field="sentiment_id",
                    rule="duplicate_sentiment_id",
                    severity="hard",
                    message=(
                        "Canonical sentiment_id appears "
                        "more than once."
                    ),
                )
            )
        else:
            sentiment_ids.add(
                sentiment.sentiment_id
            )

        logical_key = (
            sentiment.article_id,
            sentiment.asset,
            sentiment.model_name,
            sentiment.model_revision,
            sentiment.scoring_method,
        )

        if logical_key in logical_observations:
            report.hard_failures.append(
                SentimentIssue(
                    sentiment_id=sentiment.sentiment_id,
                    field=None,
                    rule="duplicate_logical_observation",
                    severity="hard",
                    message=(
                        "Logical sentiment observation "
                        "appears more than once."
                    ),
                )
            )
        else:
            logical_observations.add(logical_key)

        if sentiment.model_name != expected_model_name:
            report.hard_failures.append(
                SentimentIssue(
                    sentiment_id=sentiment.sentiment_id,
                    field="model_name",
                    rule="unexpected_model_name",
                    severity="hard",
                    message=(
                        "Sentiment model_name does not "
                        "match the expected model."
                    ),
                )
            )

        if (
            sentiment.model_revision
            != expected_model_revision
        ):
            report.hard_failures.append(
                SentimentIssue(
                    sentiment_id=sentiment.sentiment_id,
                    field="model_revision",
                    rule="unexpected_model_revision",
                    severity="hard",
                    message=(
                        "Sentiment model_revision does not "
                        "match the expected revision."
                    ),
                )
            )

        if (
            sentiment.scoring_method
            != expected_scoring_method
        ):
            report.hard_failures.append(
                SentimentIssue(
                    sentiment_id=sentiment.sentiment_id,
                    field="scoring_method",
                    rule="unexpected_scoring_method",
                    severity="hard",
                    message=(
                        "Sentiment scoring_method does not "
                        "match the expected method."
                    ),
                )
            )

        if (
            sentiment.schema_version
            != expected_schema_version
        ):
            report.hard_failures.append(
                SentimentIssue(
                    sentiment_id=sentiment.sentiment_id,
                    field="schema_version",
                    rule="unexpected_schema_version",
                    severity="hard",
                    message=(
                        "Sentiment schema_version does not "
                        "match the expected version."
                    ),
                )
            )

    return report