"""Strict validation contracts for canonical news datasets.

This module validates already-normalized ArticleRecord objects.

It deliberately contains no provider fetching, storage, sentiment
scoring, duplicate merging, market-data alignment, or network checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)


class NewsStatus(str, Enum):
    """Validation status for a canonical news dataset."""

    PASSED = "PASSED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class NewsIssue:
    """One validation issue discovered in a news dataset."""

    article_id: str | None
    field: str | None
    rule: str
    severity: str
    message: str


@dataclass
class NewsValidationReport:
    """Validation result for one canonical news dataset."""

    article_count: int
    hard_failures: list[NewsIssue] = field(
        default_factory=list
    )
    soft_anomalies: list[NewsIssue] = field(
        default_factory=list
    )
    skipped_checks: list[str] = field(
        default_factory=list
    )

    @property
    def status(self) -> NewsStatus:
        """Return BLOCKED when validation is incomplete or failed."""

        if self.hard_failures or self.skipped_checks:
            return NewsStatus.BLOCKED

        return NewsStatus.PASSED


def _require_aware_utc(
    value: datetime,
    *,
    field_name: str,
) -> None:
    """Require an aware UTC validation timestamp."""

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be timezone-aware"
        )

    if value.utcoffset().total_seconds() != 0:
        raise ValueError(
            f"{field_name} must be UTC"
        )


def validate_articles(
    articles: list[ArticleRecord],
    *,
    allowed_assets: set[str],
    allowed_query_ids: set[str],
    allowed_languages: set[str],
    retrieved_at_utc: datetime,
    max_future_seconds: int,
    skipped_checks: list[str] | None = None,
) -> NewsValidationReport:
    """Validate canonical news records.

    Validation is intentionally strict about identity, configured
    mappings, language, and information-time safety.

    Date-only timestamps and uncertain provenance/licensing are retained
    as soft anomalies because they can still be used later under explicit
    conservative availability rules.

    Any hard failure or skipped validation check blocks the dataset.
    """

    _require_aware_utc(
        retrieved_at_utc,
        field_name="retrieved_at_utc",
    )

    if max_future_seconds < 0:
        raise ValueError(
            "max_future_seconds cannot be negative"
        )

    report = NewsValidationReport(
        article_count=len(articles),
        skipped_checks=list(
            skipped_checks or []
        ),
    )

    if not articles:
        report.hard_failures.append(
            NewsIssue(
                article_id=None,
                field=None,
                rule="empty_dataset",
                severity="hard",
                message=(
                    "Canonical news dataset contains "
                    "no articles."
                ),
            )
        )

        return report

    article_ids: set[str] = set()
    provider_items: set[tuple[str, str]] = set()

    future_limit = (
        retrieved_at_utc
        + timedelta(seconds=max_future_seconds)
    )

    for article in articles:
        unknown_assets = sorted(
            set(article.matched_assets)
            - allowed_assets
        )

        for asset in unknown_assets:
            report.hard_failures.append(
                NewsIssue(
                    article_id=article.article_id,
                    field="matched_assets",
                    rule="unknown_matched_asset",
                    severity="hard",
                    message=(
                        "Article references unknown "
                        f"asset '{asset}'."
                    ),
                )
            )

        if article.query_id not in allowed_query_ids:
            report.hard_failures.append(
                NewsIssue(
                    article_id=article.article_id,
                    field="query_id",
                    rule="unknown_query_id",
                    severity="hard",
                    message=(
                        "Article references unknown "
                        f"query_id '{article.query_id}'."
                    ),
                )
            )

        if article.language not in allowed_languages:
            report.hard_failures.append(
                NewsIssue(
                    article_id=article.article_id,
                    field="language",
                    rule="disallowed_language",
                    severity="hard",
                    message=(
                        "Article language "
                        f"'{article.language}' is not "
                        "allowed."
                    ),
                )
            )

        if article.published_at_utc > future_limit:
            report.hard_failures.append(
                NewsIssue(
                    article_id=article.article_id,
                    field="published_at_utc",
                    rule="future_publication",
                    severity="hard",
                    message=(
                        "Article publication timestamp "
                        "is later than the permitted "
                        "retrieval-time tolerance."
                    ),
                )
            )

        if article.article_id in article_ids:
            report.hard_failures.append(
                NewsIssue(
                    article_id=article.article_id,
                    field="article_id",
                    rule="duplicate_article_id",
                    severity="hard",
                    message=(
                        "Canonical article_id appears "
                        "more than once."
                    ),
                )
            )
        else:
            article_ids.add(article.article_id)

        provider_key = (
            article.provider,
            article.provider_item_id,
        )

        if provider_key in provider_items:
            report.hard_failures.append(
                NewsIssue(
                    article_id=article.article_id,
                    field="provider_item_id",
                    rule="duplicate_provider_item",
                    severity="hard",
                    message=(
                        "Provider and provider_item_id "
                        "combination appears more than "
                        "once."
                    ),
                )
            )
        else:
            provider_items.add(provider_key)

        if (
            article.published_at_precision
            is PublishedAtPrecision.DATE
        ):
            report.soft_anomalies.append(
                NewsIssue(
                    article_id=article.article_id,
                    field="published_at_utc",
                    rule="date_only_timestamp",
                    severity="soft",
                    message=(
                        "Article publication time has "
                        "date-only precision and requires "
                        "conservative downstream "
                        "availability handling."
                    ),
                )
            )

        if (
            article.timestamp_source
            is TimestampSource.UNKNOWN
        ):
            report.soft_anomalies.append(
                NewsIssue(
                    article_id=article.article_id,
                    field="timestamp_source",
                    rule="unknown_timestamp_source",
                    severity="soft",
                    message=(
                        "Article publication timestamp "
                        "source is unknown."
                    ),
                )
            )

        if (
            article.text_license_class
            is TextLicenseClass.UNKNOWN
        ):
            report.soft_anomalies.append(
                NewsIssue(
                    article_id=article.article_id,
                    field="text_license_class",
                    rule="unknown_text_license",
                    severity="soft",
                    message=(
                        "Article text licensing class "
                        "is unknown."
                    ),
                )
            )

    return report