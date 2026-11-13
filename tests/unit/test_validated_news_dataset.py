"""Tests for the validated news dataset persistence gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)
from market_prediction.news.storage import (
    NewsValidationGateError,
    ValidatedNewsDataset,
    news_dataset_fingerprint,
)
from market_prediction.news.validation import (
    NewsIssue,
    NewsValidationReport,
)


UTC = timezone.utc


def make_article(
    **overrides,
) -> ArticleRecord:
    published = datetime(
        2026,
        10,
        30,
        12,
        0,
        tzinfo=UTC,
    )

    values = {
        "article_id": "a" * 64,
        "matched_assets": ["AAPL"],
        "query_id": "aapl_company",
        "title": "Apple reports quarterly earnings",
        "snippet": "Revenue increased during the quarter.",
        "full_text": None,
        "canonical_url": (
            "https://example.com/apple-earnings"
        ),
        "publisher": "Example News",
        "published_at_utc": published,
        "published_at_precision": (
            PublishedAtPrecision.SECOND
        ),
        "original_timezone": "UTC",
        "provider_seen_at_utc": (
            published + timedelta(minutes=1)
        ),
        "retrieved_at_utc": (
            published + timedelta(minutes=2)
        ),
        "provider": "marketaux",
        "provider_item_id": "article-123",
        "language": "en",
        "content_hash": "b" * 64,
        "duplicate_group_id": None,
        "raw_payload_reference": None,
        "schema_version": "1",
        "timestamp_source": (
            TimestampSource.PROVIDER_PUBLICATION
        ),
        "text_license_class": (
            TextLicenseClass.PROVIDER_METADATA
        ),
    }

    values.update(overrides)

    return ArticleRecord(**values)


def make_passed_report(
    articles: list[ArticleRecord],
) -> NewsValidationReport:
    return NewsValidationReport(
        article_count=len(articles),
        dataset_fingerprint=(
            news_dataset_fingerprint(articles)
        ),
    )


def test_validated_news_dataset_accepts_passed_dataset() -> None:
    articles = [make_article()]
    report = make_passed_report(articles)

    dataset = ValidatedNewsDataset.from_validation(
        articles,
        report,
    )

    assert dataset.articles == articles
    assert dataset.report is report
    assert (
        dataset.dataset_id
        == news_dataset_fingerprint(articles)
    )


def test_validated_news_dataset_cannot_be_constructed_directly() -> None:
    articles = [make_article()]
    report = make_passed_report(articles)

    with pytest.raises(NewsValidationGateError):
        ValidatedNewsDataset(
            articles,
            report,
        )


def test_blocked_report_is_rejected() -> None:
    articles = [make_article()]

    report = NewsValidationReport(
        article_count=1,
        dataset_fingerprint=(
            news_dataset_fingerprint(articles)
        ),
        hard_failures=[
            NewsIssue(
                article_id=articles[0].article_id,
                field="language",
                rule="disallowed_language",
                severity="hard",
                message="Language is not allowed.",
            )
        ],
    )

    with pytest.raises(NewsValidationGateError):
        ValidatedNewsDataset.from_validation(
            articles,
            report,
        )


def test_skipped_validation_report_is_rejected() -> None:
    articles = [make_article()]

    report = NewsValidationReport(
        article_count=1,
        dataset_fingerprint=(
            news_dataset_fingerprint(articles)
        ),
        skipped_checks=["language_check"],
    )

    with pytest.raises(NewsValidationGateError):
        ValidatedNewsDataset.from_validation(
            articles,
            report,
        )


def test_article_count_mismatch_is_rejected() -> None:
    articles = [make_article()]

    report = NewsValidationReport(
        article_count=2,
        dataset_fingerprint=(
            news_dataset_fingerprint(articles)
        ),
    )

    with pytest.raises(NewsValidationGateError):
        ValidatedNewsDataset.from_validation(
            articles,
            report,
        )


def test_missing_validation_fingerprint_is_rejected() -> None:
    articles = [make_article()]

    report = NewsValidationReport(
        article_count=1,
        dataset_fingerprint=None,
    )

    with pytest.raises(NewsValidationGateError):
        ValidatedNewsDataset.from_validation(
            articles,
            report,
        )


def test_changed_content_after_validation_is_rejected() -> None:
    original = make_article()
    report = make_passed_report([original])

    changed = original.model_copy(
        update={
            "title": "Changed after validation",
        }
    )

    with pytest.raises(NewsValidationGateError):
        ValidatedNewsDataset.from_validation(
            [changed],
            report,
        )


def test_article_order_change_after_validation_is_rejected() -> None:
    first = make_article()

    second = make_article(
        article_id="c" * 64,
        provider_item_id="article-456",
        canonical_url="https://example.com/second",
        title="Second article",
    )

    report = make_passed_report(
        [first, second]
    )

    with pytest.raises(NewsValidationGateError):
        ValidatedNewsDataset.from_validation(
            [second, first],
            report,
        )


def test_articles_property_returns_defensive_list_copy() -> None:
    article = make_article()
    articles = [article]
    report = make_passed_report(articles)

    dataset = ValidatedNewsDataset.from_validation(
        articles,
        report,
    )

    returned = dataset.articles
    returned.clear()

    assert dataset.articles == [article]


def test_mutating_original_list_does_not_change_dataset() -> None:
    article = make_article()
    articles = [article]
    report = make_passed_report(articles)

    dataset = ValidatedNewsDataset.from_validation(
        articles,
        report,
    )

    articles.clear()

    assert dataset.articles == [article]
    assert (
        dataset.dataset_id
        == news_dataset_fingerprint([article])
    )