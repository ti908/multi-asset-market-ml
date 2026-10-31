"""Tests for strict canonical news validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)
from market_prediction.news.validation import (
    NewsStatus,
    validate_articles,
)


UTC = timezone.utc


def make_article(
    **overrides,
) -> ArticleRecord:
    """Return one valid canonical article."""

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


def test_valid_articles_pass() -> None:
    article = make_article()

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.PASSED
    assert report.article_count == 1
    assert report.hard_failures == []
    assert report.soft_anomalies == []
    assert report.skipped_checks == []


def test_empty_dataset_is_blocked() -> None:
    report = validate_articles(
        [],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.BLOCKED

    assert any(
        issue.rule == "empty_dataset"
        for issue in report.hard_failures
    )


def test_unknown_matched_asset_is_hard_failure() -> None:
    article = make_article(
        matched_assets=["UNKNOWN"],
    )

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.BLOCKED

    assert any(
        issue.rule == "unknown_matched_asset"
        for issue in report.hard_failures
    )


def test_unknown_query_id_is_hard_failure() -> None:
    article = make_article(
        query_id="unknown_query",
    )

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.BLOCKED

    assert any(
        issue.rule == "unknown_query_id"
        for issue in report.hard_failures
    )


def test_disallowed_language_is_hard_failure() -> None:
    article = make_article(
        language="fr",
    )

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.BLOCKED

    assert any(
        issue.rule == "disallowed_language"
        for issue in report.hard_failures
    )


def test_future_publication_is_hard_failure() -> None:
    retrieval_time = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    article = make_article(
        published_at_utc=(
            retrieval_time + timedelta(seconds=1)
        ),
        provider_seen_at_utc=(
            retrieval_time + timedelta(seconds=2)
        ),
        retrieved_at_utc=(
            retrieval_time + timedelta(seconds=3)
        ),
    )

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=retrieval_time,
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.BLOCKED

    assert any(
        issue.rule == "future_publication"
        for issue in report.hard_failures
    )


def test_future_tolerance_is_respected() -> None:
    retrieval_time = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    article = make_article(
        published_at_utc=(
            retrieval_time + timedelta(seconds=5)
        ),
        provider_seen_at_utc=(
            retrieval_time + timedelta(seconds=6)
        ),
        retrieved_at_utc=(
            retrieval_time + timedelta(seconds=7)
        ),
    )

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=retrieval_time,
        max_future_seconds=5,
    )

    assert not any(
        issue.rule == "future_publication"
        for issue in report.hard_failures
    )


def test_duplicate_article_id_is_hard_failure() -> None:
    first = make_article()

    second = make_article(
        provider_item_id="article-456",
    )

    report = validate_articles(
        [first, second],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.BLOCKED

    assert any(
        issue.rule == "duplicate_article_id"
        for issue in report.hard_failures
    )


def test_duplicate_provider_item_is_hard_failure() -> None:
    first = make_article(
        article_id="a" * 64,
    )

    second = make_article(
        article_id="c" * 64,
    )

    report = validate_articles(
        [first, second],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.BLOCKED

    assert any(
        issue.rule == "duplicate_provider_item"
        for issue in report.hard_failures
    )


def test_date_precision_is_soft_anomaly() -> None:
    article = make_article(
        published_at_precision=(
            PublishedAtPrecision.DATE
        ),
        timestamp_source=TimestampSource.DATE_ONLY,
    )

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.PASSED

    assert any(
        issue.rule == "date_only_timestamp"
        for issue in report.soft_anomalies
    )


def test_unknown_timestamp_source_is_soft_anomaly() -> None:
    article = make_article(
        timestamp_source=TimestampSource.UNKNOWN,
    )

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.PASSED

    assert any(
        issue.rule == "unknown_timestamp_source"
        for issue in report.soft_anomalies
    )


def test_unknown_text_license_is_soft_anomaly() -> None:
    article = make_article(
        text_license_class=TextLicenseClass.UNKNOWN,
    )

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.PASSED

    assert any(
        issue.rule == "unknown_text_license"
        for issue in report.soft_anomalies
    )


def test_report_is_blocked_when_check_is_skipped() -> None:
    article = make_article()

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
        skipped_checks=[
            "canonical_url_reachability",
        ],
    )

    assert report.status is NewsStatus.BLOCKED
    assert report.hard_failures == []

    assert report.skipped_checks == [
        "canonical_url_reachability",
    ]

def test_validation_rejects_naive_retrieval_boundary() -> None:
    article = make_article()

    naive_retrieval = datetime(
        2026,
        10,
        30,
        13,
        0,
    )

    try:
        validate_articles(
            [article],
            allowed_assets={"AAPL"},
            allowed_query_ids={"aapl_company"},
            allowed_languages={"en"},
            retrieved_at_utc=naive_retrieval,
            max_future_seconds=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected naive retrieval boundary "
            "to be rejected"
        )


def test_validation_rejects_non_utc_retrieval_boundary() -> None:
    article = make_article()

    non_utc = timezone(
        timedelta(hours=9)
    )

    retrieval_time = datetime(
        2026,
        10,
        30,
        22,
        0,
        tzinfo=non_utc,
    )

    try:
        validate_articles(
            [article],
            allowed_assets={"AAPL"},
            allowed_query_ids={"aapl_company"},
            allowed_languages={"en"},
            retrieved_at_utc=retrieval_time,
            max_future_seconds=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected non-UTC retrieval boundary "
            "to be rejected"
        )


def test_validation_rejects_negative_future_tolerance() -> None:
    article = make_article()

    try:
        validate_articles(
            [article],
            allowed_assets={"AAPL"},
            allowed_query_ids={"aapl_company"},
            allowed_languages={"en"},
            retrieved_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
                tzinfo=UTC,
            ),
            max_future_seconds=-1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected negative future tolerance "
            "to be rejected"
        )


def test_same_provider_item_id_across_providers_is_allowed() -> None:
    first = make_article(
        article_id="a" * 64,
        provider="marketaux",
        provider_item_id="article-123",
    )

    second = make_article(
        article_id="c" * 64,
        provider="gdelt",
        provider_item_id="article-123",
    )

    report = validate_articles(
        [first, second],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.PASSED

    assert not any(
        issue.rule == "duplicate_provider_item"
        for issue in report.hard_failures
    )


def test_soft_anomalies_do_not_block_dataset() -> None:
    article = make_article(
        published_at_precision=PublishedAtPrecision.DATE,
        timestamp_source=TimestampSource.UNKNOWN,
        text_license_class=TextLicenseClass.UNKNOWN,
    )

    report = validate_articles(
        [article],
        allowed_assets={"AAPL"},
        allowed_query_ids={"aapl_company"},
        allowed_languages={"en"},
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    assert report.status is NewsStatus.PASSED
    assert report.hard_failures == []

    rules = {
        issue.rule
        for issue in report.soft_anomalies
    }

    assert rules == {
        "date_only_timestamp",
        "unknown_timestamp_source",
        "unknown_text_license",
    }