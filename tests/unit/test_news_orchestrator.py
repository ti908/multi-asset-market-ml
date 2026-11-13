"""Tests for canonical news ingestion orchestration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)
from market_prediction.news.orchestrator import (
    NewsIngestionOrchestrator,
)
from market_prediction.providers.news.base import (
    NewsProvider,
    NewsQuery,
)
from market_prediction.news.storage import (
    NewsStore,
    NewsValidationGateError,
)
from market_prediction.news.validation import (
    NewsStatus,
)
from market_prediction.news.dedup import (
    duplicate_group_identity,
)
from market_prediction.providers.news.exceptions import (
    NewsProviderUnavailableError,
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


class StubNewsProvider(NewsProvider):
    def __init__(
        self,
        articles: list[ArticleRecord],
    ) -> None:
        self._articles = list(articles)
        self.calls: list[
            tuple[
                NewsQuery,
                datetime,
                datetime,
            ]
        ] = []

    def fetch_articles(
        self,
        *,
        query: NewsQuery,
        start_at_utc: datetime,
        end_at_utc: datetime,
    ) -> list[ArticleRecord]:
        self.calls.append(
            (
                query,
                start_at_utc,
                end_at_utc,
            )
        )

        return list(self._articles)

class FailingNewsProvider(NewsProvider):
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                NewsQuery,
                datetime,
                datetime,
            ]
        ] = []

    def fetch_articles(
        self,
        *,
        query: NewsQuery,
        start_at_utc: datetime,
        end_at_utc: datetime,
    ) -> list[ArticleRecord]:
        self.calls.append(
            (
                query,
                start_at_utc,
                end_at_utc,
            )
        )

        raise NewsProviderUnavailableError(
            "Provider unavailable."
        )

def test_run_query_fetches_validates_and_persists_articles(
    tmp_path: Path,
) -> None:
    article = make_article()

    provider = StubNewsProvider(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={"AAPL"},
        allowed_query_ids={
            "aapl_company"
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    result = orchestrator.run_query(
        query=query,
        start_at_utc=start_at,
        end_at_utc=end_at,
    )

    assert provider.calls == [
        (
            query,
            start_at,
            end_at,
        )
    ]

    assert result.provider == "marketaux"
    assert result.asset == "AAPL"
    assert (
        result.query_id
        == "aapl_company"
    )
    assert result.article_count == 1
    assert (
        result.validation_status
        is NewsStatus.PASSED
    )

    loaded = store.read()

    assert len(loaded.articles) == 1

    persisted_article = (
        loaded.articles[0]
    )

    expected_duplicate_group_id = (
        duplicate_group_identity(
            article.canonical_url
        )
    )

    assert (
            persisted_article.duplicate_group_id
            == expected_duplicate_group_id
    )

    assert (
            persisted_article
            == article.model_copy(
        update={
            "duplicate_group_id": (
                expected_duplicate_group_id
            ),
        }
    )
    )

    assert (
            loaded.dataset_id
            == result.dataset_id
    )

def test_run_query_does_not_publish_blocked_dataset(
    tmp_path: Path,
) -> None:
    article = make_article(
        language="fr",
    )

    provider = StubNewsProvider(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={"AAPL"},
        allowed_query_ids={
            "aapl_company"
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_query(
            query=query,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except NewsValidationGateError:
        pass
    else:
        raise AssertionError(
            "Invalid news dataset must "
            "raise NewsValidationGateError."
        )

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_query_does_not_publish_empty_dataset(
    tmp_path: Path,
) -> None:
    provider = StubNewsProvider([])

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={"AAPL"},
        allowed_query_ids={
            "aapl_company"
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_query(
            query=query,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except NewsValidationGateError:
        pass
    else:
        raise AssertionError(
            "Invalid news dataset must "
            "raise NewsValidationGateError."
        )

    assert provider.calls == [
        (
            query,
            start_at,
            end_at,
        )
    ]

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_query_propagates_provider_failure_without_publication(
    tmp_path: Path,
) -> None:
    provider = FailingNewsProvider()

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={"AAPL"},
        allowed_query_ids={
            "aapl_company"
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_query(
            query=query,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except NewsProviderUnavailableError:
        pass
    else:
        raise AssertionError(
            "Provider failure must propagate."
        )

    assert provider.calls == [
        (
            query,
            start_at,
            end_at,
        )
    ]

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_queries_aggregates_and_publishes_once(
    tmp_path: Path,
) -> None:
    aapl_article = make_article(
        article_id="a" * 64,
        matched_assets=["AAPL"],
        query_id="aapl_company",
        canonical_url=(
            "https://example.com/apple-earnings"
        ),
        provider_item_id="apple-123",
    )

    msft_article = make_article(
        article_id="c" * 64,
        matched_assets=["MSFT"],
        query_id="msft_company",
        title="Microsoft reports quarterly earnings",
        canonical_url=(
            "https://example.com/microsoft-earnings"
        ),
        provider_item_id="microsoft-123",
    )

    class QueryAwareProvider(NewsProvider):
        def __init__(self) -> None:
            self.calls: list[
                tuple[
                    NewsQuery,
                    datetime,
                    datetime,
                ]
            ] = []

        def fetch_articles(
            self,
            *,
            query: NewsQuery,
            start_at_utc: datetime,
            end_at_utc: datetime,
        ) -> list[ArticleRecord]:
            self.calls.append(
                (
                    query,
                    start_at_utc,
                    end_at_utc,
                )
            )

            if query.asset == "AAPL":
                return [aapl_article]

            if query.asset == "MSFT":
                return [msft_article]

            raise AssertionError(
                f"Unexpected asset: {query.asset}"
            )

    provider = QueryAwareProvider()

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={
            "AAPL",
            "MSFT",
        },
        allowed_query_ids={
            "aapl_company",
            "msft_company",
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    aapl_query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    msft_query = NewsQuery(
        asset="MSFT",
        query_id="msft_company",
        strict_terms=(
            "MSFT",
            "Microsoft",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    result = orchestrator.run_queries(
        queries=[
            aapl_query,
            msft_query,
        ],
        start_at_utc=start_at,
        end_at_utc=end_at,
    )

    assert provider.calls == [
        (
            aapl_query,
            start_at,
            end_at,
        ),
        (
            msft_query,
            start_at,
            end_at,
        ),
    ]

    assert result.article_count == 2
    assert (
        result.validation_status
        is NewsStatus.PASSED
    )

    loaded = store.read()

    assert len(
        loaded.articles
    ) == 2

    assert [
        article.article_id
        for article in loaded.articles
    ] == [
        "a" * 64,
        "c" * 64,
    ]

    assert all(
        article.duplicate_group_id
        is not None
        for article in loaded.articles
    )

    assert (
        loaded.dataset_id
        == result.dataset_id
    )

    generation_directories = list(
        (
            tmp_path
            / "generations"
        ).iterdir()
    )

    assert len(
        generation_directories
    ) == 1

def test_run_queries_provider_failure_prevents_partial_publication(
    tmp_path: Path,
) -> None:
    aapl_article = make_article(
        article_id="a" * 64,
        matched_assets=["AAPL"],
        query_id="aapl_company",
        canonical_url=(
            "https://example.com/apple-earnings"
        ),
        provider_item_id="apple-123",
    )

    class PartiallyFailingProvider(
        NewsProvider
    ):
        def __init__(self) -> None:
            self.calls: list[
                tuple[
                    NewsQuery,
                    datetime,
                    datetime,
                ]
            ] = []

        def fetch_articles(
            self,
            *,
            query: NewsQuery,
            start_at_utc: datetime,
            end_at_utc: datetime,
        ) -> list[ArticleRecord]:
            self.calls.append(
                (
                    query,
                    start_at_utc,
                    end_at_utc,
                )
            )

            if query.asset == "AAPL":
                return [aapl_article]

            if query.asset == "MSFT":
                raise (
                    NewsProviderUnavailableError(
                        "Provider unavailable."
                    )
                )

            raise AssertionError(
                f"Unexpected asset: {query.asset}"
            )

    provider = PartiallyFailingProvider()

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={
            "AAPL",
            "MSFT",
        },
        allowed_query_ids={
            "aapl_company",
            "msft_company",
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    aapl_query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    msft_query = NewsQuery(
        asset="MSFT",
        query_id="msft_company",
        strict_terms=(
            "MSFT",
            "Microsoft",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_queries(
            queries=[
                aapl_query,
                msft_query,
            ],
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except NewsProviderUnavailableError:
        pass
    else:
        raise AssertionError(
            "A provider failure during a "
            "multi-query run must propagate."
        )

    assert provider.calls == [
        (
            aapl_query,
            start_at,
            end_at,
        ),
        (
            msft_query,
            start_at,
            end_at,
        ),
    ]

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_queries_blocked_validation_prevents_publication(
    tmp_path: Path,
) -> None:
    aapl_article = make_article(
        article_id="a" * 64,
        matched_assets=["AAPL"],
        query_id="aapl_company",
        canonical_url=(
            "https://example.com/apple-earnings"
        ),
        provider_item_id="apple-123",
    )

    msft_article = make_article(
        article_id="c" * 64,
        matched_assets=["MSFT"],
        query_id="msft_company",
        title="Microsoft reports quarterly earnings",
        canonical_url=(
            "https://example.com/microsoft-earnings"
        ),
        provider_item_id="microsoft-123",
        language="fr",
    )

    class InvalidSecondQueryProvider(
        NewsProvider
    ):
        def __init__(self) -> None:
            self.calls: list[
                tuple[
                    NewsQuery,
                    datetime,
                    datetime,
                ]
            ] = []

        def fetch_articles(
            self,
            *,
            query: NewsQuery,
            start_at_utc: datetime,
            end_at_utc: datetime,
        ) -> list[ArticleRecord]:
            self.calls.append(
                (
                    query,
                    start_at_utc,
                    end_at_utc,
                )
            )

            if query.asset == "AAPL":
                return [aapl_article]

            if query.asset == "MSFT":
                return [msft_article]

            raise AssertionError(
                f"Unexpected asset: {query.asset}"
            )

    provider = InvalidSecondQueryProvider()

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={
            "AAPL",
            "MSFT",
        },
        allowed_query_ids={
            "aapl_company",
            "msft_company",
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    aapl_query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    msft_query = NewsQuery(
        asset="MSFT",
        query_id="msft_company",
        strict_terms=(
            "MSFT",
            "Microsoft",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_queries(
            queries=[
                aapl_query,
                msft_query,
            ],
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except NewsValidationGateError:
        pass
    else:
        raise AssertionError(
            "Blocked combined validation "
            "must prevent publication."
        )

    assert provider.calls == [
        (
            aapl_query,
            start_at,
            end_at,
        ),
        (
            msft_query,
            start_at,
            end_at,
        ),
    ]

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_query_rejects_mixed_providers_before_publication(
    tmp_path: Path,
) -> None:
    marketaux_article = make_article(
        article_id="a" * 64,
        provider="marketaux",
        provider_item_id="marketaux-123",
        canonical_url=(
            "https://example.com/article-one"
        ),
    )

    other_article = make_article(
        article_id="c" * 64,
        provider="other-provider",
        provider_item_id="other-123",
        canonical_url=(
            "https://example.com/article-two"
        ),
    )

    provider = StubNewsProvider(
        [
            marketaux_article,
            other_article,
        ]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={"AAPL"},
        allowed_query_ids={
            "aapl_company"
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_query(
            query=query,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except ValueError as exc:
        assert (
            "exactly one provider"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Mixed-provider query result "
            "must be rejected."
        )

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_query_rejects_non_increasing_window_before_provider_call(
    tmp_path: Path,
) -> None:
    article = make_article()

    provider = StubNewsProvider(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={"AAPL"},
        allowed_query_ids={
            "aapl_company"
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_query(
            query=query,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except ValueError as exc:
        assert (
            "start_at_utc must be before "
            "end_at_utc"
            in str(exc)
        )
    else:
        raise AssertionError(
            "A non-increasing ingestion "
            "window must be rejected."
        )

    assert provider.calls == []

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_query_rejects_naive_window_before_provider_call(
    tmp_path: Path,
) -> None:
    provider = StubNewsProvider(
        [make_article()]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={"AAPL"},
        allowed_query_ids={
            "aapl_company"
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_query(
            query=query,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except ValueError as exc:
        assert (
            "start_at_utc must be "
            "timezone-aware UTC"
            in str(exc)
        )
    else:
        raise AssertionError(
            "A naive ingestion timestamp "
            "must be rejected."
        )

    assert provider.calls == []

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()


def test_run_query_rejects_non_utc_window_before_provider_call(
    tmp_path: Path,
) -> None:
    provider = StubNewsProvider(
        [make_article()]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={"AAPL"},
        allowed_query_ids={
            "aapl_company"
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    non_utc = timezone(
        timedelta(hours=1)
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=non_utc,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_query(
            query=query,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except ValueError as exc:
        assert (
            "start_at_utc must be UTC"
            in str(exc)
        )
    else:
        raise AssertionError(
            "A non-UTC ingestion timestamp "
            "must be rejected."
        )

    assert provider.calls == []

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_queries_rejects_invalid_window_before_provider_calls(
    tmp_path: Path,
) -> None:
    provider = StubNewsProvider(
        [make_article()]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={
            "AAPL",
            "MSFT",
        },
        allowed_query_ids={
            "aapl_company",
            "msft_company",
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    queries = [
        NewsQuery(
            asset="AAPL",
            query_id="aapl_company",
            strict_terms=(
                "AAPL",
                "Apple Inc",
            ),
            context_terms=(),
        ),
        NewsQuery(
            asset="MSFT",
            query_id="msft_company",
            strict_terms=(
                "MSFT",
                "Microsoft",
            ),
            context_terms=(),
        ),
    ]

    start_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_queries(
            queries=queries,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except ValueError as exc:
        assert (
            "start_at_utc must be before "
            "end_at_utc"
            in str(exc)
        )
    else:
        raise AssertionError(
            "An invalid multi-query ingestion "
            "window must be rejected."
        )

    assert provider.calls == []

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_query_rejects_mismatched_query_id_before_publication(
    tmp_path: Path,
) -> None:
    article = make_article(
        matched_assets=["MSFT"],
        query_id="msft_company",
    )

    provider = StubNewsProvider(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={
            "AAPL",
            "MSFT",
        },
        allowed_query_ids={
            "aapl_company",
            "msft_company",
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_query(
            query=query,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except ValueError as exc:
        assert (
            "article query_id must match "
            "the fetching query"
            in str(exc)
        )
    else:
        raise AssertionError(
            "An article returned for the wrong "
            "query_id must be rejected."
        )

    assert provider.calls == [
        (
            query,
            start_at,
            end_at,
        )
    ]

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_query_rejects_missing_requested_asset_before_publication(
    tmp_path: Path,
) -> None:
    article = make_article(
        matched_assets=["MSFT"],
        query_id="aapl_company",
    )

    provider = StubNewsProvider(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={
            "AAPL",
            "MSFT",
        },
        allowed_query_ids={
            "aapl_company",
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_query(
            query=query,
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except ValueError as exc:
        assert (
            "fetching query asset must be "
            "present in article matched_assets"
            in str(exc)
        )
    else:
        raise AssertionError(
            "An article missing the requested "
            "asset must be rejected."
        )

    assert provider.calls == [
        (
            query,
            start_at,
            end_at,
        )
    ]

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()

def test_run_queries_semantic_mismatch_prevents_partial_publication(
    tmp_path: Path,
) -> None:
    aapl_article = make_article()

    msft_wrong_article = make_article(
        article_id="article-msft-wrong",
        matched_assets=["AAPL"],
        query_id="msft_company",
        canonical_url=(
            "https://example.com/"
            "msft-wrong-asset"
        ),
        provider_item_id="item-msft-wrong",
    )

    class QueryAwareProvider(NewsProvider):
        def __init__(self) -> None:
            self.calls: list[
                tuple[
                    NewsQuery,
                    datetime,
                    datetime,
                ]
            ] = []

        def fetch_articles(
            self,
            *,
            query: NewsQuery,
            start_at_utc: datetime,
            end_at_utc: datetime,
        ) -> list[ArticleRecord]:
            self.calls.append(
                (
                    query,
                    start_at_utc,
                    end_at_utc,
                )
            )

            if query.query_id == "aapl_company":
                return [aapl_article]

            if query.query_id == "msft_company":
                return [msft_wrong_article]

            raise AssertionError(
                "Unexpected query."
            )

    provider = QueryAwareProvider()

    store = NewsStore(
        storage_dir=tmp_path,
    )

    retrieved_at = datetime(
        2026,
        10,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets={
            "AAPL",
            "MSFT",
        },
        allowed_query_ids={
            "aapl_company",
            "msft_company",
        },
        allowed_languages={"en"},
        now=lambda: retrieved_at,
    )

    aapl_query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )

    msft_query = NewsQuery(
        asset="MSFT",
        query_id="msft_company",
        strict_terms=(
            "MSFT",
            "Microsoft",
        ),
        context_terms=(),
    )

    start_at = datetime(
        2026,
        10,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    end_at = datetime(
        2026,
        10,
        31,
        0,
        0,
        tzinfo=UTC,
    )

    try:
        orchestrator.run_queries(
            queries=[
                aapl_query,
                msft_query,
            ],
            start_at_utc=start_at,
            end_at_utc=end_at,
        )
    except ValueError as exc:
        assert (
            "fetching query asset must be "
            "present in article matched_assets"
            in str(exc)
        )
    else:
        raise AssertionError(
            "A semantic mismatch in one query "
            "must reject the entire run."
        )

    assert provider.calls == [
        (
            aapl_query,
            start_at,
            end_at,
        ),
        (
            msft_query,
            start_at,
            end_at,
        ),
    ]

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()