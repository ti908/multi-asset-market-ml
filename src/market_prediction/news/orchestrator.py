"""Canonical news ingestion orchestration.

This module connects canonical news providers, validation, the
ValidatedNewsDataset persistence gate, and NewsStore.

GDELT discovery candidates are intentionally outside this orchestration
path until they have been enriched into canonical ArticleRecord objects.

This layer does not perform sentiment scoring, feature engineering,
price/news alignment, or modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from market_prediction.news.dedup import (
    assign_duplicate_groups,
)
from market_prediction.news.models import (
    ArticleRecord,
)
from market_prediction.news.storage import (
    NewsStore,
    ValidatedNewsDataset,
)
from market_prediction.news.validation import (
    NewsStatus,
    validate_articles,
)
from market_prediction.providers.news.base import (
    NewsProvider,
    NewsQuery,
)


@dataclass(frozen=True)
class NewsIngestionResult:
    """Observable result of one canonical news query ingestion."""

    provider: str
    asset: str
    query_id: str
    article_count: int
    validation_status: NewsStatus
    dataset_id: str


@dataclass(frozen=True)
class NewsIngestionRunResult:
    """Observable result of one multi-query news ingestion run."""

    article_count: int
    validation_status: NewsStatus
    dataset_id: str


@dataclass(frozen=True)
class _PublishedNewsDataset:
    """Internal result of validated atomic publication."""

    article_count: int
    validation_status: NewsStatus
    dataset_id: str


class NewsIngestionOrchestrator:
    """Fetch, validate, and atomically persist canonical news."""

    def __init__(
        self,
        *,
        provider: NewsProvider,
        store: NewsStore,
        allowed_assets: set[str],
        allowed_query_ids: set[str],
        allowed_languages: set[str],
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._allowed_assets = set(
            allowed_assets
        )
        self._allowed_query_ids = set(
            allowed_query_ids
        )
        self._allowed_languages = set(
            allowed_languages
        )
        self._now = now or (
            lambda: datetime.now(timezone.utc)
        )

    def run_query(
        self,
        *,
        query: NewsQuery,
        start_at_utc: datetime,
        end_at_utc: datetime,
    ) -> NewsIngestionResult:
        """Fetch and atomically publish one canonical news query."""
        self._validate_window(
            start_at_utc=start_at_utc,
            end_at_utc=end_at_utc,
        )

        retrieved_at_utc = self._utc_now()

        articles = (
            self._provider.fetch_articles(
                query=query,
                start_at_utc=start_at_utc,
                end_at_utc=end_at_utc,
            )
        )

        self._validate_query_articles(
            query=query,
            articles=articles,
        )

        providers = {
            article.provider
            for article in articles
        }

        if len(providers) > 1:
            raise ValueError(
                "One canonical news query must "
                "return articles from exactly "
                "one provider."
            )

        published = self._publish_articles(
            articles=articles,
            retrieved_at_utc=retrieved_at_utc,
        )

        provider = next(
            iter(providers)
        )

        return NewsIngestionResult(
            provider=provider,
            asset=query.asset,
            query_id=query.query_id,
            article_count=(
                published.article_count
            ),
            validation_status=(
                published.validation_status
            ),
            dataset_id=published.dataset_id,
        )

    def run_queries(
        self,
        *,
        queries: list[NewsQuery],
        start_at_utc: datetime,
        end_at_utc: datetime,
    ) -> NewsIngestionRunResult:
        """Fetch and atomically publish one multi-query dataset."""
        if not isinstance(
            queries,
            list,
        ):
            raise TypeError(
                "queries must be a list."
            )

        if not queries:
            raise ValueError(
                "queries must not be empty."
            )
        self._validate_window(
            start_at_utc=start_at_utc,
            end_at_utc=end_at_utc,
        )

        retrieved_at_utc = self._utc_now()

        articles: list[ArticleRecord] = []

        for query in queries:
            if not isinstance(
                query,
                NewsQuery,
            ):
                raise TypeError(
                    "Every query must be "
                    "a NewsQuery."
                )

            query_articles = (
                self._provider.fetch_articles(
                    query=query,
                    start_at_utc=start_at_utc,
                    end_at_utc=end_at_utc,
                )
            )

            self._validate_query_articles(
                query=query,
                articles=query_articles,
            )

            articles.extend(
                query_articles
            )

        published = self._publish_articles(
            articles=articles,
            retrieved_at_utc=retrieved_at_utc,
        )

        return NewsIngestionRunResult(
            article_count=(
                published.article_count
            ),
            validation_status=(
                published.validation_status
            ),
            dataset_id=published.dataset_id,
        )

    @staticmethod
    def _validate_query_articles(
        *,
        query: NewsQuery,
        articles: list[ArticleRecord],
    ) -> None:
        """Require provider results to belong to their fetching query."""
        for article in articles:
            if article.query_id != query.query_id:
                raise ValueError(
                    "article query_id must match "
                    "the fetching query."
                )

            if query.asset not in article.matched_assets:
                raise ValueError(
                    "fetching query asset must be "
                    "present in article matched_assets."
                )


    def _publish_articles(
        self,
        *,
        articles: list[ArticleRecord],
        retrieved_at_utc: datetime,
    ) -> _PublishedNewsDataset:
        """Validate and atomically publish one complete dataset."""
        articles = assign_duplicate_groups(
            articles
        )

        report = validate_articles(
            articles,
            allowed_assets=self._allowed_assets,
            allowed_query_ids=(
                self._allowed_query_ids
            ),
            allowed_languages=(
                self._allowed_languages
            ),
            retrieved_at_utc=(
                retrieved_at_utc
            ),
            max_future_seconds=0,
        )

        validated = (
            ValidatedNewsDataset.from_validation(
                articles,
                report,
            )
        )

        manifest = self._store.write(
            validated,
            created_at_utc=retrieved_at_utc,
        )

        return _PublishedNewsDataset(
            article_count=len(articles),
            validation_status=report.status,
            dataset_id=(
                manifest.dataset_fingerprint
            ),
        )

    @staticmethod
    def _validate_window(
        *,
        start_at_utc: datetime,
        end_at_utc: datetime,
    ) -> None:
        """Require one strictly increasing UTC ingestion window."""
        for name, value in (
            ("start_at_utc", start_at_utc),
            ("end_at_utc", end_at_utc),
        ):
            if not isinstance(
                value,
                datetime,
            ):
                raise TypeError(
                    f"{name} must be a datetime."
                )

            if (
                value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise ValueError(
                    f"{name} must be timezone-aware UTC."
                )

            if (
                value.utcoffset().total_seconds()
                != 0
            ):
                raise ValueError(
                    f"{name} must be UTC."
                )

        if start_at_utc >= end_at_utc:
            raise ValueError(
                "start_at_utc must be before "
                "end_at_utc."
            )

    def _utc_now(
        self,
    ) -> datetime:
        value = self._now()

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "now() must return a "
                "timezone-aware UTC datetime."
            )

        if (
            value.utcoffset().total_seconds()
            != 0
        ):
            raise ValueError(
                "now() must return UTC."
            )

        return value