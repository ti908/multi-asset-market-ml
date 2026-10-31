"""Provider-neutral contract for market-news sources.

Concrete providers normalize their output into canonical ArticleRecord
objects before returning data to downstream validation or orchestration.

This module contains no provider-specific HTTP, storage, sentiment
scoring, deduplication, or market-data alignment logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from market_prediction.news.models import ArticleRecord


def _validate_non_empty(
    value: str,
    *,
    field_name: str,
) -> None:
    """Require a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )


def _validate_terms(
    terms: tuple[str, ...],
    *,
    field_name: str,
    require_non_empty: bool,
) -> None:
    """Validate immutable query terms."""

    if require_non_empty and not terms:
        raise ValueError(
            f"{field_name} must contain at least one term"
        )

    if any(
        not isinstance(term, str)
        or not term.strip()
        for term in terms
    ):
        raise ValueError(
            f"{field_name} must contain "
            "non-empty strings"
        )

    if len(terms) != len(set(terms)):
        raise ValueError(
            f"{field_name} must contain unique terms"
        )

def validate_news_query_window(
    *,
    start_at_utc: datetime,
    end_at_utc: datetime,
) -> None:
    """Validate a half-open UTC news query window.

    News providers use the interval:

        [start_at_utc, end_at_utc)

    Both boundaries must be timezone-aware UTC datetimes and the start
    boundary must be strictly earlier than the end boundary.
    """

    for field_name, value in (
        ("start_at_utc", start_at_utc),
        ("end_at_utc", end_at_utc),
    ):
        if not isinstance(value, datetime):
            raise ValueError(
                f"{field_name} must be a datetime"
            )

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

    if start_at_utc >= end_at_utc:
        raise ValueError(
            "start_at_utc must be earlier than end_at_utc"
        )

@dataclass(frozen=True)
class NewsQuery:
    """Provider-neutral query definition for one project asset.

    Query terms describe the canonical project intent. Concrete
    providers remain responsible for translating this definition into
    their own supported query syntax.
    """

    asset: str
    query_id: str
    strict_terms: tuple[str, ...]
    context_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate immutable query fields."""

        _validate_non_empty(
            self.asset,
            field_name="asset",
        )

        _validate_non_empty(
            self.query_id,
            field_name="query_id",
        )

        _validate_terms(
            self.strict_terms,
            field_name="strict_terms",
            require_non_empty=True,
        )

        _validate_terms(
            self.context_terms,
            field_name="context_terms",
            require_non_empty=False,
        )


class NewsProvider(ABC):
    """Abstract interface implemented by market-news providers."""

    @abstractmethod
    def fetch_articles(
        self,
        *,
        query: NewsQuery,
        start_at_utc: datetime,
        end_at_utc: datetime,
    ) -> list[ArticleRecord]:
        """Fetch and normalize articles for one query window.

        Implementations must either return canonical ArticleRecord
        objects or raise an exception from the news-provider exception
        hierarchy.

        Returning an empty list is valid when the upstream provider
        truthfully returns no matching records.
        """

        raise NotImplementedError