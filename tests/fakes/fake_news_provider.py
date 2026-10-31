"""Deterministic fake implementation of the news-provider contract."""

from __future__ import annotations

from datetime import datetime

from market_prediction.news.models import ArticleRecord
from market_prediction.providers.news.base import (
    NewsProvider,
    NewsQuery,
)


class FakeNewsProvider(NewsProvider):
    """Return predetermined canonical articles."""

    def __init__(
        self,
        articles: list[ArticleRecord],
    ) -> None:
        self.articles = articles
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

        return list(self.articles)