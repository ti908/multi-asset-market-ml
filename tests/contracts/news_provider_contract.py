"""Reusable contract assertions for market-news providers."""

from __future__ import annotations

from datetime import datetime

from market_prediction.news.models import ArticleRecord
from market_prediction.providers.news.base import (
    NewsProvider,
    NewsQuery,
)


def assert_news_provider_contract(
    provider: NewsProvider,
    *,
    query: NewsQuery,
    start_at_utc: datetime,
    end_at_utc: datetime,
) -> list[ArticleRecord]:
    """Assert the common output contract for a news provider."""

    articles = provider.fetch_articles(
        query=query,
        start_at_utc=start_at_utc,
        end_at_utc=end_at_utc,
    )

    assert isinstance(articles, list)

    for article in articles:
        assert isinstance(
            article,
            ArticleRecord,
        )

    return articles