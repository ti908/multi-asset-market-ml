"""Construction of provider-neutral news queries from configuration."""

from __future__ import annotations

from market_prediction.config.loader import NewsConfig
from market_prediction.providers.news.base import NewsQuery


def build_news_queries(
    config: NewsConfig,
) -> tuple[NewsQuery, ...]:
    """Build news queries while preserving configuration order."""

    queries: list[NewsQuery] = []

    for asset, configured_queries in config.queries.items():
        for configured_query in configured_queries:
            queries.append(
                NewsQuery(
                    asset=asset,
                    query_id=configured_query.query_id,
                    strict_terms=tuple(
                        configured_query.strict_terms
                    ),
                    context_terms=tuple(
                        configured_query.context_terms
                    ),
                )
            )

    return tuple(queries)