from __future__ import annotations

from market_prediction.config.loader import (
    NewsConfig,
)
from market_prediction.news.query_builder import (
    build_news_queries,
)


def make_news_config() -> NewsConfig:
    """Return a small valid news configuration."""

    return NewsConfig.model_validate(
        {
            "providers": {
                "gdelt": {
                    "enabled": True,
                    "base_url": (
                        "https://data.gdeltproject.org/"
                        "gdeltv2"
                    ),
                    "request_delay_seconds": 1.0,
                },
                "marketaux": {
                    "enabled": True,
                    "base_url": (
                        "https://api.marketaux.com/v1"
                    ),
                    "api_key_env": (
                        "MARKETAUX_API_TOKEN"
                    ),
                    "request_delay_seconds": 1.0,
                    "timeout_seconds": 30.0,
                },
            },
            "queries": {
                "AAPL": [
                    {
                        "query_id": "aapl_company",
                        "strict_terms": [
                            "AAPL",
                            "Apple Inc",
                        ],
                        "context_terms": [],
                    }
                ],
                "MSFT": [
                    {
                        "query_id": "msft_company",
                        "strict_terms": [
                            "MSFT",
                            "Microsoft",
                        ],
                        "context_terms": [
                            "software",
                        ],
                    }
                ],
            },
            "availability": {
                "equity": {
                    "strategy": "session_cutoff",
                },
                "crypto": {
                    "strategy": "utc_cutoff",
                },
                "forex": {
                    "strategy": "forex_24_5_cutoff",
                },
                "commodity": {
                    "strategy": "session_cutoff",
                },
            },
            "validation": {
                "allowed_languages": ["en"],
                "require_title": True,
                "require_canonical_url": True,
                "max_future_seconds": 0,
            },
        }
    )


def test_build_news_queries_preserves_config_order() -> None:
    queries = build_news_queries(
        make_news_config()
    )

    assert tuple(
        query.asset
        for query in queries
    ) == (
        "AAPL",
        "MSFT",
    )

    assert tuple(
        query.query_id
        for query in queries
    ) == (
        "aapl_company",
        "msft_company",
    )


def test_build_news_queries_preserves_terms() -> None:
    queries = build_news_queries(
        make_news_config()
    )

    assert queries[0].strict_terms == (
        "AAPL",
        "Apple Inc",
    )

    assert queries[0].context_terms == ()

    assert queries[1].strict_terms == (
        "MSFT",
        "Microsoft",
    )

    assert queries[1].context_terms == (
        "software",
    )


def test_build_news_queries_returns_immutable_terms() -> None:
    queries = build_news_queries(
        make_news_config()
    )

    assert isinstance(
        queries[0].strict_terms,
        tuple,
    )

    assert isinstance(
        queries[0].context_terms,
        tuple,
    )