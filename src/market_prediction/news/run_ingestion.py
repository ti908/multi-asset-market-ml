"""Production construction and CLI for canonical news ingestion.

This module wires validated project configuration to the canonical
Marketaux news provider, NewsStore, and NewsIngestionOrchestrator.

GDELT discovery candidates remain outside this canonical ingestion path
until they can be enriched into ArticleRecord objects.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from market_prediction.config.loader import (
    ConfigLoadError,
    load_project_config,
)
from market_prediction.news.orchestrator import (
    NewsIngestionOrchestrator,
)
from market_prediction.news.query_builder import (
    build_news_queries,
)
from market_prediction.news.storage import (
    NewsStore,
)
from market_prediction.providers.news.base import (
    NewsQuery,
)
from market_prediction.providers.news.marketaux import (
    MarketauxNewsProvider,
)
from market_prediction.providers.news.runtime import (
    SystemUtcClock,
    UrllibJsonTransport,
)
from market_prediction.providers.news.exceptions import (
    NewsProviderError,
)


def _parse_utc_datetime(
    value: str,
) -> datetime:
    """Parse one explicitly UTC ISO-8601 timestamp."""

    if not isinstance(value, str):
        raise ValueError(
            "timestamp must be a string."
        )

    candidate = value.strip()

    if not candidate:
        raise ValueError(
            "timestamp must not be empty."
        )

    try:
        parsed = datetime.fromisoformat(
            candidate.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid UTC timestamp: {value}"
        ) from exc

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise ValueError(
            "timestamp must be timezone-aware UTC."
        )

    if (
        parsed.utcoffset().total_seconds()
        != 0
    ):
        raise ValueError(
            "timestamp must be expressed in UTC."
        )

    return parsed.astimezone(
        timezone.utc
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the canonical-news ingestion CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Fetch, validate, and atomically "
            "store canonical news data."
        ),
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help=(
            "Directory containing assets.yaml "
            "and pipeline.yaml (default: config)."
        ),
    )

    parser.add_argument(
        "--start",
        required=True,
        help=(
            "Inclusive UTC start timestamp "
            "in ISO-8601 format."
        ),
    )

    parser.add_argument(
        "--end",
        required=True,
        help=(
            "Exclusive UTC end timestamp "
            "in ISO-8601 format."
        ),
    )

    return parser


def build_news_ingestion(
    config_dir: Path,
) -> tuple[
    NewsIngestionOrchestrator,
    tuple[NewsQuery, ...],
]:
    """Construct production canonical-news ingestion dependencies."""

    config_dir = Path(config_dir)

    assets_config, pipeline_config = (
        load_project_config(
            config_dir / "assets.yaml",
            config_dir / "pipeline.yaml",
        )
    )

    news_config = pipeline_config.news

    if news_config is None:
        raise ValueError(
            "News configuration is required."
        )

    marketaux_config = (
        news_config.providers.marketaux
    )

    if not marketaux_config.enabled:
        raise ValueError(
            "Marketaux must be enabled for "
            "canonical news ingestion."
        )

    api_token = os.getenv(
        marketaux_config.api_key_env
    )

    if api_token is None or not api_token.strip():
        raise ValueError(
            "Required Marketaux API token "
            f"environment variable "
            f"'{marketaux_config.api_key_env}' "
            "is not configured."
        )

    queries = build_news_queries(
        news_config
    )

    provider = MarketauxNewsProvider(
        base_url=marketaux_config.base_url,
        api_token=api_token,
        timeout_seconds=(
            marketaux_config.timeout_seconds
        ),
        transport=UrllibJsonTransport(),
        clock=SystemUtcClock(),
    )

    store = NewsStore(
        storage_dir=(
            Path(pipeline_config.paths.raw)
            / "news"
            / "canonical"
        )
    )

    allowed_assets = set(
        assets_config.assets
    )

    allowed_query_ids = {
        query.query_id
        for query in queries
    }

    allowed_languages = set(
        news_config.validation.allowed_languages
    )

    orchestrator = NewsIngestionOrchestrator(
        provider=provider,
        store=store,
        allowed_assets=allowed_assets,
        allowed_query_ids=allowed_query_ids,
        allowed_languages=allowed_languages,
    )

    return orchestrator, queries


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run canonical news ingestion."""

    parser = build_parser()

    try:
        args = parser.parse_args(
            argv
        )

        start_at_utc = _parse_utc_datetime(
            args.start
        )

        end_at_utc = _parse_utc_datetime(
            args.end
        )

        orchestrator, queries = (
            build_news_ingestion(
                args.config_dir
            )
        )

        result = orchestrator.run_queries(
            queries=list(queries),
            start_at_utc=start_at_utc,
            end_at_utc=end_at_utc,
        )


    except (

            ConfigLoadError,

            NewsProviderError,

            ValueError,

    ) as exc:
        print(
            str(exc),
            file=sys.stderr,
        )
        return 2

    print(
        "Canonical news ingestion complete"
    )
    print(
        f"Articles: {result.article_count}"
    )
    print(
        "Validation: "
        f"{result.validation_status.value}"
    )
    print(
        f"Dataset ID: {result.dataset_id}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
