"""Command-line entry point for market-data ingestion."""

from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path
from typing import Sequence


from market_prediction.cache import (
    CacheManager,
    ColumnPriceBasis,
    HistoricalStability,
    PriceAdjustment,
    YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
)
from market_prediction.config import (
    ConfigLoadError,
    ProviderName,
    load_project_config,
)
from market_prediction.pipeline.orchestrator import (
    IngestionOrchestrator,
    ProviderRegistration,
)
from market_prediction.pipeline.results import (
    AssetIngestionResult,
    IngestionRunResult,
)
from market_prediction.providers.alpha_vantage_provider import (
    AlphaVantageProvider,
)
from market_prediction.providers.twelve_data_provider import (
    TwelveDataProvider,
)
from market_prediction.providers.yfinance_provider import (
    YFinanceProvider,
)
from market_prediction.validation import validate_ohlcv

SPOT_FX_PRICE_BASIS = {
    "open": ColumnPriceBasis(
        adjustment=PriceAdjustment.NOT_APPLICABLE,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
    "high": ColumnPriceBasis(
        adjustment=PriceAdjustment.NOT_APPLICABLE,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
    "low": ColumnPriceBasis(
        adjustment=PriceAdjustment.NOT_APPLICABLE,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
    "close": ColumnPriceBasis(
        adjustment=PriceAdjustment.NOT_APPLICABLE,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
    "adj_close": ColumnPriceBasis(
        adjustment=PriceAdjustment.NOT_APPLICABLE,
        historical_stability=HistoricalStability.NOT_APPLICABLE,
    ),
    "volume": ColumnPriceBasis(
        adjustment=PriceAdjustment.NOT_APPLICABLE,
        historical_stability=HistoricalStability.NOT_APPLICABLE,
    ),
}

def build_parser() -> argparse.ArgumentParser:
    """Build the ingestion command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch, validate, and cache configured daily market data."
        ),
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help=(
            "Directory containing assets.yaml and pipeline.yaml "
            "(default: config)."
        ),
    )

    parser.add_argument(
        "--asset",
        action="append",
        default=None,
        help=(
            "Configured asset id to ingest. Repeat this option to "
            "select multiple assets. By default all assets are run."
        ),
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Fetch again even when a fresh cache entry exists.",
    )

    return parser


def build_orchestrator(
    config_dir: Path,
    asset_ids: Sequence[str] | None = None,
) -> IngestionOrchestrator:
    """Build the production ingestion orchestrator from project config."""
    assets_config, pipeline_config = load_project_config(
        config_dir / "assets.yaml",
        config_dir / "pipeline.yaml",
    )

    if asset_ids is None:
        selected_assets = list(
            assets_config.assets.values()
        )
    else:
        unknown_asset_ids = [
            asset_id
            for asset_id in asset_ids
            if asset_id not in assets_config.assets
        ]

        if unknown_asset_ids:
            unknown = ", ".join(unknown_asset_ids)
            raise ValueError(
                f"Unknown configured asset id(s): {unknown}"
            )

        selected_assets = [
            assets_config.assets[asset_id]
            for asset_id in asset_ids
        ]

    requires_twelve_data = any(
        asset.primary_provider is ProviderName.TWELVE_DATA
        or asset.backup_provider is ProviderName.TWELVE_DATA
        for asset in selected_assets
    )

    requires_alpha_vantage = any(
        asset.primary_provider is ProviderName.ALPHA_VANTAGE
        or asset.backup_provider is ProviderName.ALPHA_VANTAGE
        for asset in selected_assets
    )

    yfinance_provider = YFinanceProvider(
        max_attempts=pipeline_config.retry.max_attempts,
        initial_delay_seconds=(
            pipeline_config.retry.initial_delay_seconds
        ),
        backoff_multiplier=(
            pipeline_config.retry.backoff_multiplier
        ),
    )

    providers = {
        ProviderName.YFINANCE: ProviderRegistration(
            provider=yfinance_provider,
            price_adjustment_mode="auto_adjust=False",
            price_basis=YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
            bar_convention=(
                "provider_native_daily_bar;"
                "timestamp_normalized_to_utc"
            ),
        ),
    }

    if requires_twelve_data:
        twelve_data_config = (
            pipeline_config.providers.twelve_data
        )

        api_key = os.environ.get(
            twelve_data_config.api_key_env
        )

        if api_key is None or not api_key.strip():
            raise ValueError(
                "Required environment variable "
                f"'{twelve_data_config.api_key_env}' "
                "is not set for Twelve Data."
            )

        twelve_data_provider = TwelveDataProvider(
            api_key=api_key,
            base_url=twelve_data_config.base_url,
            max_attempts=(
                pipeline_config.retry.max_attempts
            ),
            initial_delay_seconds=(
                pipeline_config.retry.initial_delay_seconds
            ),
            backoff_multiplier=(
                pipeline_config.retry.backoff_multiplier
            ),
            timeout_seconds=(
                twelve_data_config.timeout_seconds
            ),
        )

        providers[
            ProviderName.TWELVE_DATA
        ] = ProviderRegistration(
            provider=twelve_data_provider,
            price_adjustment_mode=(
                "not_applicable_spot_fx"
            ),
            price_basis=SPOT_FX_PRICE_BASIS,
            bar_convention=(
                "twelve_data_1day_provider_local_date_label;"
                "forex_default_timezone_australia_sydney;"
                "date_label_normalized_to_utc_midnight"
            ),
        )

    if requires_alpha_vantage:
        alpha_vantage_config = (
            pipeline_config.providers.alpha_vantage
        )

        api_key = os.environ.get(
            alpha_vantage_config.api_key_env
        )

        if api_key is None or not api_key.strip():
            raise ValueError(
                "Required environment variable "
                f"'{alpha_vantage_config.api_key_env}' "
                "is not set for Alpha Vantage."
            )

        alpha_vantage_provider = AlphaVantageProvider(
            api_key=api_key,
            base_url=alpha_vantage_config.base_url,
            max_attempts=(
                pipeline_config.retry.max_attempts
            ),
            initial_delay_seconds=(
                pipeline_config.retry.initial_delay_seconds
            ),
            backoff_multiplier=(
                pipeline_config.retry.backoff_multiplier
            ),
            timeout_seconds=(
                alpha_vantage_config.timeout_seconds
            ),
        )

        providers[
            ProviderName.ALPHA_VANTAGE
        ] = ProviderRegistration(
            provider=alpha_vantage_provider,
            price_adjustment_mode=(
                "not_applicable_spot_fx"
            ),
            price_basis=SPOT_FX_PRICE_BASIS,
            bar_convention=(
                "alpha_vantage_fx_daily_utc_date_label;"
                "provider_timezone_utc;"
                "date_label_normalized_to_utc_midnight"
            ),
        )


    cache_manager = CacheManager(
        cache_dir=Path(pipeline_config.paths.cache),
        ttl_hours=pipeline_config.cache.ttl_hours,
    )

    return IngestionOrchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        providers=providers,
        validator=validate_ohlcv,
        cache_manager=cache_manager,
    )


def _display(value: object | None) -> str:
    """Return a compact printable representation."""
    if value is None:
        return "-"

    raw_value = getattr(value, "value", value)
    return str(raw_value)


def _row(result: AssetIngestionResult) -> tuple[str, ...]:
    """Convert one asset result into printable table cells."""
    return (
        result.asset_id,
        result.ticker,
        result.outcome.value,
        _display(result.provider_used),
        "yes" if result.fallback_used else "no",
        str(result.rows_persisted),
        _display(result.validation_status),
        str(result.soft_anomaly_count),
        result.cache_action.value,
        result.data_freshness.value,
        _display(result.failure_category),
        result.failure_reason or "-",
    )


def _render_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> str:
    """Render a dependency-free aligned text table."""
    widths = [
        len(header)
        for header in headers
    ]

    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(
                widths[index],
                len(cell),
            )

    def render_row(row: tuple[str, ...]) -> str:
        return " | ".join(
            cell.ljust(widths[index])
            for index, cell in enumerate(row)
        )

    separator = "-+-".join(
        "-" * width
        for width in widths
    )

    lines = [
        render_row(headers),
        separator,
    ]

    lines.extend(
        render_row(row)
        for row in rows
    )

    return "\n".join(lines)


def render_run_result(
    run: IngestionRunResult,
) -> str:
    """Render a glanceable ingestion report."""
    headers = (
        "Asset",
        "Ticker",
        "Outcome",
        "Provider",
        "Fallback",
        "Rows",
        "Validation",
        "Anomalies",
        "Cache",
        "Freshness",
        "Failure",
        "Reason",
    )

    rows = [
        _row(result)
        for result in run.assets
    ]

    table = _render_table(
        headers,
        rows,
    )

    overall = (
        f"Overall: {run.status.value} "
        f"| exit_code={run.exit_code}"
    )

    return f"{table}\n\n{overall}"


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run market-data ingestion and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        orchestrator = build_orchestrator(
            args.config_dir,
            asset_ids=args.asset,
        )

        run = orchestrator.run(
            asset_ids=args.asset,
            force_refresh=args.force_refresh,
        )
    except ConfigLoadError as exc:
        print(
            f"Configuration error: {exc}",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(
            f"Input error: {exc}",
            file=sys.stderr,
        )
        return 2

    print(render_run_result(run))

    return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())