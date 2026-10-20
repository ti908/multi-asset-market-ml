"""Offline contracts for the market-data ingestion orchestrator."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd
import pytest

from market_prediction.cache import (
    CacheManager,
    CacheState,
    ColumnPriceBasis,
    HistoricalStability,
    PriceAdjustment,
    YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
)
from market_prediction.config import (
    AssetConfig,
    AssetsConfig,
    PipelineConfig,
    ProviderName,
)
from market_prediction.pipeline.orchestrator import (
    IngestionOrchestrator,
    ProviderRegistration,
)
from market_prediction.pipeline.results import (
    AssetOutcome,
    CacheAction,
    FailureCategory,
    RunStatus,
)
from market_prediction.providers import (
    MarketDataProvider,
    ProviderUnavailableError,
)
from market_prediction.validation import validate_ohlcv


NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)

YFINANCE_MODE = "auto_adjust=False"

ALTERNATE_PRICE_BASIS: Mapping[str, ColumnPriceBasis] = {
    "open": ColumnPriceBasis(
        adjustment=PriceAdjustment.SPLIT_AND_DIVIDEND_ADJUSTED,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
    "high": ColumnPriceBasis(
        adjustment=PriceAdjustment.SPLIT_AND_DIVIDEND_ADJUSTED,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
    "low": ColumnPriceBasis(
        adjustment=PriceAdjustment.SPLIT_AND_DIVIDEND_ADJUSTED,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
    "close": ColumnPriceBasis(
        adjustment=PriceAdjustment.SPLIT_AND_DIVIDEND_ADJUSTED,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
    "adj_close": ColumnPriceBasis(
        adjustment=PriceAdjustment.SPLIT_AND_DIVIDEND_ADJUSTED,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
    "volume": ColumnPriceBasis(
        adjustment=PriceAdjustment.SPLIT_CONSISTENT_VOLUME,
        historical_stability=HistoricalStability.MAY_BE_RESTATED,
    ),
}


class FakeProvider(MarketDataProvider):
    """Deterministic provider fake with observable calls."""

    def __init__(
        self,
        name: str,
        *,
        frames: dict[str, pd.DataFrame] | None = None,
        failures: dict[str, Exception] | None = None,
    ) -> None:
        self._name = name
        self.frames = frames or {}
        self.failures = failures or {}
        self.calls: list[tuple[str, date, date]] = []

    @property
    def name(self) -> str:
        return self._name

    def fetch_daily(
        self,
        asset: AssetConfig,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        self.calls.append(
            (asset.ticker, start_date, end_date)
        )

        failure = self.failures.get(asset.ticker)

        if failure is not None:
            raise failure

        frame = self.frames.get(asset.ticker)

        if frame is None:
            raise ProviderUnavailableError(
                f"{self.name} has no fake data for {asset.ticker}"
            )

        return frame.copy(deep=True)


def clean_frame(
    ticker: str,
    *,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Return a small deterministic normalized OHLCV frame."""

    del ticker

    timestamps = pd.to_datetime(
        [
            "2026-10-15",
            "2026-10-16",
            "2026-10-19",
            "2026-10-20",
        ],
        utc=True,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0],
            "adj_close": [101.0, 102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
            "source": [source] * 4,
        }
    )


def blocked_frame() -> pd.DataFrame:
    """Return structurally valid data that validation must block."""

    frame = clean_frame("AAPL")
    frame.loc[1, "close"] = -1.0
    return frame


@pytest.fixture
def assets_config() -> AssetsConfig:
    return AssetsConfig.model_validate(
        {
            "assets": {
                "AAPL": {
                    "display_name": "Apple",
                    "ticker": "AAPL",
                    "asset_class": "equity",
                    "currency": "USD",
                    "calendar": "XNYS",
                    "primary_provider": "yfinance",
                    "backup_provider": "stooq",
                },
                "MSFT": {
                    "display_name": "Microsoft",
                    "ticker": "MSFT",
                    "asset_class": "equity",
                    "currency": "USD",
                    "calendar": "XNYS",
                    "primary_provider": "yfinance",
                    "backup_provider": "stooq",
                },
            }
        }
    )


@pytest.fixture
def pipeline_config(
    tmp_path: Path,
) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "history": {
                "start_date": "2015-01-01",
                "interval": "1d",
            },
            "cache": {
                "ttl_hours": 24,
                "format": "parquet",
            },
            "retry": {
                "max_attempts": 3,
                "initial_delay_seconds": 2.0,
                "backoff_multiplier": 2.0,
            },
            "validation": {
                "asset_classes": {
                    "equity": {
                        "large_daily_move_threshold": 0.20,
                        "volume_spike_multiplier": 5.0,
                        "volume_rolling_window": 20,
                        "max_missing_sessions": 2,
                        "check_zero_volume": True,
                        "calendar": "XNYS",
                        "closed_dates": [],
                    },
                    "crypto": {
                        "large_daily_move_threshold": 0.40,
                        "volume_spike_multiplier": 8.0,
                        "volume_rolling_window": 20,
                        "max_missing_sessions": 1,
                        "check_zero_volume": True,
                        "calendar": "24/7",
                        "closed_dates": [],
                    },
                    "forex": {
                        "large_daily_move_threshold": 0.05,
                        "volume_spike_multiplier": 5.0,
                        "volume_rolling_window": 20,
                        "max_missing_sessions": 2,
                        "check_zero_volume": False,
                        "calendar": "24/5",
                        "closed_dates": [],
                    },
                    "commodity": {
                        "large_daily_move_threshold": 0.15,
                        "volume_spike_multiplier": 6.0,
                        "volume_rolling_window": 20,
                        "max_missing_sessions": 2,
                        "check_zero_volume": True,
                        "calendar": "COMEX",
                        "closed_dates": [],
                    },
                }
            },
            "paths": {
                "raw": str(tmp_path / "raw"),
                "cache": str(tmp_path / "cache"),
                "processed": str(tmp_path / "processed"),
                "synthetic": str(tmp_path / "synthetic"),
            },
        }
    )


@pytest.fixture
def cache_manager(
    pipeline_config: PipelineConfig,
) -> CacheManager:
    return CacheManager(
        cache_dir=pipeline_config.paths.cache,
        ttl_hours=pipeline_config.cache.ttl_hours,
        now=lambda: NOW,
    )


def registration(
    provider: MarketDataProvider,
    *,
    mode: str = YFINANCE_MODE,
    basis: Mapping[
        str,
        ColumnPriceBasis,
    ] = YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
) -> ProviderRegistration:
    return ProviderRegistration(
        provider=provider,
        price_adjustment_mode=mode,
        price_basis=basis,
    )


def make_orchestrator(
    *,
    assets_config: AssetsConfig,
    pipeline_config: PipelineConfig,
    cache_manager: CacheManager,
    primary: FakeProvider,
    backup: FakeProvider | None = None,
    backup_basis: Mapping[
        str,
        ColumnPriceBasis,
    ] = YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
) -> IngestionOrchestrator:
    providers = {
        ProviderName.YFINANCE: registration(primary),
    }

    if backup is not None:
        providers[ProviderName.STOOQ] = registration(
            backup,
            mode=YFINANCE_MODE,
            basis=backup_basis,
        )

    return IngestionOrchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        providers=providers,
        validator=validate_ohlcv,
        cache_manager=cache_manager,
        now=lambda: NOW,
    )


def result_by_asset(run, asset_id: str):
    return next(
        result
        for result in run.assets
        if result.asset_id == asset_id
    )


def seed_cache(
    *,
    cache_manager: CacheManager,
    asset: AssetConfig,
    pipeline_config: PipelineConfig,
    frame: pd.DataFrame,
    fetch_timestamp: datetime,
    provider: str = "yfinance",
    mode: str = YFINANCE_MODE,
    basis: Mapping[
        str,
        ColumnPriceBasis,
    ] = YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
) -> None:
    from market_prediction.cache import ValidatedDataset

    report = validate_ohlcv(
        frame,
        asset,
        pipeline_config,
    )

    dataset = ValidatedDataset.from_validation(
        frame,
        report,
    )

    cache_manager.write(
        dataset,
        provider=provider,
        fetch_timestamp=fetch_timestamp,
        price_adjustment_mode=mode,
        price_basis=basis,
    )


def test_all_assets_succeed(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        frames={
            "AAPL": clean_frame("AAPL"),
            "MSFT": clean_frame("MSFT"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run()

    assert run.status is RunStatus.SUCCESS
    assert run.exit_code == 0
    assert len(run.assets) == 2

    for result in run.assets:
        assert result.outcome is AssetOutcome.SUCCESS
        assert result.provider_used == "yfinance"
        assert result.fallback_used is False
        assert result.rows_persisted == 4
        assert result.cache_action is CacheAction.WRITTEN
        assert result.data_freshness is CacheState.FRESH

    assert len(primary.calls) == 2


def test_validation_rejection_does_not_stop_other_assets(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        frames={
            "AAPL": blocked_frame(),
            "MSFT": clean_frame("MSFT"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run()

    aapl = result_by_asset(run, "AAPL")
    msft = result_by_asset(run, "MSFT")

    assert aapl.outcome is AssetOutcome.REJECTED_BY_VALIDATION
    assert aapl.failure_category is FailureCategory.VALIDATION
    assert aapl.cache_action is CacheAction.NOT_WRITTEN
    assert cache_manager.read("AAPL").state is CacheState.MISSING

    assert msft.outcome is AssetOutcome.SUCCESS
    assert cache_manager.read("MSFT").state is CacheState.FRESH

    assert run.status is RunStatus.PARTIAL_FAILURE
    assert run.exit_code == 1


def test_validation_block_does_not_trigger_fallback(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        frames={
            "AAPL": blocked_frame(),
            "MSFT": clean_frame("MSFT"),
        },
    )
    backup = FakeProvider(
        "stooq",
        frames={
            "AAPL": clean_frame("AAPL", source="stooq"),
            "MSFT": clean_frame("MSFT", source="stooq"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
        backup=backup,
    )

    run = orchestrator.run(asset_ids=["AAPL"])

    result = run.assets[0]

    assert result.outcome is AssetOutcome.REJECTED_BY_VALIDATION
    assert backup.calls == []


def test_primary_failure_uses_backup(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        failures={
            "AAPL": ProviderUnavailableError("Yahoo unavailable"),
        },
    )
    backup = FakeProvider(
        "stooq",
        frames={
            "AAPL": clean_frame("AAPL", source="stooq"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
        backup=backup,
    )

    run = orchestrator.run(asset_ids=["AAPL"])

    result = run.assets[0]

    assert result.outcome is AssetOutcome.SUCCESS_VIA_FALLBACK
    assert result.provider_used == "stooq"
    assert result.fallback_used is True
    assert result.rows_persisted == 4
    assert result.cache_action is CacheAction.WRITTEN
    assert run.status is RunStatus.SUCCESS

    cached = cache_manager.read("AAPL")
    assert cached.state is CacheState.FRESH
    assert cached.metadata is not None
    assert cached.metadata.provider == "stooq"


def test_both_providers_fail_and_stale_cache_is_served(
    assets_config,
    pipeline_config,
    tmp_path,
):
    stale_cache = CacheManager(
        cache_dir=tmp_path / "stale-cache",
        ttl_hours=24,
        now=lambda: NOW,
    )
    asset = assets_config.assets["AAPL"]

    seed_cache(
        cache_manager=stale_cache,
        asset=asset,
        pipeline_config=pipeline_config,
        frame=clean_frame("AAPL"),
        fetch_timestamp=NOW - timedelta(hours=25),
    )

    assert stale_cache.read("AAPL").state is CacheState.STALE

    primary = FakeProvider(
        "yfinance",
        failures={
            "AAPL": ProviderUnavailableError("primary down"),
        },
    )
    backup = FakeProvider(
        "stooq",
        failures={
            "AAPL": ProviderUnavailableError("backup down"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=stale_cache,
        primary=primary,
        backup=backup,
    )

    run = orchestrator.run(asset_ids=["AAPL"])

    result = run.assets[0]

    assert result.outcome is AssetOutcome.SERVED_STALE
    assert result.is_ok is False
    assert result.data_freshness is CacheState.STALE
    assert result.cache_action is CacheAction.NOT_WRITTEN
    assert result.failure_category is FailureCategory.PROVIDER
    assert run.status is RunStatus.TOTAL_FAILURE
    assert run.exit_code == 2


def test_both_providers_fail_without_cache_is_clean_failure(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        failures={
            "AAPL": ProviderUnavailableError("primary down"),
        },
    )
    backup = FakeProvider(
        "stooq",
        failures={
            "AAPL": ProviderUnavailableError("backup down"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
        backup=backup,
    )

    run = orchestrator.run(asset_ids=["AAPL"])

    result = run.assets[0]

    assert result.outcome is AssetOutcome.FAILED_PROVIDER
    assert result.failure_category is FailureCategory.PROVIDER
    assert result.data_freshness is CacheState.MISSING
    assert run.status is RunStatus.TOTAL_FAILURE
    assert run.exit_code == 2


def test_fallback_price_basis_mismatch_preserves_existing_cache(
    assets_config,
    pipeline_config,
    tmp_path,
):
    cache = CacheManager(
        cache_dir=tmp_path / "basis-cache",
        ttl_hours=24,
        now=lambda: NOW,
    )
    asset = assets_config.assets["AAPL"]

    old_frame = clean_frame("AAPL")

    seed_cache(
        cache_manager=cache,
        asset=asset,
        pipeline_config=pipeline_config,
        frame=old_frame,
        fetch_timestamp=NOW - timedelta(hours=25),
    )

    before = cache.read("AAPL")

    primary = FakeProvider(
        "yfinance",
        failures={
            "AAPL": ProviderUnavailableError("primary down"),
        },
    )
    backup = FakeProvider(
        "stooq",
        frames={
            "AAPL": clean_frame("AAPL", source="stooq"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache,
        primary=primary,
        backup=backup,
        backup_basis=ALTERNATE_PRICE_BASIS,
    )

    run = orchestrator.run(
        asset_ids=["AAPL"],
        force_refresh=True,
    )

    result = run.assets[0]

    assert result.outcome is AssetOutcome.FAILED_CACHE
    assert result.failure_category is FailureCategory.PRICE_BASIS
    assert result.cache_action is CacheAction.NOT_WRITTEN

    after = cache.read("AAPL")

    assert before.metadata is not None
    assert after.metadata is not None
    assert after.metadata.dataset_id == before.metadata.dataset_id
    assert after.metadata.provider == before.metadata.provider


def test_fresh_cache_skips_provider(
    assets_config,
    pipeline_config,
    cache_manager,
):
    asset = assets_config.assets["AAPL"]

    seed_cache(
        cache_manager=cache_manager,
        asset=asset,
        pipeline_config=pipeline_config,
        frame=clean_frame("AAPL"),
        fetch_timestamp=NOW - timedelta(hours=1),
    )

    primary = FakeProvider(
        "yfinance",
        failures={
            "AAPL": AssertionError(
                "provider must not be called for fresh cache"
            )
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run(asset_ids=["AAPL"])

    result = run.assets[0]

    assert result.outcome is AssetOutcome.SKIPPED_FRESH
    assert result.cache_action is CacheAction.SKIPPED_FRESH
    assert result.data_freshness is CacheState.FRESH
    assert primary.calls == []
    assert run.status is RunStatus.SUCCESS


def test_force_refresh_bypasses_fresh_cache(
    assets_config,
    pipeline_config,
    cache_manager,
):
    asset = assets_config.assets["AAPL"]

    seed_cache(
        cache_manager=cache_manager,
        asset=asset,
        pipeline_config=pipeline_config,
        frame=clean_frame("AAPL"),
        fetch_timestamp=NOW - timedelta(hours=1),
    )

    refreshed = clean_frame("AAPL")
    refreshed["volume"] = refreshed["volume"] + 1

    primary = FakeProvider(
        "yfinance",
        frames={
            "AAPL": refreshed,
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run(
        asset_ids=["AAPL"],
        force_refresh=True,
    )

    result = run.assets[0]

    assert result.outcome is AssetOutcome.SUCCESS
    assert result.cache_action is CacheAction.WRITTEN
    assert len(primary.calls) == 1


def test_unexpected_exception_is_isolated(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        frames={
            "MSFT": clean_frame("MSFT"),
        },
        failures={
            "AAPL": RuntimeError("unexpected programming failure"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run()

    aapl = result_by_asset(run, "AAPL")
    msft = result_by_asset(run, "MSFT")

    assert aapl.outcome is AssetOutcome.FAILED_UNEXPECTED
    assert aapl.failure_category is FailureCategory.UNEXPECTED
    assert "unexpected programming failure" in (
        aapl.failure_reason or ""
    )

    assert msft.outcome is AssetOutcome.SUCCESS
    assert run.status is RunStatus.PARTIAL_FAILURE
    assert run.exit_code == 1


def test_missing_backup_registration_is_provider_failure(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        failures={
            "AAPL": ProviderUnavailableError("primary down"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run(asset_ids=["AAPL"])

    result = run.assets[0]

    assert result.outcome is AssetOutcome.FAILED_PROVIDER
    assert result.failure_category is FailureCategory.PROVIDER
    assert "stooq" in (result.failure_reason or "")


def test_multi_asset_partial_failure_has_exit_code_one(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        frames={
            "MSFT": clean_frame("MSFT"),
        },
        failures={
            "AAPL": ProviderUnavailableError("primary down"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run()

    assert run.status is RunStatus.PARTIAL_FAILURE
    assert run.exit_code == 1


def test_total_failure_has_exit_code_two(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        failures={
            "AAPL": ProviderUnavailableError("AAPL unavailable"),
            "MSFT": ProviderUnavailableError("MSFT unavailable"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run()

    assert run.status is RunStatus.TOTAL_FAILURE
    assert run.exit_code == 2


def test_second_run_skips_newly_fresh_cache(
    assets_config,
    pipeline_config,
    cache_manager,
):
    primary = FakeProvider(
        "yfinance",
        frames={
            "AAPL": clean_frame("AAPL"),
            "MSFT": clean_frame("MSFT"),
        },
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    first = orchestrator.run()
    calls_after_first = len(primary.calls)

    second = orchestrator.run()

    assert first.status is RunStatus.SUCCESS
    assert second.status is RunStatus.SUCCESS

    assert all(
        result.outcome is AssetOutcome.SKIPPED_FRESH
        for result in second.assets
    )

    assert len(primary.calls) == calls_after_first

def test_expected_cache_write_failure_is_classified_as_cache_failure(
    assets_config,
    pipeline_config,
    cache_manager,
    monkeypatch,
):
    primary = FakeProvider(
        "yfinance",
        frames={
            "AAPL": clean_frame("AAPL"),
        },
    )

    def fail_write(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        cache_manager,
        "write",
        fail_write,
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run(asset_ids=["AAPL"])

    result = run.assets[0]

    assert result.outcome is AssetOutcome.FAILED_CACHE
    assert result.failure_category is FailureCategory.CACHE
    assert "disk unavailable" in (result.failure_reason or "")
    assert run.status is RunStatus.TOTAL_FAILURE
    assert run.exit_code == 2


def test_unexpected_cache_exception_is_classified_as_unexpected(
    assets_config,
    pipeline_config,
    cache_manager,
    monkeypatch,
):
    primary = FakeProvider(
        "yfinance",
        frames={
            "AAPL": clean_frame("AAPL"),
        },
    )

    def fail_write(*args, **kwargs):
        raise RuntimeError("cache implementation bug")

    monkeypatch.setattr(
        cache_manager,
        "write",
        fail_write,
    )

    orchestrator = make_orchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        cache_manager=cache_manager,
        primary=primary,
    )

    run = orchestrator.run(asset_ids=["AAPL"])

    result = run.assets[0]

    assert result.outcome is AssetOutcome.FAILED_UNEXPECTED
    assert result.failure_category is FailureCategory.UNEXPECTED
    assert "cache implementation bug" in (
        result.failure_reason or ""
    )
    assert run.status is RunStatus.TOTAL_FAILURE
    assert run.exit_code == 2

def test_missing_primary_registration_can_use_registered_backup(
    assets_config,
    pipeline_config,
    cache_manager,
):
    backup = FakeProvider(
        "stooq",
        frames={
            "AAPL": clean_frame(
                "AAPL",
                source="stooq",
            ),
        },
    )

    orchestrator = IngestionOrchestrator(
        assets_config=assets_config,
        pipeline_config=pipeline_config,
        providers={
            ProviderName.STOOQ: registration(backup),
        },
        validator=validate_ohlcv,
        cache_manager=cache_manager,
        now=lambda: NOW,
    )

    run = orchestrator.run(asset_ids=["AAPL"])

    result = run.assets[0]

    assert result.outcome is AssetOutcome.SUCCESS_VIA_FALLBACK
    assert result.provider_used == "stooq"
    assert result.fallback_used is True
    assert result.cache_action is CacheAction.WRITTEN
    assert len(backup.calls) == 1
    assert run.status is RunStatus.SUCCESS