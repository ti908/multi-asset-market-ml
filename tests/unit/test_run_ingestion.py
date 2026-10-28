"""Offline contracts for the ingestion CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from market_prediction.cache import (
    YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
)
from market_prediction.pipeline.results import (
    AssetIngestionResult,
    AssetOutcome,
    CacheAction,
    IngestionRunResult,
    RunStatus,
)
from market_prediction.cache import CacheState
from market_prediction.validation import Status


NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def asset_result(
    asset_id: str,
    *,
    outcome: AssetOutcome = AssetOutcome.SUCCESS,
) -> AssetIngestionResult:
    """Build a deterministic CLI-facing asset result."""

    is_success = outcome in {
        AssetOutcome.SUCCESS,
        AssetOutcome.SUCCESS_VIA_FALLBACK,
        AssetOutcome.SKIPPED_FRESH,
    }

    return AssetIngestionResult(
        asset_id=asset_id,
        ticker=asset_id,
        outcome=outcome,
        provider_used="yfinance",
        fallback_used=False,
        rows_persisted=4 if is_success else 0,
        validation_status=Status.PASSED if is_success else None,
        soft_anomaly_count=0,
        cache_action=(
            CacheAction.WRITTEN
            if outcome is AssetOutcome.SUCCESS
            else CacheAction.NOT_WRITTEN
        ),
        data_freshness=(
            CacheState.FRESH
            if is_success
            else CacheState.MISSING
        ),
    )


def run_result(
    *,
    assets: tuple[AssetIngestionResult, ...],
    status: RunStatus,
) -> IngestionRunResult:
    return IngestionRunResult(
        assets=assets,
        started_at=NOW,
        ended_at=NOW,
        status=status,
    )


def test_parser_defaults_to_config_directory():
    from market_prediction.pipeline.run_ingestion import build_parser

    args = build_parser().parse_args([])

    assert args.config_dir == Path("config")
    assert args.asset is None
    assert args.force_refresh is False


def test_parser_accepts_repeatable_asset_filter():
    from market_prediction.pipeline.run_ingestion import build_parser

    args = build_parser().parse_args(
        [
            "--asset",
            "AAPL",
            "--asset",
            "GOLD",
        ]
    )

    assert args.asset == ["AAPL", "GOLD"]


def test_parser_accepts_force_refresh():
    from market_prediction.pipeline.run_ingestion import build_parser

    args = build_parser().parse_args(
        ["--force-refresh"]
    )

    assert args.force_refresh is True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RunStatus.SUCCESS, 0),
        (RunStatus.PARTIAL_FAILURE, 1),
        (RunStatus.TOTAL_FAILURE, 2),
    ],
)
def test_main_returns_run_exit_code(
    monkeypatch,
    capsys,
    tmp_path,
    status,
    expected,
):
    import market_prediction.pipeline.run_ingestion as cli

    fake_run = run_result(
        assets=(
            asset_result(
                "AAPL",
                outcome=(
                    AssetOutcome.SUCCESS
                    if status is RunStatus.SUCCESS
                    else AssetOutcome.FAILED_PROVIDER
                ),
            ),
        ),
        status=status,
    )

    class FakeOrchestrator:
        def run(self, **kwargs):
            return fake_run

    monkeypatch.setattr(
        cli,
        "build_orchestrator",
        lambda config_dir, asset_ids=None: FakeOrchestrator(),
    )

    exit_code = cli.main(
        [
            "--config-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == expected
    assert status.value in output


def test_main_passes_asset_filter_and_force_refresh(
    monkeypatch,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    observed = {}

    fake_run = run_result(
        assets=(asset_result("AAPL"),),
        status=RunStatus.SUCCESS,
    )

    class FakeOrchestrator:
        def run(
            self,
            *,
            asset_ids=None,
            force_refresh=False,
        ):
            observed["asset_ids"] = asset_ids
            observed["force_refresh"] = force_refresh
            return fake_run

    def fake_build_orchestrator(
            config_dir,
            asset_ids=None,
    ):
        observed["build_asset_ids"] = asset_ids
        return FakeOrchestrator()

    monkeypatch.setattr(
        cli,
        "build_orchestrator",
        fake_build_orchestrator,
    )

    exit_code = cli.main(
        [
            "--config-dir",
            str(tmp_path),
            "--asset",
            "AAPL",
            "--force-refresh",
        ]
    )

    assert exit_code == 0
    assert observed["asset_ids"] == ["AAPL"]
    assert observed["force_refresh"] is True
    assert observed["build_asset_ids"] == ["AAPL"]
    assert observed["asset_ids"] == ["AAPL"]
    assert observed["force_refresh"] is True


def test_output_contains_every_asset_and_overall_status(
    monkeypatch,
    capsys,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    fake_run = run_result(
        assets=(
            asset_result("AAPL"),
            asset_result("MSFT"),
            asset_result("BTC_USD"),
            asset_result("EURUSD"),
            asset_result("GOLD"),
        ),
        status=RunStatus.SUCCESS,
    )

    class FakeOrchestrator:
        def run(self, **kwargs):
            return fake_run

    monkeypatch.setattr(
        cli,
        "build_orchestrator",
        lambda config_dir, asset_ids=None: FakeOrchestrator(),
    )

    exit_code = cli.main(
        [
            "--config-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0

    for asset_id in (
        "AAPL",
        "MSFT",
        "BTC_USD",
        "EURUSD",
        "GOLD",
    ):
        assert asset_id in output

    assert "Overall" in output
    assert RunStatus.SUCCESS.value in output


def test_build_orchestrator_uses_project_config_and_retry_policy(
    monkeypatch,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    class Asset:
        primary_provider = cli.ProviderName.YFINANCE
        backup_provider = cli.ProviderName.STOOQ

    class AssetsConfig:
        assets = {
            "AAPL": Asset(),
        }

    assets_config = AssetsConfig()

    class Retry:
        max_attempts = 7
        initial_delay_seconds = 1.5
        backoff_multiplier = 3.0

    class Cache:
        ttl_hours = 48

    class Paths:
        cache = tmp_path / "cache"

    class Pipeline:
        retry = Retry()
        cache = Cache()
        paths = Paths()

    pipeline_config = Pipeline()

    observed = {}

    def fake_load_project_config(
        assets_path,
        pipeline_path,
    ):
        observed["assets_path"] = assets_path
        observed["pipeline_path"] = pipeline_path
        return assets_config, pipeline_config

    class FakeYFinanceProvider:
        @property
        def name(self):
            return "yfinance"

        def __init__(
            self,
            *,
            max_attempts,
            initial_delay_seconds,
            backoff_multiplier,
        ):
            observed["retry"] = (
                max_attempts,
                initial_delay_seconds,
                backoff_multiplier,
            )

    class FakeCacheManager:
        def __init__(
            self,
            *,
            cache_dir,
            ttl_hours,
        ):
            observed["cache"] = (
                cache_dir,
                ttl_hours,
            )

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            observed["orchestrator"] = kwargs

    monkeypatch.setattr(
        cli,
        "load_project_config",
        fake_load_project_config,
    )
    monkeypatch.setattr(
        cli,
        "YFinanceProvider",
        FakeYFinanceProvider,
    )
    monkeypatch.setattr(
        cli,
        "CacheManager",
        FakeCacheManager,
    )
    monkeypatch.setattr(
        cli,
        "IngestionOrchestrator",
        FakeOrchestrator,
    )

    cli.build_orchestrator(tmp_path)

    assert observed["assets_path"] == (
        tmp_path / "assets.yaml"
    )
    assert observed["pipeline_path"] == (
        tmp_path / "pipeline.yaml"
    )

    assert observed["retry"] == (
        7,
        1.5,
        3.0,
    )

    assert observed["cache"] == (
        tmp_path / "cache",
        48,
    )

    kwargs = observed["orchestrator"]

    assert kwargs["assets_config"] is assets_config
    assert kwargs["pipeline_config"] is pipeline_config

    registration = next(
        iter(kwargs["providers"].values())
    )

    assert (
        registration.price_adjustment_mode
        == "auto_adjust=False"
    )
    assert dict(registration.price_basis) == dict(
        YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS
    )
    assert registration.bar_convention == (
        "provider_native_daily_bar;"
        "timestamp_normalized_to_utc"
    )


def test_aapl_only_does_not_require_twelve_data_api_key(
    monkeypatch,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    class Asset:
        primary_provider = cli.ProviderName.YFINANCE
        backup_provider = cli.ProviderName.STOOQ

    class Assets:
        assets = {
            "AAPL": Asset(),
        }

    class Retry:
        max_attempts = 3
        initial_delay_seconds = 2.0
        backoff_multiplier = 2.0

    class Cache:
        ttl_hours = 24

    class Paths:
        cache = tmp_path / "cache"

    class TwelveData:
        base_url = "https://api.twelvedata.com"
        api_key_env = "TWELVE_DATA_API_KEY"
        timeout_seconds = 10.0

    class Providers:
        twelve_data = TwelveData()

    class Pipeline:
        retry = Retry()
        cache = Cache()
        paths = Paths()
        providers = Providers()

    observed = {}

    class FakeYFinanceProvider:
        name = "yfinance"

        def __init__(self, **kwargs):
            pass

    class FakeTwelveDataProvider:
        def __init__(self, **kwargs):
            raise AssertionError(
                "Twelve Data must not be constructed "
                "for an AAPL-only run."
            )

    class FakeCacheManager:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            observed["providers"] = kwargs["providers"]

    monkeypatch.delenv(
        "TWELVE_DATA_API_KEY",
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "load_project_config",
        lambda assets_path, pipeline_path: (
            Assets(),
            Pipeline(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "YFinanceProvider",
        FakeYFinanceProvider,
    )
    monkeypatch.setattr(
        cli,
        "TwelveDataProvider",
        FakeTwelveDataProvider,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "CacheManager",
        FakeCacheManager,
    )
    monkeypatch.setattr(
        cli,
        "IngestionOrchestrator",
        FakeOrchestrator,
    )

    cli.build_orchestrator(
        tmp_path,
        asset_ids=["AAPL"],
    )

    assert set(observed["providers"]) == {
        cli.ProviderName.YFINANCE,
    }

def test_eurusd_registers_twelve_data_from_config(
    monkeypatch,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    class Asset:
        primary_provider = cli.ProviderName.TWELVE_DATA
        backup_provider = cli.ProviderName.YFINANCE

    class Assets:
        assets = {
            "EURUSD": Asset(),
        }

    class Retry:
        max_attempts = 7
        initial_delay_seconds = 1.5
        backoff_multiplier = 3.0

    class Cache:
        ttl_hours = 24

    class Paths:
        cache = tmp_path / "cache"

    class TwelveData:
        base_url = "https://api.twelvedata.com"
        api_key_env = "TWELVE_DATA_API_KEY"
        timeout_seconds = 12.0

    class Providers:
        twelve_data = TwelveData()

    class Pipeline:
        retry = Retry()
        cache = Cache()
        paths = Paths()
        providers = Providers()

    observed = {}

    class FakeYFinanceProvider:
        name = "yfinance"

        def __init__(self, **kwargs):
            pass

    class FakeTwelveDataProvider:
        name = "twelve_data"

        def __init__(self, **kwargs):
            observed["twelve_data"] = kwargs

    class FakeCacheManager:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            observed["providers"] = kwargs["providers"]

    monkeypatch.setenv(
        "TWELVE_DATA_API_KEY",
        "test-secret",
    )

    monkeypatch.setattr(
        cli,
        "load_project_config",
        lambda assets_path, pipeline_path: (
            Assets(),
            Pipeline(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "YFinanceProvider",
        FakeYFinanceProvider,
    )
    monkeypatch.setattr(
        cli,
        "TwelveDataProvider",
        FakeTwelveDataProvider,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "CacheManager",
        FakeCacheManager,
    )
    monkeypatch.setattr(
        cli,
        "IngestionOrchestrator",
        FakeOrchestrator,
    )

    cli.build_orchestrator(
        tmp_path,
        asset_ids=["EURUSD"],
    )

    assert observed["twelve_data"] == {
        "api_key": "test-secret",
        "base_url": "https://api.twelvedata.com",
        "max_attempts": 7,
        "initial_delay_seconds": 1.5,
        "backoff_multiplier": 3.0,
        "timeout_seconds": 12.0,
    }

    assert set(observed["providers"]) == {
        cli.ProviderName.YFINANCE,
        cli.ProviderName.TWELVE_DATA,
    }


def test_main_reports_configuration_error(
    monkeypatch,
    capsys,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli
    from market_prediction.config import ConfigLoadError

    def fail_build(
            config_dir,
            asset_ids=None,
    ):
        raise ConfigLoadError(
            "invalid test configuration"
        )

    monkeypatch.setattr(
        cli,
        "build_orchestrator",
        fail_build,
    )

    exit_code = cli.main(
        [
            "--config-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "invalid test configuration" in captured.err

def test_eurusd_requires_configured_twelve_data_api_key(
    monkeypatch,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    class Asset:
        primary_provider = cli.ProviderName.TWELVE_DATA
        backup_provider = cli.ProviderName.YFINANCE

    class Assets:
        assets = {
            "EURUSD": Asset(),
        }

    class Retry:
        max_attempts = 3
        initial_delay_seconds = 2.0
        backoff_multiplier = 2.0

    class Cache:
        ttl_hours = 24

    class Paths:
        cache = tmp_path / "cache"

    class TwelveData:
        base_url = "https://api.twelvedata.com"
        api_key_env = "TWELVE_DATA_API_KEY"
        timeout_seconds = 10.0

    class Providers:
        twelve_data = TwelveData()

    class Pipeline:
        retry = Retry()
        cache = Cache()
        paths = Paths()
        providers = Providers()

    monkeypatch.delenv(
        "TWELVE_DATA_API_KEY",
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "load_project_config",
        lambda assets_path, pipeline_path: (
            Assets(),
            Pipeline(),
        ),
    )

    with pytest.raises(
        ValueError,
        match="TWELVE_DATA_API_KEY",
    ):
        cli.build_orchestrator(
            tmp_path,
            asset_ids=["EURUSD"],
        )

def test_aapl_only_does_not_require_alpha_vantage_api_key(
    monkeypatch,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    class Asset:
        primary_provider = cli.ProviderName.YFINANCE
        backup_provider = cli.ProviderName.STOOQ

    class Assets:
        assets = {
            "AAPL": Asset(),
        }

    class Retry:
        max_attempts = 3
        initial_delay_seconds = 2.0
        backoff_multiplier = 2.0

    class Cache:
        ttl_hours = 24

    class Paths:
        cache = tmp_path / "cache"

    class TwelveData:
        base_url = "https://api.twelvedata.com"
        api_key_env = "TWELVE_DATA_API_KEY"
        timeout_seconds = 10.0

    class AlphaVantage:
        base_url = "https://www.alphavantage.co/query"
        api_key_env = "ALPHA_VANTAGE_API_KEY"
        timeout_seconds = 30.0

    class Providers:
        twelve_data = TwelveData()
        alpha_vantage = AlphaVantage()

    class Pipeline:
        retry = Retry()
        cache = Cache()
        paths = Paths()
        providers = Providers()

    observed = {}

    class FakeYFinanceProvider:
        name = "yfinance"

        def __init__(self, **kwargs):
            pass

    class FakeAlphaVantageProvider:
        def __init__(self, **kwargs):
            raise AssertionError(
                "Alpha Vantage must not be constructed "
                "for an AAPL-only run."
            )

    class FakeCacheManager:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            observed["providers"] = kwargs["providers"]

    monkeypatch.delenv(
        "ALPHA_VANTAGE_API_KEY",
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "load_project_config",
        lambda assets_path, pipeline_path: (
            Assets(),
            Pipeline(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "YFinanceProvider",
        FakeYFinanceProvider,
    )
    monkeypatch.setattr(
        cli,
        "AlphaVantageProvider",
        FakeAlphaVantageProvider,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "CacheManager",
        FakeCacheManager,
    )
    monkeypatch.setattr(
        cli,
        "IngestionOrchestrator",
        FakeOrchestrator,
    )

    cli.build_orchestrator(
        tmp_path,
        asset_ids=["AAPL"],
    )

    assert set(observed["providers"]) == {
        cli.ProviderName.YFINANCE,
    }


def test_eurusd_registers_alpha_vantage_from_config(
    monkeypatch,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    class Asset:
        primary_provider = cli.ProviderName.ALPHA_VANTAGE
        backup_provider = cli.ProviderName.YFINANCE

    class Assets:
        assets = {
            "EURUSD": Asset(),
        }

    class Retry:
        max_attempts = 7
        initial_delay_seconds = 1.5
        backoff_multiplier = 3.0

    class Cache:
        ttl_hours = 24

    class Paths:
        cache = tmp_path / "cache"

    class TwelveData:
        base_url = "https://api.twelvedata.com"
        api_key_env = "TWELVE_DATA_API_KEY"
        timeout_seconds = 10.0

    class AlphaVantage:
        base_url = "https://www.alphavantage.co/query"
        api_key_env = "ALPHA_VANTAGE_API_KEY"
        timeout_seconds = 31.0

    class Providers:
        twelve_data = TwelveData()
        alpha_vantage = AlphaVantage()

    class Pipeline:
        retry = Retry()
        cache = Cache()
        paths = Paths()
        providers = Providers()

    observed = {}

    class FakeYFinanceProvider:
        name = "yfinance"

        def __init__(self, **kwargs):
            pass

    class FakeAlphaVantageProvider:
        name = "alpha_vantage"

        def __init__(self, **kwargs):
            observed["alpha_vantage"] = kwargs

    class FakeCacheManager:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            observed["providers"] = kwargs["providers"]

    monkeypatch.setenv(
        "ALPHA_VANTAGE_API_KEY",
        "test-alpha-secret",
    )

    monkeypatch.setattr(
        cli,
        "load_project_config",
        lambda assets_path, pipeline_path: (
            Assets(),
            Pipeline(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "YFinanceProvider",
        FakeYFinanceProvider,
    )
    monkeypatch.setattr(
        cli,
        "AlphaVantageProvider",
        FakeAlphaVantageProvider,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "CacheManager",
        FakeCacheManager,
    )
    monkeypatch.setattr(
        cli,
        "IngestionOrchestrator",
        FakeOrchestrator,
    )

    cli.build_orchestrator(
        tmp_path,
        asset_ids=["EURUSD"],
    )

    assert observed["alpha_vantage"] == {
        "api_key": "test-alpha-secret",
        "base_url": "https://www.alphavantage.co/query",
        "max_attempts": 7,
        "initial_delay_seconds": 1.5,
        "backoff_multiplier": 3.0,
        "timeout_seconds": 31.0,
    }

    assert set(observed["providers"]) == {
        cli.ProviderName.YFINANCE,
        cli.ProviderName.ALPHA_VANTAGE,
    }

    registration = observed["providers"][
        cli.ProviderName.ALPHA_VANTAGE
    ]

    assert (
        registration.price_adjustment_mode
        == "not_applicable_spot_fx"
    )

    assert registration.bar_convention == (
        "alpha_vantage_fx_daily_utc_date_label;"
        "provider_timezone_utc;"
        "date_label_normalized_to_utc_midnight"
    )


def test_eurusd_requires_configured_alpha_vantage_api_key(
    monkeypatch,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    class Asset:
        primary_provider = cli.ProviderName.ALPHA_VANTAGE
        backup_provider = cli.ProviderName.YFINANCE

    class Assets:
        assets = {
            "EURUSD": Asset(),
        }

    class Retry:
        max_attempts = 3
        initial_delay_seconds = 2.0
        backoff_multiplier = 2.0

    class Cache:
        ttl_hours = 24

    class Paths:
        cache = tmp_path / "cache"

    class TwelveData:
        base_url = "https://api.twelvedata.com"
        api_key_env = "TWELVE_DATA_API_KEY"
        timeout_seconds = 10.0

    class AlphaVantage:
        base_url = "https://www.alphavantage.co/query"
        api_key_env = "ALPHA_VANTAGE_API_KEY"
        timeout_seconds = 30.0

    class Providers:
        twelve_data = TwelveData()
        alpha_vantage = AlphaVantage()

    class Pipeline:
        retry = Retry()
        cache = Cache()
        paths = Paths()
        providers = Providers()

    monkeypatch.delenv(
        "ALPHA_VANTAGE_API_KEY",
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "load_project_config",
        lambda assets_path, pipeline_path: (
            Assets(),
            Pipeline(),
        ),
    )

    with pytest.raises(
        ValueError,
        match="ALPHA_VANTAGE_API_KEY",
    ):
        cli.build_orchestrator(
            tmp_path,
            asset_ids=["EURUSD"],
        )