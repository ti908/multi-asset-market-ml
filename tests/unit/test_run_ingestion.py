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
        lambda config_dir: FakeOrchestrator(),
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

    monkeypatch.setattr(
        cli,
        "build_orchestrator",
        lambda config_dir: FakeOrchestrator(),
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
        lambda config_dir: FakeOrchestrator(),
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

    assets_config = object()

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


def test_production_registry_contains_only_implemented_yfinance(
    monkeypatch,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli

    assets_config = object()

    class Retry:
        max_attempts = 3
        initial_delay_seconds = 2.0
        backoff_multiplier = 2.0

    class Cache:
        ttl_hours = 24

    class Paths:
        cache = tmp_path / "cache"

    class Pipeline:
        retry = Retry()
        cache = Cache()
        paths = Paths()

    observed = {}

    monkeypatch.setattr(
        cli,
        "load_project_config",
        lambda assets_path, pipeline_path: (
            assets_config,
            Pipeline(),
        ),
    )

    class FakeYFinanceProvider:
        def __init__(self, **kwargs):
            pass

        @property
        def name(self):
            return "yfinance"

    class FakeCacheManager:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            observed.update(kwargs)

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

    providers = observed["providers"]

    assert len(providers) == 1
    assert {
        provider_name.value
        for provider_name in providers
    } == {"yfinance"}


def test_main_reports_configuration_error(
    monkeypatch,
    capsys,
    tmp_path,
):
    import market_prediction.pipeline.run_ingestion as cli
    from market_prediction.config import ConfigLoadError

    def fail_build(config_dir):
        raise ConfigLoadError("invalid test configuration")

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