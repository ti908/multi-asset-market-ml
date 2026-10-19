"""Deterministic asset-aware anomaly and calendar tests (offline)."""
from datetime import date
import pandas as pd
import pytest
from market_prediction.config import load_assets_config, load_pipeline_config
from market_prediction.validation import Status, validate_ohlcv
from pathlib import Path

@pytest.fixture
def assets():
    return load_assets_config("config/assets.yaml").assets

@pytest.fixture
def config():
    return load_pipeline_config("config/pipeline.yaml")

def frame(days, closes, volumes=None):
    if volumes is None:
        volumes = [100.] * len(days)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(days, utc=True),
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * .99 for c in closes], "close": closes,
        "adj_close": closes, "volume": volumes,
        "source": ["yfinance"] * len(days),
    })

def rules(report):
    return {x.rule for x in report.soft_anomalies}

def test_same_30_percent_move_flags_fx_not_crypto(assets, config):
    data = frame(["2026-01-05", "2026-01-06"], [100., 130.])
    fx = validate_ohlcv(data, assets["EURUSD"], config)
    btc = validate_ohlcv(data, assets["BTC_USD"], config)
    assert "large_daily_move" in rules(fx)
    assert "large_daily_move" not in rules(btc)
    assert fx.status is Status.PASSED and btc.status is Status.PASSED

def test_zero_volume_ignored_for_fx_flagged_for_equity(assets, config):
    data = frame(["2026-01-05", "2026-01-06"], [100., 101.], [0., 0.])
    fx = validate_ohlcv(data, assets["EURUSD"], config)
    equity = validate_ohlcv(data, assets["AAPL"], config)
    assert "zero_volume" not in rules(fx)
    assert "zero_volume" in rules(equity)
    assert "zero_volume" not in fx.skipped_checks

def test_fx_volume_spike_checked_when_zero_volume_check_disabled(assets, config):
    days = pd.bdate_range(
        "2026-01-05",
        periods=22,
    ).strftime("%Y-%m-%d").tolist()

    data = frame(
        days,
        [100.] * 22,
        [100.] * 21 + [1000.],
    )

    r = validate_ohlcv(data, assets["EURUSD"], config)

    assert "zero_volume" not in rules(r)
    assert "volume_spike" in rules(r)
    assert "volume_spike" not in r.skipped_checks

def test_volume_spike_trailing_median(assets, config):
    days = pd.bdate_range("2026-01-05", periods=22).strftime("%Y-%m-%d").tolist()
    data = frame(days, [100.] * 22, [100.] * 21 + [1000.])
    r = validate_ohlcv(data, assets["AAPL"], config)
    assert "volume_spike" in rules(r)

def test_crypto_gap_counts_calendar_days(assets, config):
    data = frame(["2026-01-05", "2026-01-09"], [100., 100.])
    r = validate_ohlcv(data, assets["BTC_USD"], config)
    assert "missing_sessions" in rules(r)
    assert any("3 consecutive" in x.message for x in r.soft_anomalies if x.rule == "missing_sessions")

def test_equity_holiday_is_not_missing(assets, config):
    # Martin Luther King Jr. Day: Monday Jan 19, 2026; NYSE closed.
    data = frame(["2026-01-16", "2026-01-20"], [100., 100.])
    r = validate_ohlcv(data, assets["AAPL"], config)
    assert "missing_sessions" not in rules(r)

def test_gold_comex_holiday_not_naive_weekday(assets, config):
    data = frame(["2026-01-16", "2026-01-20"], [100., 100.])
    r = validate_ohlcv(data, assets["GOLD"], config)
    assert r.status is Status.PASSED

def test_fx_weekends_not_missing(assets, config):
    data = frame(["2026-01-09", "2026-01-12"], [100., 100.])
    r = validate_ohlcv(data, assets["EURUSD"], config)
    assert "missing_sessions" not in rules(r)

def test_max_missing_sessions_means_consecutive_not_total(assets, config):
    data = frame(["2026-01-05", "2026-01-08", "2026-01-09", "2026-01-13"], [100.]*4)
    r = validate_ohlcv(data, assets["EURUSD"], config)
    assert "missing_sessions" not in rules(r)  # two then one; limit=2

def test_equity_move_mentions_split_adjusted_price_basis(
    assets,
    config,
):
    data = frame(
        ["2026-01-05", "2026-01-06"],
        [100.0, 130.0],
    )

    report = validate_ohlcv(
        data,
        assets["AAPL"],
        config,
    )

    messages = [
        issue.message
        for issue in report.soft_anomalies
        if issue.rule == "large_daily_move"
    ]

    assert len(messages) == 1
    assert "corporate action or source-data restatement" in messages[0]
    assert "split-adjusted" in messages[0]
    assert "auto_adjust=False" in messages[0]
    assert "unadjusted stock split" not in messages[0]

def test_aapl_split_fixture_matches_split_adjusted_close_behavior(
    assets,
    config,
):
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "aapl_2020_split_auto_adjust_false.csv"
    )

    observed = pd.read_csv(fixture_path)

    split_row = observed.loc[
        observed["date"] == "2020-08-31"
    ].iloc[0]

    pre_split_row = observed.loc[
        observed["date"] == "2020-08-28"
    ].iloc[0]

    assert split_row["stock_splits"] == 4.0

    close_ratio = (
        split_row["close"]
        / pre_split_row["close"]
    )

    assert 0.5 < close_ratio < 2.0

    assert not observed["close"].equals(
        observed["adj_close"]
    )
