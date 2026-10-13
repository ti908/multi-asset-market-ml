"""Deterministic schema validation tests (offline)."""
from datetime import date
import numpy as np
import pandas as pd
import pytest
from market_prediction.config import load_assets_config, load_pipeline_config
from market_prediction.validation import Status, validate_ohlcv

@pytest.fixture
def assets():
    return load_assets_config("config/assets.yaml").assets

@pytest.fixture
def config():
    return load_pipeline_config("config/pipeline.yaml")

@pytest.fixture
def clean():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"], utc=True),
        "open": [100., 101., 102.], "high": [103., 104., 105.],
        "low": [99., 100., 101.], "close": [102., 103., 104.],
        "adj_close": [102., 103., 104.], "volume": [100., 120., 130.],
        "source": ["yfinance"] * 3,
    })

def issues(report):
    return {(i.rule, i.date, i.column) for i in report.hard_failures}

def test_clean_dataset_passes_with_empty_report(clean, assets, config):
    r = validate_ohlcv(clean, assets["AAPL"], config)
    assert r.status is Status.PASSED and r.can_progress
    assert not r.hard_failures and not r.soft_anomalies and not r.skipped_checks

@pytest.mark.parametrize("column", ["open", "high", "low", "close", "adj_close", "volume", "timestamp", "source"])
def test_missing_column_blocks_and_records_skipped_checks(clean, assets, config, column):
    r = validate_ohlcv(clean.drop(columns=[column]), assets["AAPL"], config)
    assert ("missing_column", None, column) in issues(r)
    assert r.status is Status.BLOCKED
    assert "large_daily_move" in r.skipped_checks and "ohlc_relationships" in r.skipped_checks

def test_unexpected_column_blocks(clean, assets, config):
    r = validate_ohlcv(clean.assign(junk=1), assets["AAPL"], config)
    assert ("unexpected_column", None, "junk") in issues(r)

def test_duplicate_column_blocks(clean, assets, config):
    broken = clean.copy()
    broken.columns = ["open" if x == "close" else x for x in broken.columns]
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert r.status is Status.BLOCKED

@pytest.mark.parametrize("value,rule", [(None, "null_or_non_numeric"), (float("inf"), "non_finite"), (0., "non_positive_price"), (-1., "non_positive_price")])
def test_invalid_close(clean, assets, config, value, rule):
    broken = clean.copy()
    broken.loc[1, "close"] = value
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert (rule, date(2026, 1, 6), "close") in issues(r)
    assert r.status is Status.BLOCKED

def test_invalid_open_and_low(clean, assets, config):
    broken = clean.copy()
    broken.loc[1, "open"] = 200.
    broken.loc[2, "low"] = -1.
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("price_outside_range", date(2026, 1, 6), "open") in issues(r)
    assert ("non_positive_price", date(2026, 1, 7), "low") in issues(r)

def test_high_below_low(clean, assets, config):
    broken = clean.copy()
    broken.loc[1, "high"] = 90.
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("high_below_low", date(2026, 1, 6), "high") in issues(r)

def test_close_outside_range(clean, assets, config):
    broken = clean.copy()
    broken.loc[1, "close"] = 110.
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("price_outside_range", date(2026, 1, 6), "close") in issues(r)

def test_negative_volume(clean, assets, config):
    broken = clean.copy()
    broken.loc[1, "volume"] = -5.
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("negative_volume", date(2026, 1, 6), "volume") in issues(r)

def test_non_finite_volume(clean, assets, config):
    broken = clean.copy()
    broken.loc[1, "volume"] = np.inf
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("non_finite", date(2026, 1, 6), "volume") in issues(r)

def test_null_optional_fields_are_allowed(clean, assets, config):
    broken = clean.copy()
    broken["volume"] = np.nan
    broken["adj_close"] = np.nan
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert r.status is Status.PASSED

def test_duplicate_timestamp(clean, assets, config):
    broken = clean.copy()
    broken.loc[2, "timestamp"] = broken.loc[1, "timestamp"]
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("duplicate_timestamp", date(2026, 1, 6), "timestamp") in issues(r)
    assert "missing_sessions" in r.skipped_checks

def test_non_monotonic_timestamp(clean, assets, config):
    broken = clean.iloc[::-1].reset_index(drop=True)
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("non_monotonic_timestamp", None, "timestamp") in issues(r)

def test_malformed_timestamp_reports_skips_and_independent_defects(clean, assets, config):
    broken = clean.copy()
    broken["timestamp"] = broken["timestamp"].astype(str)
    broken.loc[1, "timestamp"] = "not-a-date"
    broken.loc[2, "volume"] = -2.
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("malformed_timestamp", None, "timestamp") in issues(r)
    assert ("negative_volume", date(2026, 1, 7), "volume") in issues(r)
    assert set(["timestamp_order", "duplicate_timestamp", "large_daily_move", "missing_sessions"]).issubset(r.skipped_checks)

def test_multiple_problems_exact_rule_date_column(clean, assets, config):
    broken = clean.copy()
    broken.loc[1, "close"] = -3.
    broken.loc[1, "high"] = 90.
    broken.loc[2, "volume"] = -4.
    broken.loc[2, "timestamp"] = broken.loc[1, "timestamp"]
    r = validate_ohlcv(broken, assets["AAPL"], config)
    expected = {
        ("non_positive_price", date(2026, 1, 6), "close"),
        ("high_below_low", date(2026, 1, 6), "high"),
        ("negative_volume", date(2026, 1, 6), "volume"),
        ("duplicate_timestamp", date(2026, 1, 6), "timestamp"),
        ("price_outside_range", date(2026, 1, 6), "open"),
    }
    assert expected == issues(r)
    assert r.status is Status.BLOCKED and not r.can_progress


def test_invalid_source_blocks(clean, assets, config):
    broken = clean.copy()
    broken.loc[1, "source"] = None
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("invalid_source", date(2026, 1, 6), "source") in issues(r)

def test_non_finite_adjusted_close_blocks(clean, assets, config):
    broken = clean.copy()
    broken.loc[1, "adj_close"] = np.inf
    r = validate_ohlcv(broken, assets["AAPL"], config)
    assert ("non_finite", date(2026, 1, 6), "adj_close") in issues(r)
