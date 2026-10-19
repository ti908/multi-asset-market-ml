"""Offline contract tests for validated market-data cache persistence."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from market_prediction.cache import (
    CacheManager,
    CacheState,
    ValidatedDataset,
    ValidationGateError,
    YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
)
from market_prediction.validation import ValidationReport
from market_prediction.validation.fingerprint import dataframe_fingerprint


NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)


def market_frame(
    *,
    first_adj_close: float = 100.5,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-05T00:00:00Z",
                    "2026-01-06T00:00:00Z",
                ],
                utc=True,
            ),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "adj_close": [first_adj_close, 101.5],
            "volume": [1_000_000, 1_100_000],
            "source": ["yfinance", "yfinance"],
        }
    )


def validated_dataset(
    frame: pd.DataFrame | None = None,
) -> ValidatedDataset:
    frame = market_frame() if frame is None else frame

    report = ValidationReport(
        asset="AAPL",
        row_count=len(frame),
        dataset_fingerprint=dataframe_fingerprint(frame),
    )

    return ValidatedDataset.from_validation(
        frame,
        report,
    )


def manager(tmp_path, *, now=NOW) -> CacheManager:
    return CacheManager(
        cache_dir=tmp_path,
        ttl_hours=24,
        now=lambda: now,
    )


def write_aapl(
    cache: CacheManager,
    dataset: ValidatedDataset | None = None,
    *,
    fetch_timestamp=NOW,
):
    return cache.write(
        dataset or validated_dataset(),
        provider="yfinance",
        fetch_timestamp=fetch_timestamp,
        price_adjustment_mode="auto_adjust=False",
        price_basis=YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
    )


def test_parquet_json_round_trip_preserves_data_and_metadata(tmp_path):
    cache = manager(tmp_path)
    original = validated_dataset()

    written = write_aapl(
        cache,
        original,
    )

    result = cache.read("AAPL")

    assert result.state is CacheState.FRESH
    assert result.metadata is not None
    assert result.data is not None

    pd.testing.assert_frame_equal(
        result.data,
        original.data,
    )

    assert result.metadata.asset == "AAPL"
    assert result.metadata.provider == "yfinance"
    assert result.metadata.fetch_timestamp == NOW
    assert result.metadata.date_start == date(2026, 1, 5)
    assert result.metadata.date_end == date(2026, 1, 6)
    assert result.metadata.row_count == 2
    assert result.metadata.validation_status.value == "PASSED"
    assert result.metadata.price_adjustment_mode == "auto_adjust=False"
    assert result.metadata.dataset_id == original.dataset_id
    assert written.dataset_id == original.dataset_id


def test_metadata_json_contains_required_price_basis(tmp_path):
    cache = manager(tmp_path)

    metadata = write_aapl(cache)

    pointer_path = tmp_path / "AAPL" / "current.json"
    pointer = json.loads(
        pointer_path.read_text(encoding="utf-8")
    )

    metadata_path = (
        tmp_path
        / "AAPL"
        / "generations"
        / pointer["generation"]
        / "metadata.json"
    )

    raw = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert raw["asset"] == "AAPL"
    assert raw["provider"] == "yfinance"
    assert raw["price_adjustment_mode"] == "auto_adjust=False"
    assert raw["validation_status"] == "PASSED"

    assert (
        raw["price_basis"]["close"]["adjustment"]
        == "split_adjusted"
    )
    assert (
        raw["price_basis"]["adj_close"]["adjustment"]
        == "split_and_dividend_adjusted"
    )
    assert (
        raw["price_basis"]["volume"]["adjustment"]
        == "split_consistent_volume"
    )

    assert raw["dataset_id"] == metadata.dataset_id


def test_soft_anomalies_are_summarized_in_metadata(tmp_path):
    frame = market_frame()

    report = ValidationReport(
        asset="AAPL",
        row_count=len(frame),
        dataset_fingerprint=dataframe_fingerprint(frame),
    )

    report.add(
        "large_daily_move",
        "warning",
        date(2026, 1, 6),
        "close",
        "Example warning",
    )
    report.add(
        "large_daily_move",
        "warning",
        date(2026, 1, 6),
        "close",
        "Second example warning",
    )
    report.add(
        "volume_spike",
        "warning",
        date(2026, 1, 6),
        "volume",
        "Example volume warning",
    )

    dataset = ValidatedDataset.from_validation(
        frame,
        report,
    )

    cache = manager(tmp_path)
    metadata = write_aapl(
        cache,
        dataset,
    )

    summary = {
        item.rule: item.count
        for item in metadata.soft_anomalies
    }

    assert summary == {
        "large_daily_move": 2,
        "volume_spike": 1,
    }


def test_fresh_cache_is_reported_fresh(tmp_path):
    cache = manager(
        tmp_path,
        now=NOW,
    )

    write_aapl(
        cache,
        fetch_timestamp=NOW - timedelta(hours=23),
    )

    result = cache.read("AAPL")

    assert result.state is CacheState.FRESH
    assert result.exists is True
    assert result.is_fresh is True


def test_cache_at_ttl_boundary_is_fresh(tmp_path):
    cache = manager(
        tmp_path,
        now=NOW,
    )

    write_aapl(
        cache,
        fetch_timestamp=NOW - timedelta(hours=24),
    )

    result = cache.read("AAPL")

    assert result.state is CacheState.FRESH


def test_cache_older_than_ttl_is_stale(tmp_path):
    cache = manager(
        tmp_path,
        now=NOW,
    )

    write_aapl(
        cache,
        fetch_timestamp=NOW - timedelta(hours=24, seconds=1),
    )

    result = cache.read("AAPL")

    assert result.state is CacheState.STALE
    assert result.exists is True
    assert result.is_fresh is False
    assert result.data is not None
    assert result.metadata is not None


def test_missing_cache_has_explicit_state(tmp_path):
    result = manager(tmp_path).read("AAPL")

    assert result.state is CacheState.MISSING
    assert result.data is None
    assert result.metadata is None
    assert result.error is None


def test_corrupted_pointer_is_detected(tmp_path):
    cache = manager(tmp_path)
    write_aapl(cache)

    pointer_path = tmp_path / "AAPL" / "current.json"
    pointer_path.write_text(
        "{not-json",
        encoding="utf-8",
    )

    result = cache.read("AAPL")

    assert result.state is CacheState.CORRUPTED
    assert result.data is None
    assert result.metadata is None
    assert result.error is not None


def test_corrupted_parquet_is_detected(tmp_path):
    cache = manager(tmp_path)
    write_aapl(cache)

    pointer_path = tmp_path / "AAPL" / "current.json"
    pointer = json.loads(
        pointer_path.read_text(encoding="utf-8")
    )

    data_path = (
        tmp_path
        / "AAPL"
        / "generations"
        / pointer["generation"]
        / "data.parquet"
    )

    data_path.write_bytes(b"not-a-parquet-file")

    result = cache.read("AAPL")

    assert result.state is CacheState.CORRUPTED
    assert result.data is None
    assert result.error is not None


def test_metadata_data_fingerprint_mismatch_is_corrupted(tmp_path):
    cache = manager(tmp_path)
    write_aapl(cache)

    pointer_path = tmp_path / "AAPL" / "current.json"
    pointer = json.loads(
        pointer_path.read_text(encoding="utf-8")
    )

    metadata_path = (
        tmp_path
        / "AAPL"
        / "generations"
        / pointer["generation"]
        / "metadata.json"
    )

    raw = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    raw["dataset_id"] = "wrong-dataset-id"

    metadata_path.write_text(
        json.dumps(raw),
        encoding="utf-8",
    )

    result = cache.read("AAPL")

    assert result.state is CacheState.CORRUPTED
    assert result.data is None
    assert result.error is not None


def test_missing_generation_file_is_corrupted(tmp_path):
    cache = manager(tmp_path)
    write_aapl(cache)

    pointer_path = tmp_path / "AAPL" / "current.json"
    pointer = json.loads(
        pointer_path.read_text(encoding="utf-8")
    )

    metadata_path = (
        tmp_path
        / "AAPL"
        / "generations"
        / pointer["generation"]
        / "metadata.json"
    )
    metadata_path.unlink()

    result = cache.read("AAPL")

    assert result.state is CacheState.CORRUPTED
    assert result.data is None
    assert result.error is not None


def test_cache_write_rejects_raw_dataframe(tmp_path):
    cache = manager(tmp_path)

    with pytest.raises(
        ValidationGateError,
        match="ValidatedDataset",
    ):
        cache.write(
            market_frame(),
            provider="yfinance",
            fetch_timestamp=NOW,
            price_adjustment_mode="auto_adjust=False",
            price_basis=YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
        )

    assert cache.read("AAPL").state is CacheState.MISSING


def test_blocked_data_cannot_reach_cache_write_api(tmp_path):
    frame = market_frame()

    report = ValidationReport(
        asset="AAPL",
        row_count=len(frame),
        dataset_fingerprint=dataframe_fingerprint(frame),
    )

    report.add(
        "negative_volume",
        "error",
        date(2026, 1, 6),
        "volume",
        "Negative volume",
    )

    with pytest.raises(ValidationGateError):
        ValidatedDataset.from_validation(
            frame,
            report,
        )

    assert manager(tmp_path).read("AAPL").state is CacheState.MISSING


def test_failed_replacement_preserves_previous_good_cache(
    tmp_path,
    monkeypatch,
):
    cache = manager(tmp_path)

    original = validated_dataset()
    write_aapl(cache, original)

    import market_prediction.cache.manager as manager_module

    real_replace = manager_module.os.replace

    def fail_pointer_publish(src, dst):
        if str(dst).endswith("current.json"):
            raise OSError("simulated pointer publication failure")
        return real_replace(src, dst)

    monkeypatch.setattr(
        manager_module.os,
        "replace",
        fail_pointer_publish,
    )

    changed_frame = market_frame(
        first_adj_close=99.25,
    )
    changed = validated_dataset(
        changed_frame,
    )

    with pytest.raises(
        OSError,
        match="simulated pointer publication failure",
    ):
        write_aapl(
            cache,
            changed,
            fetch_timestamp=NOW + timedelta(hours=1),
        )

    result = cache.read("AAPL")

    assert result.state is CacheState.FRESH
    assert result.data is not None

    pd.testing.assert_frame_equal(
        result.data,
        original.data,
    )


def test_restatement_is_detected_in_overlapping_history(tmp_path):
    cache = manager(tmp_path)

    original = validated_dataset()
    write_aapl(cache, original)

    changed_frame = market_frame(
        first_adj_close=99.25,
    )
    changed = validated_dataset(
        changed_frame,
    )

    metadata = write_aapl(
        cache,
        changed,
        fetch_timestamp=NOW + timedelta(hours=1),
    )

    assert metadata.restatement is not None
    assert metadata.restatement.detected is True
    assert metadata.restatement.overlap_start == date(2026, 1, 5)
    assert metadata.restatement.overlap_end == date(2026, 1, 6)
    assert metadata.restatement.changed_rows == 1
    assert metadata.restatement.changed_columns == (
        "adj_close",
    )

    result = cache.read("AAPL")

    assert result.data is not None
    pd.testing.assert_frame_equal(
        result.data,
        changed.data,
    )


def test_identical_refresh_records_no_restatement(tmp_path):
    cache = manager(tmp_path)

    first = validated_dataset()
    write_aapl(cache, first)

    second = validated_dataset()

    metadata = write_aapl(
        cache,
        second,
        fetch_timestamp=NOW + timedelta(hours=1),
    )

    assert metadata.restatement is not None
    assert metadata.restatement.detected is False
    assert metadata.restatement.changed_rows == 0
    assert metadata.restatement.changed_columns == ()


def test_first_write_has_no_restatement_comparison(tmp_path):
    cache = manager(tmp_path)

    metadata = write_aapl(cache)

    assert metadata.restatement is None


def test_successful_refresh_publishes_complete_replacement(tmp_path):
    cache = manager(tmp_path)

    write_aapl(cache)

    replacement_frame = market_frame().iloc[[1]].reset_index(drop=True)

    replacement_report = ValidationReport(
        asset="AAPL",
        row_count=len(replacement_frame),
        dataset_fingerprint=dataframe_fingerprint(replacement_frame),
    )

    replacement = ValidatedDataset.from_validation(
        replacement_frame,
        replacement_report,
    )

    write_aapl(
        cache,
        replacement,
        fetch_timestamp=NOW + timedelta(hours=1),
    )

    result = cache.read("AAPL")

    assert result.data is not None
    assert len(result.data) == 1
    pd.testing.assert_frame_equal(
        result.data,
        replacement.data,
    )
    