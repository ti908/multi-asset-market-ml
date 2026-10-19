"""Tests for validated-cache domain contracts."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from market_prediction.cache import (
    CacheMetadata,
    CacheReadResult,
    CacheState,
    HistoricalStability,
    PriceAdjustment,
    SoftAnomalySummary,
    ValidatedDataset,
    ValidationGateError,
    YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
)
from market_prediction.validation import Status, ValidationReport
from market_prediction.validation.fingerprint import dataframe_fingerprint


def market_frame() -> pd.DataFrame:
    """Return a minimal normalized provider-contract DataFrame."""

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
            "adj_close": [100.5, 101.5],
            "volume": [1_000_000, 1_100_000],
            "source": ["yfinance", "yfinance"],
        }
    )


def passed_report(frame: pd.DataFrame) -> ValidationReport:
    """Return a PASSED report bound to the supplied frame."""

    return ValidationReport(
        asset="AAPL",
        row_count=len(frame),
        dataset_fingerprint=dataframe_fingerprint(frame),
    )


def metadata() -> CacheMetadata:
    """Return representative cache metadata."""

    return CacheMetadata(
        asset="AAPL",
        provider="yfinance",
        fetch_timestamp=datetime(
            2026,
            1,
            6,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        date_start=date(2026, 1, 5),
        date_end=date(2026, 1, 6),
        row_count=2,
        price_adjustment_mode="auto_adjust=False",
        validation_status=Status.PASSED,
        soft_anomalies=(
            SoftAnomalySummary(
                rule="large_daily_move",
                count=1,
            ),
        ),
        price_basis=YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
        dataset_id="example-dataset-id",
    )


def test_passed_report_creates_validated_dataset():
    frame = market_frame()
    report = passed_report(frame)

    dataset = ValidatedDataset.from_validation(
        frame,
        report,
    )

    pd.testing.assert_frame_equal(
        dataset.data,
        frame,
    )
    assert dataset.report is report
    assert dataset.report.status is Status.PASSED
    assert dataset.dataset_id == dataframe_fingerprint(frame)


def test_blocked_report_cannot_cross_validation_gate():
    frame = market_frame()
    report = passed_report(frame)

    report.add(
        "negative_volume",
        "error",
        date(2026, 1, 6),
        "volume",
        "Negative volume",
    )

    assert report.status is Status.BLOCKED

    with pytest.raises(
        ValidationGateError,
        match="validation status is BLOCKED",
    ):
        ValidatedDataset.from_validation(
            frame,
            report,
        )


def test_skipped_check_report_cannot_cross_validation_gate():
    frame = market_frame()
    report = passed_report(frame)

    report.skip("large_daily_move")

    assert report.status is Status.BLOCKED

    with pytest.raises(
        ValidationGateError,
        match="validation status is BLOCKED",
    ):
        ValidatedDataset.from_validation(
            frame,
            report,
        )


def test_validated_dataset_cannot_be_constructed_directly():
    frame = market_frame()
    report = passed_report(frame)

    with pytest.raises(
        ValidationGateError,
        match="cannot be constructed directly",
    ):
        ValidatedDataset(
            frame,
            report,
        )


def test_report_row_count_must_match_dataset():
    frame = market_frame()

    report = ValidationReport(
        asset="AAPL",
        row_count=999,
        dataset_fingerprint=dataframe_fingerprint(frame),
    )

    assert report.status is Status.PASSED

    with pytest.raises(
        ValidationGateError,
        match="does not match dataset row count",
    ):
        ValidatedDataset.from_validation(
            frame,
            report,
        )

def test_same_length_different_dataframe_cannot_reuse_passed_report():
    original = market_frame()
    report = passed_report(original)

    substituted = original.copy(deep=True)
    substituted.loc[0, "close"] = 100.75

    assert len(substituted) == len(original)

    with pytest.raises(
        ValidationGateError,
        match="does not match the data",
    ):
        ValidatedDataset.from_validation(
            substituted,
            report,
        )


def test_passed_report_without_fingerprint_is_rejected():
    frame = market_frame()

    report = ValidationReport(
        asset="AAPL",
        row_count=len(frame),
    )

    assert report.status is Status.PASSED

    with pytest.raises(
        ValidationGateError,
        match="has no dataset fingerprint",
    ):
        ValidatedDataset.from_validation(
            frame,
            report,
        )

def test_validated_dataset_isolated_from_original_dataframe_mutation():
    frame = market_frame()
    report = passed_report(frame)

    dataset = ValidatedDataset.from_validation(
        frame,
        report,
    )

    frame.loc[0, "close"] = -999.0

    assert dataset.data.loc[0, "close"] == 101.0


def test_validated_dataset_data_property_returns_defensive_copy():
    frame = market_frame()
    report = passed_report(frame)

    dataset = ValidatedDataset.from_validation(
        frame,
        report,
    )

    exposed = dataset.data
    exposed.loc[0, "close"] = -999.0

    assert dataset.data.loc[0, "close"] == 101.0


def test_yfinance_price_basis_records_verified_column_semantics():
    basis = YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS

    assert set(basis) == {
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    }

    for column in (
        "open",
        "high",
        "low",
        "close",
    ):
        assert (
            basis[column].adjustment
            is PriceAdjustment.SPLIT_ADJUSTED
        )
        assert (
            basis[column].historical_stability
            is HistoricalStability.MAY_BE_RESTATED
        )

    assert (
        basis["adj_close"].adjustment
        is PriceAdjustment.SPLIT_AND_DIVIDEND_ADJUSTED
    )

    assert (
        basis["volume"].adjustment
        is PriceAdjustment.SPLIT_CONSISTENT_VOLUME
    )


@pytest.mark.parametrize(
    ("state", "exists", "is_fresh"),
    [
        (CacheState.FRESH, True, True),
        (CacheState.STALE, True, False),
        (CacheState.MISSING, False, False),
        (CacheState.CORRUPTED, False, False),
    ],
)
def test_cache_read_result_has_predictable_state_semantics(
    state,
    exists,
    is_fresh,
):
    frame = market_frame()
    cache_metadata = metadata()

    if state in {
        CacheState.FRESH,
        CacheState.STALE,
    }:
        result = CacheReadResult(
            state=state,
            data=frame,
            metadata=cache_metadata,
        )
    else:
        result = CacheReadResult(
            state=state,
            data=None,
            metadata=None,
        )

    assert result.state is state
    assert result.exists is exists
    assert result.is_fresh is is_fresh


def test_cache_metadata_records_required_provenance():
    cache_metadata = metadata()

    assert cache_metadata.asset == "AAPL"
    assert cache_metadata.provider == "yfinance"
    assert cache_metadata.fetch_timestamp == datetime(
        2026,
        1,
        6,
        12,
        0,
        tzinfo=timezone.utc,
    )
    assert (
            cache_metadata.price_adjustment_mode
            == "auto_adjust=False"
    )
    assert cache_metadata.date_start == date(2026, 1, 5)
    assert cache_metadata.date_end == date(2026, 1, 6)
    assert cache_metadata.row_count == 2
    assert cache_metadata.validation_status is Status.PASSED
    assert cache_metadata.soft_anomalies == (
        SoftAnomalySummary(
            rule="large_daily_move",
            count=1,
        ),
    )
    assert (
        cache_metadata.price_basis["adj_close"].adjustment
        is PriceAdjustment.SPLIT_AND_DIVIDEND_ADJUSTED
    )
    assert cache_metadata.dataset_id == "example-dataset-id"

def test_dataset_fingerprint_ignores_dataframe_index():
    first = market_frame()

    second = first.copy(deep=True)
    second.index = [10, 20]

    assert dataframe_fingerprint(
        first
    ) == dataframe_fingerprint(
        second
    )