"""Accumulating OHLCV schema and financial consistency checks."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import numpy as np
import pandas as pd
from market_prediction.config import AssetConfig, PipelineConfig

COLUMNS = ("timestamp", "open", "high", "low", "close", "adj_close", "volume", "source")
PRICE_COLUMNS = ("open", "high", "low", "close")
NUMERIC_COLUMNS = (*PRICE_COLUMNS, "adj_close", "volume")
ANOMALY_CHECKS = ("large_daily_move", "volume_spike", "zero_volume", "missing_sessions")

class Status(str, Enum):
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"

@dataclass(frozen=True)
class Issue:
    asset: str
    date: date | None
    column: str | None
    rule: str
    severity: str
    message: str

@dataclass
class ValidationReport:
    asset: str
    row_count: int
    hard_failures: list[Issue] = field(default_factory=list)
    soft_anomalies: list[Issue] = field(default_factory=list)
    skipped_checks: list[str] = field(default_factory=list)

    @property
    def status(self) -> Status:
        return Status.BLOCKED if self.hard_failures or self.skipped_checks else Status.PASSED

    @property
    def can_progress(self) -> bool:
        return self.status is Status.PASSED

    def add(self, rule: str, severity: str, date_value: date | None, column: str | None, message: str) -> None:
        issue = Issue(self.asset, date_value, column, rule, severity, message)
        (self.hard_failures if severity == "error" else self.soft_anomalies).append(issue)

    def skip(self, *checks: str) -> None:
        for check in checks:
            if check not in self.skipped_checks:
                self.skipped_checks.append(check)


def validate_ohlcv(frame: pd.DataFrame, asset: AssetConfig, config: PipelineConfig) -> ValidationReport:
    """Return all observable issues; never mutate input or raise for malformed market rows.

    Structural failures skip dependent checks. Invalid numeric values skip row-level
    dependent comparisons and may skip entire return/volume-spike checks when unsafe.
    """
    report = ValidationReport(asset=asset.ticker, row_count=len(frame))
    if frame.empty:
        report.add("empty_dataset", "error", None, None, "Dataset has no observations")
        report.skip(*ANOMALY_CHECKS)
        return report
    if not frame.columns.is_unique:
        report.add("duplicate_column", "error", None, None, "Column labels are not unique")
        report.skip("timestamp_order", "duplicate_timestamp", "numeric_checks", "ohlc_relationships", *ANOMALY_CHECKS)
        return report
    missing = set(COLUMNS) - set(frame.columns)
    unexpected = set(frame.columns) - set(COLUMNS)
    for col in sorted(missing):
        report.add("missing_column", "error", None, col, f"Missing required column {col}")
    for col in sorted(unexpected):
        report.add("unexpected_column", "error", None, col, f"Unexpected column {col}")
    if missing:
        report.skip("timestamp_order", "duplicate_timestamp", "numeric_checks", "ohlc_relationships", *ANOMALY_CHECKS)
        return report

    stamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    # Reject unparseable dates, and reject date-like strings even if parsable:
    # normalized providers must supply datetime dtype.
    dtype_ok = isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(frame["timestamp"])
    bad_stamp = stamps.isna()
    if not dtype_ok:
        report.add("malformed_timestamp_dtype", "error", None, "timestamp", "timestamp must be datetime dtype")
    for position in np.flatnonzero(bad_stamp.to_numpy()):
        report.add("malformed_timestamp", "error", None, "timestamp", f"Unparseable timestamp at row {position}")
    if bad_stamp.any() or not dtype_ok:
        report.skip("timestamp_order", "duplicate_timestamp", "ohlc_relationships", *ANOMALY_CHECKS)
        # Still report independent numeric defects even if timestamp is malformed.
    else:
        for position in np.flatnonzero(stamps.duplicated().to_numpy()):
            report.add("duplicate_timestamp", "error", stamps.iloc[position].date(), "timestamp", "Repeated timestamp")
        if not stamps.is_monotonic_increasing:
            report.add("non_monotonic_timestamp", "error", None, "timestamp", "Timestamps must be ascending")

    dates = [None if pd.isna(x) else x.date() for x in stamps]
    for i, value in enumerate(frame["source"]):
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            report.add("invalid_source", "error", dates[i], "source", f"Missing or invalid source at row {i}")
    numeric: dict[str, pd.Series] = {}
    for col in NUMERIC_COLUMNS:
        series = pd.to_numeric(frame[col], errors="coerce")
        numeric[col] = series
        original = frame[col]
        for i in range(len(frame)):
            value = series.iloc[i]
            raw = original.iloc[i]
            if pd.isna(value):
                if col in PRICE_COLUMNS or (not pd.isna(raw)):
                    report.add("null_or_non_numeric", "error", dates[i], col, f"Null or non-numeric {col} at row {i}")
            elif not np.isfinite(float(value)):
                report.add("non_finite", "error", dates[i], col, f"Non-finite {col} at row {i}")
            elif col in PRICE_COLUMNS and value <= 0:
                report.add("non_positive_price", "error", dates[i], col, f"Non-positive {col} at row {i}")
            elif col == "volume" and value < 0:
                report.add("negative_volume", "error", dates[i], col, f"Negative volume at row {i}")

    for i in range(len(frame)):
        row = {col: numeric[col].iloc[i] for col in PRICE_COLUMNS}
        if not all(pd.notna(row[c]) and np.isfinite(float(row[c])) and row[c] > 0 for c in ("low", "high")):
            continue
        low, high = row["low"], row["high"]
        if high < low:
            report.add("high_below_low", "error", dates[i], "high", f"high < low at row {i}")
        for col in ("open", "close"):
            if pd.notna(row[col]) and np.isfinite(float(row[col])) and row[col] > 0 and not low <= row[col] <= high:
                report.add("price_outside_range", "error", dates[i], col, f"{col} outside [low, high] at row {i}")

    # Dependent anomalies are intentionally not computed on corrupted inputs.
    from .anomalies import inspect_anomalies
    structural_bad = bool(bad_stamp.any() or not dtype_ok or stamps.duplicated().any() or not stamps.is_monotonic_increasing)
    if structural_bad:
        report.skip(*ANOMALY_CHECKS)
        return report
    inspect_anomalies(frame, stamps, numeric, asset, config, report)
    return report
