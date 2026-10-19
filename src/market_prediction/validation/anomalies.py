"""Asset-aware, non-blocking OHLCV anomaly checks."""
from __future__ import annotations
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from market_prediction.config import AssetConfig, PipelineConfig
from .schema import ValidationReport


def _expected_sessions(start: pd.Timestamp, end: pd.Timestamp, calendar: str, closed_dates: list) -> pd.DatetimeIndex:
    first, last = start.date(), end.date()
    if calendar == "24/7":
        sessions = pd.date_range(first, last, freq="D")
    elif calendar == "24/5":
        sessions = pd.date_range(first, last, freq="B")
    else:
        sessions = xcals.get_calendar(calendar).sessions_in_range(first, last).tz_localize(None)
    closures = set(closed_dates)
    return pd.DatetimeIndex([day for day in sessions if day.date() not in closures])


def inspect_anomalies(frame: pd.DataFrame, stamps: pd.Series, numeric: dict[str, pd.Series], asset: AssetConfig, config: PipelineConfig, report: ValidationReport) -> None:
    """Accumulate independent warnings, skipping only checks with unsafe inputs."""
    rules = getattr(config.validation.asset_classes, asset.asset_class.value)
    dates = stamps.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    close = numeric["close"].astype(float)
    if (close.isna() | ~np.isfinite(close) | (close <= 0)).any():
        report.skip("large_daily_move")
    else:
        returns = close.pct_change(fill_method=None).abs()
        for i in range(1, len(frame)):
            if returns.iloc[i] > rules.large_daily_move_threshold:
                extra = (
                    " Possible corporate action or source-data restatement; "
                    "Yahoo OHLC prices are split-adjusted even when auto_adjust=False. "
                    "Verify before interpreting as a market move."
                    if asset.asset_class.value == "equity"
                    else ""
                )
                report.add("large_daily_move", "warning", dates.iloc[i].date(), "close", f"Absolute close-to-close move {returns.iloc[i]:.2%} exceeds {rules.large_daily_move_threshold:.2%}.{extra}")

    volume = numeric["volume"].astype(float)
    volume_bad = volume.notna() & (~np.isfinite(volume) | (volume < 0))

    if volume_bad.any():
        report.skip("volume_spike", "zero_volume")
    else:
        if rules.check_zero_volume:
            for i in range(len(frame)):
                if pd.notna(volume.iloc[i]) and volume.iloc[i] == 0:
                    report.add(
                        "zero_volume",
                        "warning",
                        dates.iloc[i].date(),
                        "volume",
                        "Zero reported volume",
                    )

        window = rules.volume_rolling_window
        for i in range(window, len(frame)):
            history = volume.iloc[i - window:i]
            baseline = history.median()
            current = volume.iloc[i]

            if (
                    pd.notna(current)
                    and pd.notna(baseline)
                    and baseline > 0
                    and current > baseline * rules.volume_spike_multiplier
            ):
                report.add(
                    "volume_spike",
                    "warning",
                    dates.iloc[i].date(),
                    "volume",
                    (
                        f"Volume {current:g} exceeds "
                        f"{rules.volume_spike_multiplier:g}x trailing "
                        f"{window}-session median {baseline:g}"
                    ),
                )
    try:
        expected = _expected_sessions(dates.iloc[0], dates.iloc[-1], rules.calendar, rules.closed_dates)
    except (ValueError, KeyError, OverflowError) as exc:
        report.add("calendar_unavailable", "error", None, "timestamp", f"Cannot calculate expected {rules.calendar} sessions: {exc}")
        report.skip("missing_sessions")
        return
    actual = set(dates)
    # max_missing_sessions is MAX CONSECUTIVE missing expected sessions, not total.
    streak: list[pd.Timestamp] = []
    for session in expected:
        if session not in actual:
            streak.append(session)
        else:
            if len(streak) > rules.max_missing_sessions:
                report.add("missing_sessions", "warning", streak[0].date(), "timestamp", f"{len(streak)} consecutive expected {rules.calendar} sessions missing: {streak[0].date()} through {streak[-1].date()}")
            streak = []
    if len(streak) > rules.max_missing_sessions:
        report.add("missing_sessions", "warning", streak[0].date(), "timestamp", f"{len(streak)} consecutive expected {rules.calendar} sessions missing: {streak[0].date()} through {streak[-1].date()}")
