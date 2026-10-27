"""Domain models and safety contracts for validated market-data caching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from market_prediction.validation import Status, ValidationReport
from market_prediction.validation.fingerprint import dataframe_fingerprint


class ValidationGateError(ValueError):
    """Raised when data that did not pass validation enters the cache gate."""


class CacheState(str, Enum):
    """Observable state of one asset's cache."""

    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"
    CORRUPTED = "corrupted"


class PriceAdjustment(str, Enum):
    """Adjustment semantics of a cached market-data column."""

    SPLIT_ADJUSTED = "split_adjusted"
    SPLIT_AND_DIVIDEND_ADJUSTED = "split_and_dividend_adjusted"
    SPLIT_CONSISTENT_VOLUME = "split_consistent_volume"
    NOT_APPLICABLE = "not_applicable"


class HistoricalStability(str, Enum):
    """Whether a provider may later restate historical values."""

    MAY_BE_RESTATED = "may_be_restated"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class ColumnPriceBasis:
    """Price-basis metadata for one cached market-data column."""

    adjustment: PriceAdjustment
    historical_stability: HistoricalStability


YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS: Mapping[
    str,
    ColumnPriceBasis,
] = MappingProxyType(
    {
        "open": ColumnPriceBasis(
            adjustment=PriceAdjustment.SPLIT_ADJUSTED,
            historical_stability=HistoricalStability.MAY_BE_RESTATED,
        ),
        "high": ColumnPriceBasis(
            adjustment=PriceAdjustment.SPLIT_ADJUSTED,
            historical_stability=HistoricalStability.MAY_BE_RESTATED,
        ),
        "low": ColumnPriceBasis(
            adjustment=PriceAdjustment.SPLIT_ADJUSTED,
            historical_stability=HistoricalStability.MAY_BE_RESTATED,
        ),
        "close": ColumnPriceBasis(
            adjustment=PriceAdjustment.SPLIT_ADJUSTED,
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
)


class ValidatedDataset:
    """A DataFrame that has successfully crossed the validation gate."""

    __slots__ = ("_frame", "_report")

    _CREATION_TOKEN = object()

    def __init__(
        self,
        frame: pd.DataFrame,
        report: ValidationReport,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not self._CREATION_TOKEN:
            raise ValidationGateError(
                "ValidatedDataset cannot be constructed directly; "
                "use ValidatedDataset.from_validation()."
            )

        object.__setattr__(
            self,
            "_frame",
            frame.copy(deep=True),
        )
        object.__setattr__(
            self,
            "_report",
            report,
        )

    @classmethod
    def from_validation(
        cls,
        frame: pd.DataFrame,
        report: ValidationReport,
    ) -> "ValidatedDataset":
        """Create a cache-eligible dataset from a PASSED validation report."""

        if report.status is not Status.PASSED:
            raise ValidationGateError(
                f"Dataset for '{report.asset}' is not cache-eligible: "
                f"validation status is {report.status.value}."
            )

        if report.row_count != len(frame):
            raise ValidationGateError(
                f"Validation report row count ({report.row_count}) does not "
                f"match dataset row count ({len(frame)})."
            )

        if report.dataset_fingerprint is None:
            raise ValidationGateError(
                f"Validation report for '{report.asset}' "
                "has no dataset fingerprint."
            )

        actual_fingerprint = dataframe_fingerprint(frame)

        if report.dataset_fingerprint != actual_fingerprint:
            raise ValidationGateError(
                f"Dataset for '{report.asset}' does not match the data "
                "that produced its validation report."
            )

        return cls(
            frame,
            report,
            _token=cls._CREATION_TOKEN,
        )

    @property
    def data(self) -> pd.DataFrame:
        """Return a defensive copy of the validated market data."""

        return self._frame.copy(deep=True)

    @property
    def report(self) -> ValidationReport:
        """Return the report that admitted this dataset through the gate."""

        return self._report

    @property
    def dataset_id(self) -> str:
        """Return the validated content identity of this dataset."""

        fingerprint = self._report.dataset_fingerprint

        if fingerprint is None:
            raise RuntimeError(
                "ValidatedDataset unexpectedly has no dataset fingerprint."
            )

        return fingerprint


@dataclass(frozen=True)
class SoftAnomalySummary:
    """Compact persisted summary of one soft-anomaly rule."""

    rule: str
    count: int


@dataclass(frozen=True)
class RestatementSummary:
    """Comparison of fresh history with the previously cached overlap."""

    detected: bool
    overlap_start: date
    overlap_end: date
    changed_rows: int
    changed_columns: tuple[str, ...]


@dataclass(frozen=True)
class CacheMetadata:
    """Persisted metadata describing one validated cached dataset."""

    asset: str
    provider: str
    fetch_timestamp: datetime
    date_start: date
    date_end: date
    row_count: int
    price_adjustment_mode: str
    validation_status: Status
    soft_anomalies: tuple[SoftAnomalySummary, ...]
    price_basis: Mapping[str, ColumnPriceBasis]
    dataset_id: str
    restatement: RestatementSummary | None = None
    bar_convention: str | None = None


@dataclass(frozen=True)
class CacheReadResult:
    """Single predictable return type for all cache-read outcomes."""

    state: CacheState
    data: pd.DataFrame | None
    metadata: CacheMetadata | None
    error: str | None = None

    @property
    def exists(self) -> bool:
        """Whether a usable cached dataset was successfully read."""

        return self.state in {
            CacheState.FRESH,
            CacheState.STALE,
        }

    @property
    def is_fresh(self) -> bool:
        """Whether the cache exists and is inside its configured TTL."""

        return self.state is CacheState.FRESH