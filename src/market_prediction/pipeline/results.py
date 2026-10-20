"""Typed results for market-data ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from market_prediction.cache import CacheState
from market_prediction.validation import Status


class AssetOutcome(str, Enum):
    """Final outcome of one asset ingestion attempt."""

    SUCCESS = "success"
    SUCCESS_VIA_FALLBACK = "success_via_fallback"
    SKIPPED_FRESH = "skipped_fresh"
    REJECTED_BY_VALIDATION = "rejected_by_validation"
    FAILED_PROVIDER = "failed_provider"
    FAILED_CACHE = "failed_cache"
    SERVED_STALE = "served_stale"
    FAILED_UNEXPECTED = "failed_unexpected"


class FailureCategory(str, Enum):
    """Stable classification of asset-level ingestion failures."""

    PROVIDER = "provider"
    VALIDATION = "validation"
    CACHE = "cache"
    PRICE_BASIS = "price_basis"
    UNEXPECTED = "unexpected"


class CacheAction(str, Enum):
    """Action taken against the validated-data cache."""

    WRITTEN = "written"
    SKIPPED_FRESH = "skipped_fresh"
    NOT_WRITTEN = "not_written"


class RunStatus(str, Enum):
    """Overall status of a multi-asset ingestion run."""

    SUCCESS = "success"
    PARTIAL_FAILURE = "partial_failure"
    TOTAL_FAILURE = "total_failure"


OK_OUTCOMES = frozenset(
    {
        AssetOutcome.SUCCESS,
        AssetOutcome.SUCCESS_VIA_FALLBACK,
        AssetOutcome.SKIPPED_FRESH,
    }
)


@dataclass(frozen=True)
class AssetIngestionResult:
    """Final observable result for one configured asset."""

    asset_id: str
    ticker: str
    outcome: AssetOutcome
    provider_used: str | None
    fallback_used: bool
    rows_persisted: int
    validation_status: Status | None
    soft_anomaly_count: int
    cache_action: CacheAction
    data_freshness: CacheState
    failure_category: FailureCategory | None = None
    failure_reason: str | None = None

    @property
    def is_ok(self) -> bool:
        """Whether this asset completed without failure or degradation."""

        return self.outcome in OK_OUTCOMES


@dataclass(frozen=True)
class IngestionRunResult:
    """Result of one complete ingestion run."""

    assets: tuple[AssetIngestionResult, ...]
    started_at: datetime
    ended_at: datetime
    status: RunStatus

    @property
    def exit_code(self) -> int:
        """Return the CLI exit code represented by this run."""

        if self.status is RunStatus.SUCCESS:
            return 0

        if self.status is RunStatus.PARTIAL_FAILURE:
            return 1

        return 2


def derive_run_status(
    assets: tuple[AssetIngestionResult, ...],
) -> RunStatus:
    """Derive run status from final asset outcomes."""

    if not assets:
        return RunStatus.TOTAL_FAILURE

    successful = sum(result.is_ok for result in assets)

    if successful == len(assets):
        return RunStatus.SUCCESS

    if successful == 0:
        return RunStatus.TOTAL_FAILURE

    return RunStatus.PARTIAL_FAILURE