"""Market-data ingestion orchestration.

This module connects the existing provider, validation, and cache layers.

Responsibilities:
- honor fresh-cache short-circuiting;
- select configured primary and backup providers;
- isolate failures per asset;
- validate every newly acquired dataset;
- enforce the ValidatedDataset cache gate;
- prevent incompatible price bases from replacing cached history;
- return typed per-asset and per-run results.

Provider-specific retry behavior remains inside provider implementations.
This layer does not perform feature engineering or modeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Mapping, Sequence

from market_prediction.cache import (
    CacheManager,
    CacheReadResult,
    CacheState,
    ColumnPriceBasis,
    ValidatedDataset,
    ValidationGateError,
)

from market_prediction.config import (
    AssetConfig,
    AssetsConfig,
    PipelineConfig,
    ProviderName,
)
from market_prediction.pipeline.results import (
    AssetIngestionResult,
    AssetOutcome,
    CacheAction,
    FailureCategory,
    IngestionRunResult,
    derive_run_status,
)
from market_prediction.providers import (
    MarketDataProvider,
    ProviderError,
)
from market_prediction.validation import (
    Status,
    ValidationReport,
)


logger = logging.getLogger(__name__)


Validator = Callable[
    [object, AssetConfig, PipelineConfig],
    ValidationReport,
]


@dataclass(frozen=True)
class ProviderRegistration:
    """Provider plus the price semantics of data it returns."""

    provider: MarketDataProvider
    price_adjustment_mode: str
    price_basis: Mapping[str, ColumnPriceBasis]

    def __post_init__(self) -> None:
        if not self.price_adjustment_mode.strip():
            raise ValueError(
                "price_adjustment_mode must not be empty"
            )

        if not self.price_basis:
            raise ValueError(
                "price_basis must not be empty"
            )


class IngestionOrchestrator:
    """Run market-data ingestion for configured assets."""

    def __init__(
        self,
        *,
        assets_config: AssetsConfig,
        pipeline_config: PipelineConfig,
        providers: Mapping[
            ProviderName,
            ProviderRegistration,
        ],
        validator: Validator,
        cache_manager: CacheManager,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._assets_config = assets_config
        self._pipeline_config = pipeline_config
        self._providers = dict(providers)
        self._validator = validator
        self._cache_manager = cache_manager
        self._now = now or (
            lambda: datetime.now(timezone.utc)
        )

    def run(
        self,
        *,
        asset_ids: Sequence[str] | None = None,
        force_refresh: bool = False,
    ) -> IngestionRunResult:
        """Run ingestion sequentially for selected configured assets."""

        started_at = self._utc_now()

        selected = self._select_assets(asset_ids)

        results: list[AssetIngestionResult] = []

        for asset_id, asset in selected:
            try:
                result = self._run_asset(
                    asset_id=asset_id,
                    asset=asset,
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                logger.exception(
                    "asset=%s ticker=%s outcome=%s reason=%s",
                    asset_id,
                    asset.ticker,
                    AssetOutcome.FAILED_UNEXPECTED.value,
                    exc,
                )

                result = AssetIngestionResult(
                    asset_id=asset_id,
                    ticker=asset.ticker,
                    outcome=AssetOutcome.FAILED_UNEXPECTED,
                    provider_used=None,
                    fallback_used=False,
                    rows_persisted=0,
                    validation_status=None,
                    soft_anomaly_count=0,
                    cache_action=CacheAction.NOT_WRITTEN,
                    data_freshness=self._safe_cache_state(
                        asset.ticker
                    ),
                    failure_category=FailureCategory.UNEXPECTED,
                    failure_reason=str(exc),
                )

            results.append(result)

            logger.info(
                "asset=%s ticker=%s outcome=%s "
                "provider=%s fallback=%s cache=%s reason=%s",
                result.asset_id,
                result.ticker,
                result.outcome.value,
                result.provider_used or "-",
                result.fallback_used,
                result.cache_action.value,
                result.failure_reason or "-",
            )

        ended_at = self._utc_now()
        frozen_results = tuple(results)

        return IngestionRunResult(
            assets=frozen_results,
            started_at=started_at,
            ended_at=ended_at,
            status=derive_run_status(frozen_results),
        )

    def _run_asset(
        self,
        *,
        asset_id: str,
        asset: AssetConfig,
        force_refresh: bool,
    ) -> AssetIngestionResult:
        cache_before = self._cache_manager.read(
            asset.ticker
        )

        if (
            cache_before.state is CacheState.FRESH
            and not force_refresh
        ):
            metadata = cache_before.metadata

            return AssetIngestionResult(
                asset_id=asset_id,
                ticker=asset.ticker,
                outcome=AssetOutcome.SKIPPED_FRESH,
                provider_used=(
                    metadata.provider
                    if metadata is not None
                    else None
                ),
                fallback_used=False,
                rows_persisted=0,
                validation_status=(
                    metadata.validation_status
                    if metadata is not None
                    else None
                ),
                soft_anomaly_count=(
                    self._cached_soft_anomaly_count(
                        cache_before
                    )
                ),
                cache_action=CacheAction.SKIPPED_FRESH,
                data_freshness=CacheState.FRESH,
            )

        primary = self._providers.get(
            asset.primary_provider
        )

        if primary is None:
            return self._try_backup(
                asset_id=asset_id,
                asset=asset,
                cache_before=cache_before,
                primary_error=ProviderError(
                    "Primary provider "
                    f"'{asset.primary_provider.value}' "
                    "is not registered."
                ),
            )

        try:
            frame = primary.provider.fetch_daily(
                asset,
                self._pipeline_config.history.start_date,
                self._today(),
            )

        except ProviderError as primary_error:
            return self._try_backup(
                asset_id=asset_id,
                asset=asset,
                cache_before=cache_before,
                primary_error=primary_error,
            )

        return self._validate_and_cache(
            asset_id=asset_id,
            asset=asset,
            registration=primary,
            frame=frame,
            cache_before=cache_before,
            fallback_used=False,
        )

    def _try_backup(
        self,
        *,
        asset_id: str,
        asset: AssetConfig,
        cache_before: CacheReadResult,
        primary_error: ProviderError,
    ) -> AssetIngestionResult:
        backup = self._providers.get(
            asset.backup_provider
        )

        if backup is None:
            return self._provider_failure_result(
                asset_id=asset_id,
                asset=asset,
                cache_before=cache_before,
                reason=(
                    "Primary provider "
                    f"'{asset.primary_provider.value}' failed: "
                    f"{primary_error}. Backup provider "
                    f"'{asset.backup_provider.value}' "
                    "is not registered."
                ),
            )

        try:
            frame = backup.provider.fetch_daily(
                asset,
                self._pipeline_config.history.start_date,
                self._today(),
            )
        except ProviderError as backup_error:
            return self._provider_failure_result(
                asset_id=asset_id,
                asset=asset,
                cache_before=cache_before,
                reason=(
                    "Primary provider "
                    f"'{asset.primary_provider.value}' failed: "
                    f"{primary_error}. Backup provider "
                    f"'{asset.backup_provider.value}' failed: "
                    f"{backup_error}."
                ),
            )

        return self._validate_and_cache(
            asset_id=asset_id,
            asset=asset,
            registration=backup,
            frame=frame,
            cache_before=cache_before,
            fallback_used=True,
        )

    def _validate_and_cache(
        self,
        *,
        asset_id: str,
        asset: AssetConfig,
        registration: ProviderRegistration,
        frame: object,
        cache_before: CacheReadResult,
        fallback_used: bool,
    ) -> AssetIngestionResult:
        report = self._validator(
            frame,
            asset,
            self._pipeline_config,
        )

        if report.status is not Status.PASSED:
            return AssetIngestionResult(
                asset_id=asset_id,
                ticker=asset.ticker,
                outcome=AssetOutcome.REJECTED_BY_VALIDATION,
                provider_used=registration.provider.name,
                fallback_used=fallback_used,
                rows_persisted=0,
                validation_status=report.status,
                soft_anomaly_count=len(
                    report.soft_anomalies
                ),
                cache_action=CacheAction.NOT_WRITTEN,
                data_freshness=cache_before.state,
                failure_category=FailureCategory.VALIDATION,
                failure_reason=self._validation_reason(
                    report
                ),
            )

        dataset = ValidatedDataset.from_validation(
            frame,
            report,
        )

        basis_problem = self._price_basis_problem(
            cache_before=cache_before,
            registration=registration,
        )

        if basis_problem is not None:
            return AssetIngestionResult(
                asset_id=asset_id,
                ticker=asset.ticker,
                outcome=AssetOutcome.FAILED_CACHE,
                provider_used=registration.provider.name,
                fallback_used=fallback_used,
                rows_persisted=0,
                validation_status=report.status,
                soft_anomaly_count=len(
                    report.soft_anomalies
                ),
                cache_action=CacheAction.NOT_WRITTEN,
                data_freshness=cache_before.state,
                failure_category=FailureCategory.PRICE_BASIS,
                failure_reason=basis_problem,
            )

        try:
            metadata = self._cache_manager.write(
                dataset,
                provider=registration.provider.name,
                fetch_timestamp=self._utc_now(),
                price_adjustment_mode=(
                    registration.price_adjustment_mode
                ),
                price_basis=registration.price_basis,
            )
        except (OSError, ValidationGateError, ValueError) as exc:
            return AssetIngestionResult(
                asset_id=asset_id,
                ticker=asset.ticker,
                outcome=AssetOutcome.FAILED_CACHE,
                provider_used=registration.provider.name,
                fallback_used=fallback_used,
                rows_persisted=0,
                validation_status=report.status,
                soft_anomaly_count=len(
                    report.soft_anomalies
                ),
                cache_action=CacheAction.NOT_WRITTEN,
                data_freshness=self._safe_cache_state(
                    asset.ticker
                ),
                failure_category=FailureCategory.CACHE,
                failure_reason=str(exc),
            )

        return AssetIngestionResult(
            asset_id=asset_id,
            ticker=asset.ticker,
            outcome=(
                AssetOutcome.SUCCESS_VIA_FALLBACK
                if fallback_used
                else AssetOutcome.SUCCESS
            ),
            provider_used=registration.provider.name,
            fallback_used=fallback_used,
            rows_persisted=metadata.row_count,
            validation_status=report.status,
            soft_anomaly_count=len(
                report.soft_anomalies
            ),
            cache_action=CacheAction.WRITTEN,
            data_freshness=CacheState.FRESH,
        )

    def _provider_failure_result(
        self,
        *,
        asset_id: str,
        asset: AssetConfig,
        cache_before: CacheReadResult,
        reason: str,
    ) -> AssetIngestionResult:
        if cache_before.state is CacheState.STALE:
            metadata = cache_before.metadata

            return AssetIngestionResult(
                asset_id=asset_id,
                ticker=asset.ticker,
                outcome=AssetOutcome.SERVED_STALE,
                provider_used=(
                    metadata.provider
                    if metadata is not None
                    else None
                ),
                fallback_used=False,
                rows_persisted=0,
                validation_status=(
                    metadata.validation_status
                    if metadata is not None
                    else None
                ),
                soft_anomaly_count=(
                    self._cached_soft_anomaly_count(
                        cache_before
                    )
                ),
                cache_action=CacheAction.NOT_WRITTEN,
                data_freshness=CacheState.STALE,
                failure_category=FailureCategory.PROVIDER,
                failure_reason=reason,
            )

        return AssetIngestionResult(
            asset_id=asset_id,
            ticker=asset.ticker,
            outcome=AssetOutcome.FAILED_PROVIDER,
            provider_used=None,
            fallback_used=False,
            rows_persisted=0,
            validation_status=None,
            soft_anomaly_count=0,
            cache_action=CacheAction.NOT_WRITTEN,
            data_freshness=cache_before.state,
            failure_category=FailureCategory.PROVIDER,
            failure_reason=reason,
        )

    def _price_basis_problem(
        self,
        *,
        cache_before: CacheReadResult,
        registration: ProviderRegistration,
    ) -> str | None:
        if cache_before.state not in {
            CacheState.FRESH,
            CacheState.STALE,
        }:
            return None

        metadata = cache_before.metadata

        if metadata is None:
            return (
                "Usable cache has no metadata; refusing "
                "replacement."
            )

        if (
            metadata.price_adjustment_mode
            != registration.price_adjustment_mode
        ):
            return (
                "Price adjustment mode mismatch: cached "
                f"'{metadata.price_adjustment_mode}', incoming "
                f"'{registration.price_adjustment_mode}'."
            )

        if dict(metadata.price_basis) != dict(
            registration.price_basis
        ):
            return (
                "Price basis mismatch between cached history "
                f"and provider '{registration.provider.name}'."
            )

        return None

    def _select_assets(
        self,
        asset_ids: Sequence[str] | None,
    ) -> list[tuple[str, AssetConfig]]:
        configured = self._assets_config.assets

        if asset_ids is None:
            return list(configured.items())

        requested = list(asset_ids)

        unknown = [
            asset_id
            for asset_id in requested
            if asset_id not in configured
        ]

        if unknown:
            names = ", ".join(unknown)
            raise ValueError(
                f"Unknown configured asset id(s): {names}"
            )

        return [
            (asset_id, configured[asset_id])
            for asset_id in requested
        ]

    def _validation_reason(
        self,
        report: ValidationReport,
    ) -> str:
        hard_count = len(report.hard_failures)
        skipped_count = len(report.skipped_checks)

        return (
            "Validation blocked dataset: "
            f"{hard_count} hard failure(s), "
            f"{skipped_count} skipped check(s)."
        )

    def _cached_soft_anomaly_count(
        self,
        cache_result: CacheReadResult,
    ) -> int:
        metadata = cache_result.metadata

        if metadata is None:
            return 0

        return sum(
            summary.count
            for summary in metadata.soft_anomalies
        )

    def _safe_cache_state(
        self,
        asset: str,
    ) -> CacheState:
        try:
            return self._cache_manager.read(asset).state
        except Exception:
            return CacheState.CORRUPTED

    def _utc_now(self) -> datetime:
        value = self._now()

        if value.tzinfo is None:
            raise ValueError(
                "Orchestrator clock must return a "
                "timezone-aware datetime."
            )

        return value.astimezone(timezone.utc)

    def _today(self) -> date:
        return self._utc_now().date()