"""Contracts for ingestion result models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from market_prediction.cache import CacheState
from market_prediction.pipeline.results import (
    AssetIngestionResult,
    AssetOutcome,
    CacheAction,
    FailureCategory,
    IngestionRunResult,
    RunStatus,
    derive_run_status,
)
from market_prediction.validation import Status


NOW = datetime(
    2026,
    10,
    20,
    12,
    0,
    tzinfo=timezone.utc,
)


def asset_result(
    outcome: AssetOutcome,
) -> AssetIngestionResult:
    return AssetIngestionResult(
        asset_id="aapl",
        ticker="AAPL",
        outcome=outcome,
        provider_used="yfinance",
        fallback_used=(
            outcome
            is AssetOutcome.SUCCESS_VIA_FALLBACK
        ),
        rows_persisted=10 if outcome in {
            AssetOutcome.SUCCESS,
            AssetOutcome.SUCCESS_VIA_FALLBACK,
        } else 0,
        validation_status=(
            Status.PASSED
            if outcome in {
                AssetOutcome.SUCCESS,
                AssetOutcome.SUCCESS_VIA_FALLBACK,
            }
            else None
        ),
        soft_anomaly_count=0,
        cache_action=(
            CacheAction.WRITTEN
            if outcome in {
                AssetOutcome.SUCCESS,
                AssetOutcome.SUCCESS_VIA_FALLBACK,
            }
            else CacheAction.NOT_WRITTEN
        ),
        data_freshness=CacheState.FRESH,
    )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (AssetOutcome.SUCCESS, True),
        (AssetOutcome.SUCCESS_VIA_FALLBACK, True),
        (AssetOutcome.SKIPPED_FRESH, True),
        (AssetOutcome.REJECTED_BY_VALIDATION, False),
        (AssetOutcome.FAILED_PROVIDER, False),
        (AssetOutcome.FAILED_CACHE, False),
        (AssetOutcome.SERVED_STALE, False),
        (AssetOutcome.FAILED_UNEXPECTED, False),
    ],
)
def test_asset_outcome_ok_semantics(
    outcome,
    expected,
):
    assert asset_result(outcome).is_ok is expected


def test_all_ok_assets_produce_success():
    assets = (
        asset_result(AssetOutcome.SUCCESS),
        asset_result(AssetOutcome.SKIPPED_FRESH),
        asset_result(AssetOutcome.SUCCESS_VIA_FALLBACK),
    )

    assert derive_run_status(assets) is RunStatus.SUCCESS


def test_mixed_results_produce_partial_failure():
    assets = (
        asset_result(AssetOutcome.SUCCESS),
        asset_result(AssetOutcome.FAILED_PROVIDER),
    )

    assert derive_run_status(assets) is RunStatus.PARTIAL_FAILURE


def test_served_stale_is_degraded_not_success():
    assets = (
        asset_result(AssetOutcome.SUCCESS),
        asset_result(AssetOutcome.SERVED_STALE),
    )

    assert derive_run_status(assets) is RunStatus.PARTIAL_FAILURE


def test_all_failed_assets_produce_total_failure():
    assets = (
        asset_result(AssetOutcome.FAILED_PROVIDER),
        asset_result(AssetOutcome.SERVED_STALE),
    )

    assert derive_run_status(assets) is RunStatus.TOTAL_FAILURE


def test_empty_run_is_total_failure():
    assert derive_run_status(()) is RunStatus.TOTAL_FAILURE


@pytest.mark.parametrize(
    ("status", "expected_exit_code"),
    [
        (RunStatus.SUCCESS, 0),
        (RunStatus.PARTIAL_FAILURE, 1),
        (RunStatus.TOTAL_FAILURE, 2),
    ],
)
def test_run_status_has_stable_exit_code(
    status,
    expected_exit_code,
):
    run = IngestionRunResult(
        assets=(),
        started_at=NOW,
        ended_at=NOW,
        status=status,
    )

    assert run.exit_code == expected_exit_code


def test_failure_result_can_record_category_and_reason():
    result = AssetIngestionResult(
        asset_id="aapl",
        ticker="AAPL",
        outcome=AssetOutcome.FAILED_PROVIDER,
        provider_used=None,
        fallback_used=False,
        rows_persisted=0,
        validation_status=None,
        soft_anomaly_count=0,
        cache_action=CacheAction.NOT_WRITTEN,
        data_freshness=CacheState.MISSING,
        failure_category=FailureCategory.PROVIDER,
        failure_reason="all configured providers failed",
    )

    assert result.failure_category is FailureCategory.PROVIDER
    assert result.failure_reason == (
        "all configured providers failed"
    )