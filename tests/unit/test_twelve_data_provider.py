"""Offline unit tests for TwelveDataProvider."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from market_prediction.config import (
    AssetClass,
    AssetConfig,
    ProviderName,
)
from market_prediction.providers import (
    EmptyDataError,
    InvalidProviderDataError,
    MarketDataProvider,
    ProviderError,
    ProviderUnavailableError,
    UnsupportedAssetError,
)
from market_prediction.providers.base import OHLCV_COLUMNS
from market_prediction.providers.twelve_data_provider import (
    TwelveDataProvider,
)
from tests.contracts.market_data_provider_contract import (
    assert_market_data_provider_contract,
)


@pytest.fixture
def asset() -> AssetConfig:
    """Return the EUR/USD asset configuration."""
    return AssetConfig(
        display_name="EUR/USD",
        ticker="EURUSD=X",
        provider_symbols={
            ProviderName.YFINANCE: "EURUSD=X",
            ProviderName.TWELVE_DATA: "EUR/USD",
        },
        asset_class=AssetClass.FOREX,
        currency="USD",
        calendar="24/5",
        primary_provider=ProviderName.TWELVE_DATA,
        backup_provider=ProviderName.YFINANCE,
    )


@pytest.fixture
def observed_response() -> dict:
    """Return values captured during the Step 3g real-source probe."""
    return {
        "meta": {
            "symbol": "EUR/USD",
            "interval": "1day",
            "type": "Physical Currency",
        },
        "values": [
            {
                "datetime": "2026-10-19",
                "open": "1.11896",
                "high": "1.1259",
                "low": "1.11781",
                "close": "1.12113",
            },
            {
                "datetime": "2015-01-01",
                "open": "1.20986",
                "high": "1.20986",
                "low": "1.20986",
                "close": "1.20986",
            },
        ],
        "status": "ok",
    }


@pytest.fixture
def http_get() -> MagicMock:
    """Return an injected HTTP GET callable."""
    return MagicMock()


@pytest.fixture
def provider(http_get: MagicMock) -> TwelveDataProvider:
    """Return a provider whose HTTP and sleep behavior is deterministic."""
    return TwelveDataProvider(
        api_key="test-api-key",
        base_url="https://example.test",
        max_attempts=3,
        initial_delay_seconds=2.0,
        backoff_multiplier=2.0,
        http_get=http_get,
        sleep_fn=lambda _: None,
    )


def response_with_json(
    payload: dict,
    *,
    status_code: int = 200,
) -> MagicMock:
    """Build a minimal HTTP response fake."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_twelve_data_provider_implements_market_data_provider(
    provider: TwelveDataProvider,
) -> None:
    assert isinstance(provider, MarketDataProvider)
    assert provider.name == "twelve_data"


def test_successful_fetch_returns_exact_contract_shape(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    result = provider.fetch_daily(
        asset,
        date(2026, 10, 16),
        date(2026, 10, 19),
    )

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == OHLCV_COLUMNS
    assert len(result) == 2

    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique
    assert (result["source"] == "twelve_data").all()


def test_daily_dates_are_normalized_to_utc_midnight(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    result = provider.fetch_daily(
        asset,
        date(2026, 10, 16),
        date(2026, 10, 19),
    )

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2015-01-01T00:00:00Z"),
        pd.Timestamp("2026-10-19T00:00:00Z"),
    ]


def test_ohlc_values_are_normalized_to_numeric_columns(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    result = provider.fetch_daily(
        asset,
        date(2026, 10, 16),
        date(2026, 10, 19),
    )

    for column in ("open", "high", "low", "close"):
        assert pd.api.types.is_numeric_dtype(result[column])

    assert result.iloc[0]["open"] == pytest.approx(1.20986)
    assert result.iloc[0]["high"] == pytest.approx(1.20986)
    assert result.iloc[0]["low"] == pytest.approx(1.20986)
    assert result.iloc[0]["close"] == pytest.approx(1.20986)


def test_forex_has_no_fabricated_adjusted_close_or_volume(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    result = provider.fetch_daily(
        asset,
        date(2026, 10, 16),
        date(2026, 10, 19),
    )

    assert result["adj_close"].isna().all()
    assert result["volume"].isna().all()


def test_request_uses_provider_symbol_and_complete_date_range(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    provider.fetch_daily(
        asset,
        date(2015, 1, 1),
        date(2026, 10, 19),
    )

    _, kwargs = http_get.call_args

    assert kwargs["params"]["symbol"] == "EUR/USD"
    assert kwargs["params"]["interval"] == "1day"
    assert kwargs["params"]["start_date"] == "2015-01-01"
    assert kwargs["params"]["end_date"] == "2026-10-19"
    assert "outputsize" not in kwargs["params"]


def test_api_key_is_passed_to_request(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    provider.fetch_daily(
        asset,
        date(2026, 10, 16),
        date(2026, 10, 19),
    )

    _, kwargs = http_get.call_args

    assert kwargs["params"]["apikey"] == "test-api-key"


def test_invalid_date_range_fails_before_http_call(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    with pytest.raises(
        InvalidProviderDataError,
        match="start_date",
    ):
        provider.fetch_daily(
            asset,
            date(2026, 10, 20),
            date(2026, 10, 19),
        )

    http_get.assert_not_called()


def test_empty_values_raise_empty_data_error_without_retry(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.return_value = response_with_json(
        {
            "meta": {
                "symbol": "EUR/USD",
                "interval": "1day",
            },
            "values": [],
            "status": "ok",
        }
    )

    with pytest.raises(EmptyDataError):
        provider.fetch_daily(
            asset,
            date(2026, 10, 16),
            date(2026, 10, 19),
        )

    assert http_get.call_count == 1


def test_missing_values_field_is_invalid_provider_data(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.return_value = response_with_json(
        {
            "meta": {
                "symbol": "EUR/USD",
                "interval": "1day",
            },
            "status": "ok",
        }
    )

    with pytest.raises(InvalidProviderDataError):
        provider.fetch_daily(
            asset,
            date(2026, 10, 16),
            date(2026, 10, 19),
        )

    assert http_get.call_count == 1


def test_missing_ohlc_field_is_invalid_provider_data(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    broken = {
        **observed_response,
        "values": [
            {
                "datetime": "2026-10-19",
                "open": "1.16500",
                "high": "1.17200",
                "low": "1.16300",
            }
        ],
    }

    http_get.return_value = response_with_json(broken)

    with pytest.raises(
        InvalidProviderDataError,
        match="close",
    ):
        provider.fetch_daily(
            asset,
            date(2026, 10, 16),
            date(2026, 10, 19),
        )


def test_malformed_numeric_value_is_invalid_provider_data(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    broken = {
        **observed_response,
        "values": [
            {
                "datetime": "2026-10-19",
                "open": "not-a-number",
                "high": "1.17200",
                "low": "1.16300",
                "close": "1.17000",
            }
        ],
    }

    http_get.return_value = response_with_json(broken)

    with pytest.raises(InvalidProviderDataError):
        provider.fetch_daily(
            asset,
            date(2026, 10, 16),
            date(2026, 10, 19),
        )


def test_rate_limit_retries_then_succeeds(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    limited = response_with_json(
        {
            "status": "error",
            "code": 429,
            "message": "API credits exhausted",
        },
        status_code=429,
    )

    http_get.side_effect = [
        limited,
        response_with_json(observed_response),
    ]

    result = provider.fetch_daily(
        asset,
        date(2026, 10, 16),
        date(2026, 10, 19),
    )

    assert not result.empty
    assert http_get.call_count == 2


def test_rate_limit_exhaustion_maps_to_provider_unavailable(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.return_value = response_with_json(
        {
            "status": "error",
            "code": 429,
            "message": "API credits exhausted",
        },
        status_code=429,
    )

    with pytest.raises(ProviderUnavailableError):
        provider.fetch_daily(
            asset,
            date(2026, 10, 16),
            date(2026, 10, 19),
        )

    assert http_get.call_count == 3


def test_timeout_retries_then_succeeds(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.side_effect = [
        TimeoutError("temporary timeout"),
        response_with_json(observed_response),
    ]

    result = provider.fetch_daily(
        asset,
        date(2026, 10, 16),
        date(2026, 10, 19),
    )

    assert not result.empty
    assert http_get.call_count == 2


def test_timeout_exhaustion_maps_to_provider_unavailable(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.side_effect = TimeoutError("provider timed out")

    with pytest.raises(ProviderUnavailableError):
        provider.fetch_daily(
            asset,
            date(2026, 10, 16),
            date(2026, 10, 19),
        )

    assert http_get.call_count == 3


def test_symbol_not_found_maps_to_unsupported_asset(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.return_value = response_with_json(
        {
            "status": "error",
            "code": 404,
            "message": "symbol not found",
        },
        status_code=404,
    )

    with pytest.raises(UnsupportedAssetError):
        provider.fetch_daily(
            asset,
            date(2026, 10, 16),
            date(2026, 10, 19),
        )

    assert http_get.call_count == 1


def test_unknown_exception_never_leaks_raw(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.side_effect = RuntimeError(
        "unexpected HTTP client failure"
    )

    with pytest.raises(ProviderError) as exc_info:
        provider.fetch_daily(
            asset,
            date(2026, 10, 16),
            date(2026, 10, 19),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_twelve_data_provider_satisfies_shared_contract(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    result = provider.fetch_daily(
        asset,
        date(2026, 10, 16),
        date(2026, 10, 19),
    )

    assert_market_data_provider_contract(
        provider,
        result,
    )

def test_conflicting_real_observed_duplicate_daily_bar_is_rejected(
    forex_asset,
):
    """Regression: reject conflicting duplicate Twelve Data daily bars."""

    payload = {
        "status": "ok",
        "values": [
            {
                "datetime": "2026-10-04",
                "open": "1.1255",
                "high": "1.1262",
                "low": "1.12514",
                "close": "1.12578",
            },
            {
                "datetime": "2026-10-04",
                "open": "1.1255",
                "high": "1.1262",
                "low": "1.12514",
                "close": "1.1258",
            },
        ],
    }

    response = FakeResponse(
        status_code=200,
        payload=payload,
    )

    provider = TwelveDataProvider(
        api_key="test-key",
        base_url="https://api.twelvedata.com",
        http_get=lambda *args, **kwargs: response,
    )

    with pytest.raises(
        InvalidProviderDataError,
        match="duplicate timestamps",
    ):
        provider.fetch_daily(
            asset=forex_asset,
            start_date=date(2026, 10, 4),
            end_date=date(2026, 10, 4),
        )

def test_conflicting_real_observed_duplicate_daily_bar_is_rejected(
    provider: TwelveDataProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    """Reject conflicting duplicate bars observed in Twelve Data EUR/USD."""

    payload = {
        "meta": {
            "symbol": "EUR/USD",
            "interval": "1day",
            "type": "Physical Currency",
        },
        "values": [
            {
                "datetime": "2026-10-04",
                "open": "1.1255",
                "high": "1.1262",
                "low": "1.12514",
                "close": "1.12578",
            },
            {
                "datetime": "2026-10-04",
                "open": "1.1255",
                "high": "1.1262",
                "low": "1.12514",
                "close": "1.1258",
            },
        ],
        "status": "ok",
    }

    http_get.return_value = response_with_json(payload)

    with pytest.raises(
        InvalidProviderDataError,
        match="duplicate timestamps",
    ):
        provider.fetch_daily(
            asset,
            date(2026, 10, 4),
            date(2026, 10, 4),
        )

    assert http_get.call_count == 1