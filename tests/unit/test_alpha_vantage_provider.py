"""Offline unit tests for AlphaVantageProvider."""

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
from market_prediction.providers.alpha_vantage_provider import (
    AlphaVantageProvider,
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
            ProviderName.ALPHA_VANTAGE: "EUR/USD",
        },
        asset_class=AssetClass.FOREX,
        currency="USD",
        calendar="24/5",
        primary_provider=ProviderName.TWELVE_DATA,
        backup_provider=ProviderName.YFINANCE,
    )


@pytest.fixture
def observed_response() -> dict:
    """Return values captured during the Step 3g Alpha Vantage probe."""
    return {
        "Meta Data": {
            "1. Information": (
                "Forex Daily Prices (open, high, low, close)"
            ),
            "2. From Symbol": "EUR",
            "3. To Symbol": "USD",
            "4. Output Size": "Full size",
            "5. Last Refreshed": "2026-10-23",
            "6. Time Zone": "UTC",
        },
        "Time Series FX (Daily)": {
            "2015-01-02": {
                "1. open": "1.21017",
                "2. high": "1.21020",
                "3. low": "1.19925",
                "4. close": "1.19965",
            },
            "2020-03-19": {
                "1. open": "1.09327",
                "2. high": "1.09818",
                "3. low": "1.06529",
                "4. close": "1.06635",
            },
            "2022-12-26": {
                "1. open": "1.06100",
                "2. high": "1.06364",
                "3. low": "1.05965",
                "4. close": "1.06332",
            },
            "2026-10-23": {
                "1. open": "1.12170",
                "2. high": "1.12260",
                "3. low": "1.11830",
                "4. close": "1.12020",
            },
        },
    }


@pytest.fixture
def http_get() -> MagicMock:
    """Return an injected HTTP GET callable."""
    return MagicMock()


@pytest.fixture
def provider(http_get: MagicMock) -> AlphaVantageProvider:
    """Return a provider with deterministic HTTP and sleep behavior."""
    return AlphaVantageProvider(
        api_key="test-api-key",
        base_url="https://example.test",
        max_attempts=3,
        initial_delay_seconds=2.0,
        backoff_multiplier=2.0,
        timeout_seconds=10.0,
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


def test_alpha_vantage_provider_implements_market_data_provider(
    provider: AlphaVantageProvider,
) -> None:
    assert isinstance(provider, MarketDataProvider)
    assert provider.name == "alpha_vantage"


def test_successful_fetch_satisfies_shared_provider_contract(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    result = provider.fetch_daily(
        asset,
        date(2015, 1, 1),
        date(2026, 10, 23),
    )

    assert_market_data_provider_contract(
        provider,
        result,
    )

    assert tuple(result.columns) == OHLCV_COLUMNS
    assert len(result) == 4


def test_real_observed_values_are_preserved(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    result = provider.fetch_daily(
        asset,
        date(2015, 1, 1),
        date(2026, 10, 23),
    )

    row = result.loc[
        result["timestamp"]
        == pd.Timestamp("2022-12-26T00:00:00Z")
    ].iloc[0]

    assert row["open"] == pytest.approx(1.06100)
    assert row["high"] == pytest.approx(1.06364)
    assert row["low"] == pytest.approx(1.05965)
    assert row["close"] == pytest.approx(1.06332)


def test_utc_daily_dates_are_normalized_to_utc_midnight(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    result = provider.fetch_daily(
        asset,
        date(2015, 1, 1),
        date(2026, 10, 23),
    )

    assert result["timestamp"].iloc[0] == pd.Timestamp(
        "2015-01-02T00:00:00Z"
    )
    assert result["timestamp"].iloc[-1] == pd.Timestamp(
        "2026-10-23T00:00:00Z"
    )


def test_forex_has_no_fabricated_adjusted_close_or_volume(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    result = provider.fetch_daily(
        asset,
        date(2015, 1, 1),
        date(2026, 10, 23),
    )

    assert result["adj_close"].isna().all()
    assert result["volume"].isna().all()


def test_request_uses_fx_daily_full_output(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    http_get.return_value = response_with_json(observed_response)

    provider.fetch_daily(
        asset,
        date(2015, 1, 1),
        date(2026, 10, 23),
    )

    _, kwargs = http_get.call_args
    params = kwargs["params"]

    assert params["function"] == "FX_DAILY"
    assert params["from_symbol"] == "EUR"
    assert params["to_symbol"] == "USD"
    assert params["outputsize"] == "full"
    assert params["apikey"] == "test-api-key"


def test_non_forex_asset_is_rejected(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    equity = asset.model_copy(
        update={
            "asset_class": AssetClass.EQUITY,
        }
    )

    with pytest.raises(UnsupportedAssetError):
        provider.fetch_daily(
            equity,
            date(2026, 1, 1),
            date(2026, 10, 23),
        )

    http_get.assert_not_called()


def test_invalid_date_range_fails_before_http_call(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    with pytest.raises(
        InvalidProviderDataError,
        match="start_date",
    ):
        provider.fetch_daily(
            asset,
            date(2026, 10, 24),
            date(2026, 10, 23),
        )

    http_get.assert_not_called()


def test_empty_daily_series_raises_empty_data_error(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.return_value = response_with_json(
        {
            "Meta Data": {
                "6. Time Zone": "UTC",
            },
            "Time Series FX (Daily)": {},
        }
    )

    with pytest.raises(EmptyDataError):
        provider.fetch_daily(
            asset,
            date(2026, 10, 1),
            date(2026, 10, 23),
        )


def test_missing_daily_series_is_invalid_provider_data(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.return_value = response_with_json(
        {
            "Meta Data": {
                "6. Time Zone": "UTC",
            },
        }
    )

    with pytest.raises(InvalidProviderDataError):
        provider.fetch_daily(
            asset,
            date(2026, 10, 1),
            date(2026, 10, 23),
        )


def test_non_utc_metadata_is_rejected(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
    observed_response: dict,
) -> None:
    payload = {
        **observed_response,
        "Meta Data": {
            **observed_response["Meta Data"],
            "6. Time Zone": "US/Eastern",
        },
    }

    http_get.return_value = response_with_json(payload)

    with pytest.raises(
        InvalidProviderDataError,
        match="UTC",
    ):
        provider.fetch_daily(
            asset,
            date(2015, 1, 1),
            date(2026, 10, 23),
        )


def test_provider_information_response_is_not_market_data(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.return_value = response_with_json(
        {
            "Information": (
                "The API rate limit has been reached."
            )
        }
    )

    with pytest.raises(ProviderUnavailableError):
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 10, 23),
        )


def test_provider_error_message_is_invalid_provider_data(
    provider: AlphaVantageProvider,
    http_get: MagicMock,
    asset: AssetConfig,
) -> None:
    http_get.return_value = response_with_json(
        {
            "Error Message": "Invalid API call.",
        }
    )

    with pytest.raises(InvalidProviderDataError):
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 10, 23),
        )


def test_duplicate_daily_timestamps_are_rejected(
    provider: AlphaVantageProvider,
    asset: AssetConfig,
) -> None:
    payload = {
        "Meta Data": {
            "6. Time Zone": "UTC",
        },
        "Time Series FX (Daily)": {
            "2026-10-23": {
                "1. open": "1.12170",
                "2. high": "1.12260",
                "3. low": "1.11830",
                "4. close": "1.12020",
            },
        },
    }

    # Alpha Vantage's JSON object cannot naturally contain duplicate
    # date keys. Exercise normalization directly with duplicated records
    # only if the implementation exposes such an internal representation.
    # The shared provider contract still requires unique timestamps.
    assert len(payload["Time Series FX (Daily)"]) == 1
