"""Contract tests for MarketDataProvider implementations."""

from datetime import date

import pandas as pd
import pytest

from market_prediction.config import AssetClass, AssetConfig, ProviderName
from market_prediction.providers.base import (
    OHLCV_COLUMNS,
    MarketDataProvider,
)
from market_prediction.providers.exceptions import (
    EmptyDataError,
    InvalidProviderDataError,
    ProviderError,
    ProviderUnavailableError,
    UnsupportedAssetError,
)
from tests.fakes.fake_market_data_provider import (
    FakeMarketDataProvider,
    FakeProviderMode,
)


@pytest.fixture
def asset() -> AssetConfig:
    """Return a valid asset configuration for provider tests."""
    return AssetConfig(
        display_name="Apple",
        ticker="AAPL",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        calendar="XNYS",
        primary_provider=ProviderName.YFINANCE,
        backup_provider=ProviderName.STOOQ,
    )


@pytest.fixture
def start_date() -> date:
    """Return the beginning of the fake request range."""
    return date(2026, 1, 1)


@pytest.fixture
def end_date() -> date:
    """Return the end of the fake request range."""
    return date(2026, 1, 31)


def test_fake_provider_implements_market_data_provider() -> None:
    """A concrete provider must implement the shared abstraction."""
    provider = FakeMarketDataProvider()

    assert isinstance(provider, MarketDataProvider)


def test_successful_provider_returns_dataframe(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Successful provider calls must return pandas DataFrames."""
    provider = FakeMarketDataProvider()

    result = provider.fetch_daily(
        asset,
        start_date,
        end_date,
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_successful_provider_returns_exact_column_contract(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Successful results must expose the exact normalized schema."""
    provider = FakeMarketDataProvider()

    result = provider.fetch_daily(
        asset,
        start_date,
        end_date,
    )

    assert tuple(result.columns) == OHLCV_COLUMNS


def test_successful_provider_returns_chronological_unique_timestamps(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Successful results must have ordered, unique timestamps."""
    provider = FakeMarketDataProvider()

    result = provider.fetch_daily(
        asset,
        start_date,
        end_date,
    )

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique


def test_successful_provider_has_required_non_null_values(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Required OHLC fields must not contain null values."""
    provider = FakeMarketDataProvider()

    result = provider.fetch_daily(
        asset,
        start_date,
        end_date,
    )

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "source",
    ]

    assert not result[required_columns].isnull().any().any()


def test_successful_provider_identifies_its_source(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Every returned row must identify the provider that produced it."""
    provider = FakeMarketDataProvider()

    result = provider.fetch_daily(
        asset,
        start_date,
        end_date,
    )

    assert (result["source"] == provider.name).all()


def test_unreachable_source_raises_provider_unavailable(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Network-like failures must use ProviderUnavailableError."""
    provider = FakeMarketDataProvider(
        FakeProviderMode.UNAVAILABLE
    )

    with pytest.raises(ProviderUnavailableError):
        provider.fetch_daily(
            asset,
            start_date,
            end_date,
        )


def test_unsupported_asset_raises_specific_exception(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Invalid or unsupported assets must use UnsupportedAssetError."""
    provider = FakeMarketDataProvider(
        FakeProviderMode.UNSUPPORTED_ASSET
    )

    with pytest.raises(UnsupportedAssetError):
        provider.fetch_daily(
            asset,
            start_date,
            end_date,
        )


def test_empty_response_raises_empty_data_error(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """An empty upstream response must not count as success."""
    provider = FakeMarketDataProvider(
        FakeProviderMode.EMPTY
    )

    with pytest.raises(EmptyDataError):
        provider.fetch_daily(
            asset,
            start_date,
            end_date,
        )


def test_invalid_response_raises_invalid_provider_data_error(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Malformed provider output must fail explicitly."""
    provider = FakeMarketDataProvider(
        FakeProviderMode.INVALID_DATA
    )

    with pytest.raises(InvalidProviderDataError):
        provider.fetch_daily(
            asset,
            start_date,
            end_date,
        )


def test_partial_response_is_never_returned_as_success(
    asset: AssetConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Known partial results must fail rather than silently succeed."""
    provider = FakeMarketDataProvider(
        FakeProviderMode.PARTIAL_DATA
    )

    with pytest.raises(InvalidProviderDataError):
        provider.fetch_daily(
            asset,
            start_date,
            end_date,
        )


def test_invalid_date_range_is_rejected(
    asset: AssetConfig,
) -> None:
    """A logically invalid request range must fail explicitly."""
    provider = FakeMarketDataProvider()

    with pytest.raises(
        InvalidProviderDataError,
        match="start_date",
    ):
        provider.fetch_daily(
            asset,
            start_date=date(2026, 2, 1),
            end_date=date(2026, 1, 1),
        )


@pytest.mark.parametrize(
    "exception_type",
    [
        ProviderUnavailableError,
        UnsupportedAssetError,
        EmptyDataError,
        InvalidProviderDataError,
    ],
)
def test_specific_provider_exceptions_inherit_from_provider_error(
    exception_type: type[ProviderError],
) -> None:
    """All provider failures must be catchable via ProviderError."""
    assert issubclass(exception_type, ProviderError)