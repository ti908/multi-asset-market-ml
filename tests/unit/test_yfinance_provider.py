"""Offline unit tests for YFinanceProvider."""

from datetime import date
from unittest.mock import MagicMock, patch

from tests.contracts.market_data_provider_contract import (
    assert_market_data_provider_contract,
)

import pandas as pd
import pytest
from yfinance.exceptions import (
    YFDataException,
    YFPricesMissingError,
    YFRateLimitError,
)

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
from market_prediction.providers.yfinance_provider import (
    YFinanceProvider,
)


@pytest.fixture
def asset() -> AssetConfig:
    """Return a valid AAPL asset configuration."""
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
def yahoo_frame() -> pd.DataFrame:
    """Return representative output from yfinance Ticker.history()."""
    index = pd.DatetimeIndex(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
        ],
        name="Date",
        tz="America/New_York",
    )

    return pd.DataFrame(
        {
            "Open": [100.0, 102.0, 101.0],
            "High": [103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 100.5],
            "Close": [102.0, 101.0, 104.0],
            "Adj Close": [102.0, 101.0, 104.0],
            "Volume": [1_000_000, 1_200_000, 1_100_000],
        },
        index=index,
    )


@pytest.fixture
def provider() -> YFinanceProvider:
    """Return a provider whose retry sleeps take no real time."""
    return YFinanceProvider(
        max_attempts=3,
        initial_delay_seconds=2.0,
        backoff_multiplier=2.0,
        sleep_fn=lambda _: None,
    )


def test_yfinance_provider_implements_market_data_provider() -> None:
    provider = YFinanceProvider()

    assert isinstance(provider, MarketDataProvider)
    assert provider.name == "yfinance"


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_successful_fetch_returns_exact_contract_shape(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
    yahoo_frame: pd.DataFrame,
) -> None:
    mock_ticker.return_value.history.return_value = yahoo_frame

    result = provider.fetch_daily(
        asset,
        date(2026, 1, 1),
        date(2026, 1, 6),
    )

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == OHLCV_COLUMNS
    assert len(result) == 3
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique
    assert (result["source"] == "yfinance").all()


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_end_date_is_converted_from_inclusive_to_yahoo_exclusive(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
    yahoo_frame: pd.DataFrame,
) -> None:
    mock_ticker.return_value.history.return_value = yahoo_frame

    provider.fetch_daily(
        asset,
        date(2026, 1, 1),
        date(2026, 1, 6),
    )

    _, kwargs = mock_ticker.return_value.history.call_args

    assert kwargs["start"] == "2026-01-01"
    assert kwargs["end"] == "2026-01-07"


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_yahoo_call_uses_daily_unadjusted_prices(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
    yahoo_frame: pd.DataFrame,
) -> None:
    mock_ticker.return_value.history.return_value = yahoo_frame

    provider.fetch_daily(
        asset,
        date(2026, 1, 1),
        date(2026, 1, 6),
    )

    _, kwargs = mock_ticker.return_value.history.call_args

    assert kwargs["interval"] == "1d"
    assert kwargs["auto_adjust"] is False
    assert kwargs["actions"] is False


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_empty_result_fails_without_retry(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
) -> None:
    mock_ticker.return_value.history.return_value = pd.DataFrame()

    with pytest.raises(EmptyDataError):
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 1, 31),
        )

    assert mock_ticker.return_value.history.call_count == 1


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_missing_price_data_maps_to_unsupported_asset(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
) -> None:
    mock_ticker.return_value.history.side_effect = (
        YFPricesMissingError(
            "AAPL",
            "(1d 2026-01-01 -> 2026-01-31)",
        )
    )

    with pytest.raises(UnsupportedAssetError):
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 1, 31),
        )

    assert mock_ticker.return_value.history.call_count == 1


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_timeout_retries_then_succeeds(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
    yahoo_frame: pd.DataFrame,
) -> None:
    history = mock_ticker.return_value.history
    history.side_effect = [
        TimeoutError("temporary timeout"),
        yahoo_frame,
    ]

    result = provider.fetch_daily(
        asset,
        date(2026, 1, 1),
        date(2026, 1, 31),
    )

    assert not result.empty
    assert history.call_count == 2


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_timeout_exhaustion_maps_to_provider_unavailable(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
) -> None:
    history = mock_ticker.return_value.history
    history.side_effect = TimeoutError("Yahoo timed out")

    with pytest.raises(ProviderUnavailableError):
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 1, 31),
        )

    assert history.call_count == 3


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_rate_limit_retries_then_succeeds(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
    yahoo_frame: pd.DataFrame,
) -> None:
    history = mock_ticker.return_value.history
    history.side_effect = [
        YFRateLimitError(),
        yahoo_frame,
    ]

    result = provider.fetch_daily(
        asset,
        date(2026, 1, 1),
        date(2026, 1, 31),
    )

    assert not result.empty
    assert history.call_count == 2


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_rate_limit_exhaustion_maps_to_provider_unavailable(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
) -> None:
    history = mock_ticker.return_value.history
    history.side_effect = YFRateLimitError()

    with pytest.raises(ProviderUnavailableError):
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 1, 31),
        )

    assert history.call_count == 3


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_yfinance_data_error_is_wrapped(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
) -> None:
    mock_ticker.return_value.history.side_effect = (
        YFDataException("invalid Yahoo response")
    )

    with pytest.raises(InvalidProviderDataError):
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 1, 31),
        )


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_unknown_exception_never_leaks_raw(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
) -> None:
    mock_ticker.return_value.history.side_effect = RuntimeError(
        "unexpected library failure"
    )

    with pytest.raises(ProviderError) as exc_info:
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 1, 31),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_missing_required_column_is_rejected(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
    yahoo_frame: pd.DataFrame,
) -> None:
    broken = yahoo_frame.drop(columns=["Close"])
    mock_ticker.return_value.history.return_value = broken

    with pytest.raises(
        InvalidProviderDataError,
        match="Close",
    ):
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 1, 31),
        )


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_null_required_value_is_rejected(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
    yahoo_frame: pd.DataFrame,
) -> None:
    broken = yahoo_frame.copy()
    broken.iloc[0, broken.columns.get_loc("Close")] = float("nan")

    mock_ticker.return_value.history.return_value = broken

    with pytest.raises(InvalidProviderDataError):
        provider.fetch_daily(
            asset,
            date(2026, 1, 1),
            date(2026, 1, 31),
        )


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_missing_adjusted_close_is_allowed(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
    yahoo_frame: pd.DataFrame,
) -> None:
    frame = yahoo_frame.drop(columns=["Adj Close"])
    mock_ticker.return_value.history.return_value = frame

    result = provider.fetch_daily(
        asset,
        date(2026, 1, 1),
        date(2026, 1, 31),
    )

    assert result["adj_close"].isna().all()


@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_yfinance_provider_satisfies_shared_contract(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
    yahoo_frame: pd.DataFrame,
) -> None:
    """YFinanceProvider must satisfy the same contract as the fake provider."""
    mock_ticker.return_value.history.return_value = yahoo_frame

    result = provider.fetch_daily(
        asset,
        date(2026, 1, 1),
        date(2026, 1, 31),
    )

    assert_market_data_provider_contract(
        provider,
        result,
    )

@patch(
    "market_prediction.providers.yfinance_provider.yf.Ticker"
)
def test_invalid_date_range_fails_before_yahoo_call(
    mock_ticker: MagicMock,
    provider: YFinanceProvider,
    asset: AssetConfig,
) -> None:
    with pytest.raises(
        InvalidProviderDataError,
        match="start_date",
    ):
        provider.fetch_daily(
            asset,
            date(2026, 2, 1),
            date(2026, 1, 1),
        )

    mock_ticker.assert_not_called()

