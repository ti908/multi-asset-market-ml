"""Reusable assertions for MarketDataProvider implementations."""

from __future__ import annotations

import pandas as pd

from market_prediction.providers.base import (
    OHLCV_COLUMNS,
    MarketDataProvider,
)


def assert_market_data_provider_contract(
    provider: MarketDataProvider,
    result: pd.DataFrame,
) -> None:
    """Assert that a successful provider result satisfies the shared contract.

    This function contains implementation-neutral assertions only.
    It can therefore be reused by the fake provider, YFinanceProvider,
    and future providers such as Stooq or Dukascopy.
    """
    assert isinstance(provider, MarketDataProvider)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty

    assert tuple(result.columns) == OHLCV_COLUMNS

    assert pd.api.types.is_datetime64_any_dtype(
        result["timestamp"]
    )

    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "source",
    ]

    assert not result[required_columns].isnull().any().any()

    assert (result["source"] == provider.name).all()