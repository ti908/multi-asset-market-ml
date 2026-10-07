"""In-memory MarketDataProvider used exclusively by tests."""

from __future__ import annotations

from datetime import date
from enum import Enum

import pandas as pd

from market_prediction.config import AssetConfig
from market_prediction.providers.base import (
    OHLCV_COLUMNS,
    MarketDataProvider,
)
from market_prediction.providers.exceptions import (
    EmptyDataError,
    InvalidProviderDataError,
    ProviderUnavailableError,
    UnsupportedAssetError,
)


class FakeProviderMode(str, Enum):
    """Behaviors the fake provider can simulate."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_ASSET = "unsupported_asset"
    EMPTY = "empty"
    INVALID_DATA = "invalid_data"
    PARTIAL_DATA = "partial_data"


class FakeMarketDataProvider(MarketDataProvider):
    """Deterministic in-memory implementation of MarketDataProvider.

    This provider performs no network or file-system access. Its only
    purpose is to verify the MarketDataProvider contract.
    """

    def __init__(
        self,
        mode: FakeProviderMode = FakeProviderMode.SUCCESS,
    ) -> None:
        self._mode = mode

    @property
    def name(self) -> str:
        """Return the fake provider's canonical test name."""
        return "fake"

    def fetch_daily(
        self,
        asset: AssetConfig,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Return deterministic data or simulate a provider failure."""
        if start_date > end_date:
            raise InvalidProviderDataError(
                "start_date must be less than or equal to end_date."
            )

        if self._mode is FakeProviderMode.UNAVAILABLE:
            raise ProviderUnavailableError(
                "Fake provider is deliberately unavailable."
            )

        if self._mode is FakeProviderMode.UNSUPPORTED_ASSET:
            raise UnsupportedAssetError(
                f"Fake provider does not support ticker '{asset.ticker}'."
            )

        if self._mode is FakeProviderMode.EMPTY:
            raise EmptyDataError(
                f"No data returned for '{asset.ticker}' between "
                f"{start_date} and {end_date}."
            )

        if self._mode is FakeProviderMode.INVALID_DATA:
            raise InvalidProviderDataError(
                "Fake provider deliberately produced invalid data."
            )

        if self._mode is FakeProviderMode.PARTIAL_DATA:
            raise InvalidProviderDataError(
                "Fake provider detected an incomplete response and "
                "refused to return a partial result."
            )

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-01-02",
                        "2026-01-05",
                        "2026-01-06",
                    ],
                    utc=True,
                ),
                "open": [100.0, 102.0, 101.0],
                "high": [103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 100.5],
                "close": [102.0, 101.0, 104.0],
                "adj_close": [102.0, 101.0, 104.0],
                "volume": [1_000_000.0, 1_200_000.0, 1_100_000.0],
                "source": [self.name, self.name, self.name],
            }
        )

        return frame.loc[:, list(OHLCV_COLUMNS)]