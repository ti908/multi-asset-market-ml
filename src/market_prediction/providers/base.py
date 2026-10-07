"""Provider-neutral market-data interface.

Every concrete market-data source must implement this contract.

The interface deliberately contains no concepts specific to yfinance,
Dukascopy, Stooq, HTTP clients, or any other vendor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from market_prediction.config import AssetConfig


OHLCV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
)


class MarketDataProvider(ABC):
    """Abstract interface implemented by every market-data provider.

    A successful call returns a normalized pandas DataFrame containing
    exactly the columns defined by ``OHLCV_COLUMNS``.

    Provider implementations are responsible for translating their
    vendor-specific response into this normalized representation before
    returning it.

    Implementations must also translate vendor-specific failures into
    the project's ProviderError hierarchy.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the canonical provider name."""

    @abstractmethod
    def fetch_daily(
        self,
        asset: AssetConfig,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch normalized daily OHLCV observations.

        Args:
            asset:
                Validated asset configuration loaded from assets.yaml.

            start_date:
                Inclusive beginning of the requested date range.

            end_date:
                Inclusive end of the requested date range.

        Returns:
            A pandas DataFrame containing exactly these columns,
            in this order:

            - timestamp
            - open
            - high
            - low
            - close
            - adj_close
            - volume
            - source

            Additional contract requirements:

            - one row represents one daily observation;
            - timestamp values must be pandas datetime values;
            - rows must be ordered chronologically;
            - timestamps must be unique;
            - open/high/low/close must not contain null values;
            - adj_close may contain null values;
            - volume may contain null values;
            - source must identify this provider for every row;
            - an empty DataFrame must never be returned as success.

        Raises:
            ProviderUnavailableError:
                The upstream data source cannot be reached.

            UnsupportedAssetError:
                The requested asset/ticker is invalid or unsupported.

            EmptyDataError:
                The source responds successfully but provides no
                observations for the requested range.

            InvalidProviderDataError:
                The provider response cannot satisfy this normalized
                contract.

            ProviderError:
                Base type for other provider failures conforming to
                the provider exception contract.
        """