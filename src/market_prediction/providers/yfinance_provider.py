"""Yahoo Finance implementation of the market-data provider contract."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import (
    YFDataException,
    YFPricesMissingError,
    YFRateLimitError,
    YFTickerMissingError,
)

from market_prediction.config import AssetConfig
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


class YFinanceProvider(MarketDataProvider):
    """Fetch and normalize daily OHLCV data from Yahoo Finance."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay_seconds: float = 2.0,
        backoff_multiplier: float = 2.0,
        timeout_seconds: float = 10.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        if initial_delay_seconds < 0:
            raise ValueError(
                "initial_delay_seconds must be greater than or equal to 0."
            )

        if backoff_multiplier <= 0:
            raise ValueError(
                "backoff_multiplier must be greater than 0."
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0."
            )

        self._max_attempts = max_attempts
        self._initial_delay_seconds = initial_delay_seconds
        self._backoff_multiplier = backoff_multiplier
        self._timeout_seconds = timeout_seconds
        self._sleep_fn = sleep_fn

    @property
    def name(self) -> str:
        """Return the canonical provider name."""
        return "yfinance"

    def fetch_daily(
        self,
        asset: AssetConfig,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data and normalize it to the provider contract."""
        if start_date > end_date:
            raise InvalidProviderDataError(
                "start_date must be less than or equal to end_date."
            )

        raw_data = self._fetch_with_retry(
            ticker=asset.ticker,
            start_date=start_date,
            end_date=end_date,
        )

        return self._normalize(
            raw_data=raw_data,
            ticker=asset.ticker,
            start_date=start_date,
            end_date=end_date,
        )

    def _fetch_with_retry(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch from Yahoo, retrying transient failures only."""
        delay = self._initial_delay_seconds

        for attempt in range(1, self._max_attempts + 1):
            try:
                yahoo_end_date = end_date + timedelta(days=1)

                result = yf.Ticker(ticker).history(
                    start=start_date.isoformat(),
                    end=yahoo_end_date.isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    timeout=self._timeout_seconds,
                    raise_errors=True,
                )

                if result is None or result.empty:
                    raise EmptyDataError(
                        f"Yahoo Finance returned no data for '{ticker}' "
                        f"between {start_date} and {end_date}."
                    )

                return result

            except EmptyDataError:
                raise

            except YFRateLimitError as exc:
                if attempt == self._max_attempts:
                    raise ProviderUnavailableError(
                        f"Yahoo Finance rate limit persisted after "
                        f"{self._max_attempts} attempts for '{ticker}'."
                    ) from exc

                self._sleep_fn(delay)
                delay *= self._backoff_multiplier

            except (
                TimeoutError,
                ConnectionError,
                socket.timeout,
                OSError,
            ) as exc:
                if attempt == self._max_attempts:
                    raise ProviderUnavailableError(
                        f"Yahoo Finance remained unavailable after "
                        f"{self._max_attempts} attempts for '{ticker}'."
                    ) from exc

                self._sleep_fn(delay)
                delay *= self._backoff_multiplier

            except (
                YFTickerMissingError,
                YFPricesMissingError,
            ) as exc:
                raise UnsupportedAssetError(
                    f"Yahoo Finance does not provide usable price data "
                    f"for ticker '{ticker}': {exc}"
                ) from exc

            except YFDataException as exc:
                raise InvalidProviderDataError(
                    f"Yahoo Finance returned unusable data "
                    f"for ticker '{ticker}': {exc}"
                ) from exc

            except ProviderError:
                raise

            except Exception as exc:
                raise ProviderError(
                    f"Unexpected Yahoo Finance failure for "
                    f"ticker '{ticker}': {exc}"
                ) from exc

        raise ProviderUnavailableError(
            f"Yahoo Finance failed for ticker '{ticker}'."
        )

    def _normalize(
        self,
        raw_data: pd.DataFrame,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Convert Yahoo's DataFrame into the locked provider schema."""
        if not isinstance(raw_data, pd.DataFrame):
            raise InvalidProviderDataError(
                f"Yahoo Finance returned {type(raw_data).__name__} "
                f"instead of a pandas DataFrame for '{ticker}'."
            )

        if raw_data.empty:
            raise EmptyDataError(
                f"Yahoo Finance returned no data for '{ticker}' "
                f"between {start_date} and {end_date}."
            )

        frame = raw_data.copy()

        # A single-ticker Ticker.history() call should normally return
        # flat columns. Refuse an unexpected MultiIndex rather than
        # guessing how to flatten an ambiguous response.
        if isinstance(frame.columns, pd.MultiIndex):
            raise InvalidProviderDataError(
                f"Yahoo Finance returned unexpected MultiIndex columns "
                f"for single ticker '{ticker}'."
            )

        frame = frame.reset_index()

        timestamp_column = self._find_timestamp_column(frame)

        required_yahoo_columns = {
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        }

        missing_columns = required_yahoo_columns.difference(frame.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise InvalidProviderDataError(
                f"Yahoo Finance response for '{ticker}' is missing "
                f"required columns: {missing}."
            )

        normalized = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    frame[timestamp_column],
                    utc=True,
                    errors="coerce",
                ),
                "open": pd.to_numeric(
                    frame["Open"],
                    errors="coerce",
                ),
                "high": pd.to_numeric(
                    frame["High"],
                    errors="coerce",
                ),
                "low": pd.to_numeric(
                    frame["Low"],
                    errors="coerce",
                ),
                "close": pd.to_numeric(
                    frame["Close"],
                    errors="coerce",
                ),
                "adj_close": self._normalized_adjusted_close(frame),
                "volume": pd.to_numeric(
                    frame["Volume"],
                    errors="coerce",
                ),
                "source": self.name,
            }
        )

        required_non_null = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "source",
        ]

        if normalized[required_non_null].isnull().any().any():
            raise InvalidProviderDataError(
                f"Yahoo Finance returned null or non-convertible "
                f"required values for '{ticker}'."
            )

        if normalized["timestamp"].duplicated().any():
            raise InvalidProviderDataError(
                f"Yahoo Finance returned duplicate timestamps "
                f"for '{ticker}'."
            )

        normalized = normalized.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        return normalized.loc[:, list(OHLCV_COLUMNS)]

    @staticmethod
    def _find_timestamp_column(frame: pd.DataFrame) -> str:
        """Locate Yahoo's date/index column after reset_index()."""
        for candidate in ("Date", "Datetime"):
            if candidate in frame.columns:
                return candidate

        raise InvalidProviderDataError(
            "Yahoo Finance response does not contain a Date "
            "or Datetime column."
        )

    @staticmethod
    def _normalized_adjusted_close(
        frame: pd.DataFrame,
    ) -> pd.Series:
        """Return adjusted close when Yahoo supplies it.

        Some instruments/providers may not expose an adjusted close.
        The locked Step 3b contract permits adj_close to be null.
        """
        if "Adj Close" not in frame.columns:
            return pd.Series(
                pd.NA,
                index=frame.index,
                dtype="Float64",
            )

        return pd.to_numeric(
            frame["Adj Close"],
            errors="coerce",
        )