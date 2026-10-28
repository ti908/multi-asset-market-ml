"""Alpha Vantage implementation of the market-data provider contract."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from market_prediction.config import (
    AssetClass,
    AssetConfig,
    ProviderName,
)
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


HttpGet = Callable[..., Any]


class _UrllibResponse:
    """Small response adapter used by the default HTTP implementation."""

    def __init__(
        self,
        *,
        status_code: int,
        payload: dict[str, Any],
    ) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        """Return the decoded JSON payload."""
        return self._payload


class AlphaVantageProvider(MarketDataProvider):
    """Fetch and normalize Alpha Vantage daily Forex OHLC data."""

    _SERIES_KEY = "Time Series FX (Daily)"
    _TIME_ZONE_KEY = "6. Time Zone"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        max_attempts: int = 3,
        initial_delay_seconds: float = 2.0,
        backoff_multiplier: float = 2.0,
        timeout_seconds: float = 10.0,
        http_get: HttpGet | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty.")

        if not base_url.strip():
            raise ValueError("base_url must not be empty.")

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

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_attempts = max_attempts
        self._initial_delay_seconds = initial_delay_seconds
        self._backoff_multiplier = backoff_multiplier
        self._timeout_seconds = timeout_seconds
        self._http_get = http_get or self._default_http_get
        self._sleep_fn = sleep_fn

    @property
    def name(self) -> str:
        """Return the canonical provider name."""
        return "alpha_vantage"

    def fetch_daily(
        self,
        asset: AssetConfig,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily Forex OHLC and normalize it to the shared contract."""
        if start_date > end_date:
            raise InvalidProviderDataError(
                "start_date must be less than or equal to end_date."
            )

        if asset.asset_class is not AssetClass.FOREX:
            raise UnsupportedAssetError(
                "Alpha Vantage provider currently supports only "
                f"forex assets; received '{asset.asset_class.value}'."
            )

        symbol = asset.symbol_for(ProviderName.ALPHA_VANTAGE)
        from_symbol, to_symbol = self._parse_forex_symbol(symbol)

        payload = self._fetch_with_retry(
            symbol=symbol,
            from_symbol=from_symbol,
            to_symbol=to_symbol,
        )

        return self._normalize(
            payload=payload,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def _parse_forex_symbol(
        symbol: str,
    ) -> tuple[str, str]:
        """Parse a configured BASE/QUOTE Alpha Vantage Forex symbol."""
        parts = symbol.split("/")

        if (
            len(parts) != 2
            or not parts[0].strip()
            or not parts[1].strip()
        ):
            raise InvalidProviderDataError(
                "Alpha Vantage forex symbol must use "
                f"'BASE/QUOTE' format; received '{symbol}'."
            )

        return (
            parts[0].strip().upper(),
            parts[1].strip().upper(),
        )

    def _fetch_with_retry(
        self,
        *,
        symbol: str,
        from_symbol: str,
        to_symbol: str,
    ) -> dict[str, Any]:
        """Fetch Alpha Vantage data, retrying transient failures only."""
        delay = self._initial_delay_seconds

        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": "full",
            "apikey": self._api_key,
        }

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._http_get(
                    self._base_url,
                    params=params,
                    timeout=self._timeout_seconds,
                )

                status_code = getattr(
                    response,
                    "status_code",
                    None,
                )

                payload = response.json()

                if not isinstance(payload, dict):
                    raise InvalidProviderDataError(
                        "Alpha Vantage returned a non-object JSON response "
                        f"for '{symbol}'."
                    )

                if status_code == 429:
                    if attempt == self._max_attempts:
                        raise ProviderUnavailableError(
                            "Alpha Vantage rate limit persisted after "
                            f"{self._max_attempts} attempts for "
                            f"'{symbol}'."
                        )

                    self._sleep_fn(delay)
                    delay *= self._backoff_multiplier
                    continue

                if (
                    isinstance(status_code, int)
                    and status_code >= 500
                ):
                    if attempt == self._max_attempts:
                        raise ProviderUnavailableError(
                            "Alpha Vantage remained unavailable after "
                            f"{self._max_attempts} attempts for "
                            f"'{symbol}'."
                        )

                    self._sleep_fn(delay)
                    delay *= self._backoff_multiplier
                    continue

                if (
                    isinstance(status_code, int)
                    and status_code >= 400
                ):
                    raise InvalidProviderDataError(
                        "Alpha Vantage rejected the request for "
                        f"'{symbol}' with HTTP {status_code}."
                    )

                if "Error Message" in payload:
                    raise InvalidProviderDataError(
                        "Alpha Vantage returned an API error for "
                        f"'{symbol}': {payload['Error Message']}"
                    )

                information = payload.get("Information")
                note = payload.get("Note")

                if information is not None or note is not None:
                    message = str(
                        information
                        if information is not None
                        else note
                    )

                    raise ProviderUnavailableError(
                        "Alpha Vantage returned an availability "
                        f"message for '{symbol}': {message}"
                    )

                return payload

            except (
                TimeoutError,
                ConnectionError,
                socket.timeout,
                URLError,
            ) as exc:
                if attempt == self._max_attempts:
                    raise ProviderUnavailableError(
                        "Alpha Vantage remained unavailable after "
                        f"{self._max_attempts} attempts for "
                        f"'{symbol}'."
                    ) from exc

                self._sleep_fn(delay)
                delay *= self._backoff_multiplier

            except ProviderError:
                raise

            except Exception as exc:
                raise ProviderError(
                    "Unexpected Alpha Vantage failure for "
                    f"symbol '{symbol}': {exc}"
                ) from exc

        raise ProviderUnavailableError(
            f"Alpha Vantage failed for symbol '{symbol}'."
        )

    def _normalize(
        self,
        *,
        payload: dict[str, Any],
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Convert Alpha Vantage FX_DAILY JSON to the locked schema."""
        metadata = payload.get("Meta Data")

        if not isinstance(metadata, dict):
            raise InvalidProviderDataError(
                "Alpha Vantage response for "
                f"'{symbol}' is missing valid metadata."
            )

        timezone = metadata.get(self._TIME_ZONE_KEY)

        if timezone != "UTC":
            raise InvalidProviderDataError(
                "Alpha Vantage FX_DAILY timezone must be UTC; "
                f"received {timezone!r} for '{symbol}'."
            )

        if self._SERIES_KEY not in payload:
            raise InvalidProviderDataError(
                "Alpha Vantage response for "
                f"'{symbol}' is missing required daily FX series."
            )

        series = payload[self._SERIES_KEY]

        if not isinstance(series, dict):
            raise InvalidProviderDataError(
                "Alpha Vantage daily FX series must be an object "
                f"for '{symbol}'."
            )

        if not series:
            raise EmptyDataError(
                f"Alpha Vantage returned no data for '{symbol}' "
                f"between {start_date} and {end_date}."
            )

        required_fields = {
            "1. open",
            "2. high",
            "3. low",
            "4. close",
        }

        rows: list[dict[str, Any]] = []

        for date_label, values in series.items():
            if not isinstance(values, dict):
                raise InvalidProviderDataError(
                    "Alpha Vantage returned a malformed price row "
                    f"for '{symbol}'."
                )

            missing = required_fields.difference(values)

            if missing:
                missing_text = ", ".join(sorted(missing))
                raise InvalidProviderDataError(
                    f"Alpha Vantage response for '{symbol}' is missing "
                    f"required fields: {missing_text}."
                )

            try:
                timestamp = pd.to_datetime(
                    date_label,
                    utc=True,
                    errors="raise",
                )

                open_price = float(values["1. open"])
                high_price = float(values["2. high"])
                low_price = float(values["3. low"])
                close_price = float(values["4. close"])
            except (TypeError, ValueError) as exc:
                raise InvalidProviderDataError(
                    "Alpha Vantage returned malformed daily FX values "
                    f"for '{symbol}'."
                ) from exc

            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "adj_close": pd.NA,
                    "volume": pd.NA,
                    "source": self.name,
                }
            )

        frame = pd.DataFrame(rows)

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
        )

        frame = frame.loc[
            (frame["timestamp"].dt.date >= start_date)
            & (frame["timestamp"].dt.date <= end_date)
        ].copy()

        if frame.empty:
            raise EmptyDataError(
                f"Alpha Vantage returned no data for '{symbol}' "
                f"between {start_date} and {end_date}."
            )

        frame = frame.sort_values(
            "timestamp",
            kind="stable",
        ).reset_index(drop=True)

        if frame["timestamp"].duplicated().any():
            raise InvalidProviderDataError(
                "Alpha Vantage returned duplicate timestamps "
                f"for '{symbol}'."
            )

        required_output = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "source",
        ]

        if frame[required_output].isnull().any().any():
            raise InvalidProviderDataError(
                "Alpha Vantage returned null required values "
                f"for '{symbol}'."
            )

        frame["adj_close"] = frame["adj_close"].astype("Float64")
        frame["volume"] = frame["volume"].astype("Float64")

        return frame.loc[:, OHLCV_COLUMNS]

    @staticmethod
    def _default_http_get(
        url: str,
        *,
        params: dict[str, str],
        timeout: float,
    ) -> _UrllibResponse:
        """Perform an HTTP GET using only the Python standard library."""
        query = urlencode(params)

        request = Request(
            f"{url}?{query}",
            headers={
                "User-Agent": "multi-asset-market-ml/1.0",
            },
        )

        try:
            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                status_code = response.getcode()
                raw_payload = response.read()
        except HTTPError as exc:
            status_code = exc.code
            raw_payload = exc.read()

        try:
            payload = json.loads(
                raw_payload.decode("utf-8")
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise InvalidProviderDataError(
                "Alpha Vantage returned malformed JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidProviderDataError(
                "Alpha Vantage returned a non-object JSON response."
            )

        return _UrllibResponse(
            status_code=status_code,
            payload=payload,
        )
