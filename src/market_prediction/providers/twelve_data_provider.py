"""Twelve Data implementation of the market-data provider contract."""

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
        return self._payload


class TwelveDataProvider(MarketDataProvider):
    """Fetch and normalize Twelve Data daily Forex OHLC data."""

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
        return "twelve_data"

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
                "Twelve Data provider currently supports only "
                f"forex assets; received '{asset.asset_class.value}'."
            )

        symbol = asset.symbol_for(ProviderName.TWELVE_DATA)

        payload = self._fetch_with_retry(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        return self._normalize(
            payload=payload,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    def _fetch_with_retry(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Fetch Twelve Data, retrying transient failures only."""
        delay = self._initial_delay_seconds

        params = {
            "symbol": symbol,
            "interval": "1day",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "order": "asc",
            "apikey": self._api_key,
        }

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._http_get(
                    f"{self._base_url}/time_series",
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
                        "Twelve Data returned a non-object JSON response "
                        f"for '{symbol}'."
                    )

                api_status = payload.get("status")
                api_code = payload.get("code")
                message = str(
                    payload.get("message", "Unknown provider error.")
                )

                is_rate_limited = (
                    status_code == 429
                    or api_code == 429
                )

                if is_rate_limited:
                    if attempt == self._max_attempts:
                        raise ProviderUnavailableError(
                            "Twelve Data rate limit persisted after "
                            f"{self._max_attempts} attempts for "
                            f"'{symbol}'."
                        )

                    self._sleep_fn(delay)
                    delay *= self._backoff_multiplier
                    continue

                if status_code == 404 or api_code == 404:
                    raise UnsupportedAssetError(
                        "Twelve Data does not provide usable data "
                        f"for symbol '{symbol}': {message}"
                    )

                if (
                    isinstance(status_code, int)
                    and status_code >= 500
                ):
                    if attempt == self._max_attempts:
                        raise ProviderUnavailableError(
                            "Twelve Data remained unavailable after "
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
                        "Twelve Data rejected the request for "
                        f"'{symbol}': {message}"
                    )

                if api_status == "error":
                    raise InvalidProviderDataError(
                        "Twelve Data returned an API error for "
                        f"'{symbol}': {message}"
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
                        "Twelve Data remained unavailable after "
                        f"{self._max_attempts} attempts for "
                        f"'{symbol}'."
                    ) from exc

                self._sleep_fn(delay)
                delay *= self._backoff_multiplier

            except ProviderError:
                raise

            except Exception as exc:
                raise ProviderError(
                    "Unexpected Twelve Data failure for "
                    f"symbol '{symbol}': {exc}"
                ) from exc

        raise ProviderUnavailableError(
            f"Twelve Data failed for symbol '{symbol}'."
        )

    def _normalize(
        self,
        *,
        payload: dict[str, Any],
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Convert Twelve Data JSON into the locked provider schema."""
        if "values" not in payload:
            raise InvalidProviderDataError(
                "Twelve Data response for "
                f"'{symbol}' is missing required field 'values'."
            )

        values = payload["values"]

        if not isinstance(values, list):
            raise InvalidProviderDataError(
                "Twelve Data response field 'values' must be a list "
                f"for '{symbol}'."
            )

        if not values:
            raise EmptyDataError(
                f"Twelve Data returned no data for '{symbol}' "
                f"between {start_date} and {end_date}."
            )

        required_fields = {
            "datetime",
            "open",
            "high",
            "low",
            "close",
        }

        for row in values:
            if not isinstance(row, dict):
                raise InvalidProviderDataError(
                    "Twelve Data returned a malformed price row "
                    f"for '{symbol}'."
                )

            missing = required_fields.difference(row)

            if missing:
                missing_text = ", ".join(sorted(missing))
                raise InvalidProviderDataError(
                    f"Twelve Data response for '{symbol}' is missing "
                    f"required fields: {missing_text}."
                )

        frame = pd.DataFrame(values)

        normalized = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    frame["datetime"],
                    utc=True,
                    errors="coerce",
                ),
                "open": pd.to_numeric(
                    frame["open"],
                    errors="coerce",
                ),
                "high": pd.to_numeric(
                    frame["high"],
                    errors="coerce",
                ),
                "low": pd.to_numeric(
                    frame["low"],
                    errors="coerce",
                ),
                "close": pd.to_numeric(
                    frame["close"],
                    errors="coerce",
                ),
                "adj_close": pd.Series(
                    pd.NA,
                    index=frame.index,
                    dtype="Float64",
                ),
                "volume": pd.Series(
                    pd.NA,
                    index=frame.index,
                    dtype="Float64",
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
                "Twelve Data returned null or non-convertible "
                f"required values for '{symbol}'."
            )

        if normalized["timestamp"].duplicated().any():
            raise InvalidProviderDataError(
                "Twelve Data returned duplicate timestamps "
                f"for '{symbol}'."
            )

        normalized = normalized.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        return normalized.loc[:, list(OHLCV_COLUMNS)]

    def _default_http_get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        timeout: float,
    ) -> _UrllibResponse:
        """Perform an HTTP GET using only Python's standard library."""
        query = urlencode(params)
        request = Request(
            f"{url}?{query}",
            headers={
                "User-Agent": "multi-asset-market-ml/step3g",
            },
        )

        try:
            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                body = response.read().decode("utf-8")

                return _UrllibResponse(
                    status_code=response.status,
                    payload=json.loads(body),
                )

        except HTTPError as exc:
            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {
                    "status": "error",
                    "code": exc.code,
                    "message": exc.reason,
                }

            return _UrllibResponse(
                status_code=exc.code,
                payload=payload,
            )

        except json.JSONDecodeError as exc:
            raise InvalidProviderDataError(
                "Twelve Data returned malformed JSON."
            ) from exc