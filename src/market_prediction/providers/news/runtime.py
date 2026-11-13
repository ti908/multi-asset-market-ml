from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class UrllibJsonTransport:
    """Production JSON-over-HTTP transport using urllib."""

    def get_json(
        self,
        *,
        url: str,
        params: dict[str, str],
        timeout_seconds: float,
    ) -> tuple[int, object]:
        """Perform a GET request and return status plus JSON payload."""

        query_string = urlencode(params)

        request_url = url

        if query_string:
            separator = (
                "&"
                if "?" in request_url
                else "?"
            )
            request_url = (
                f"{request_url}"
                f"{separator}"
                f"{query_string}"
            )

        request = Request(
            request_url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(compatible; "
                    "multi-asset-market-ml/1.0)"
                ),
            },
        )

        try:
            with urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                status_code = response.status
                body = response.read()

        except HTTPError as exc:
            status_code = exc.code
            body = exc.read()

        try:
            payload = json.loads(
                body.decode("utf-8")
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "HTTP response must contain valid JSON"
            ) from exc

        return status_code, payload


class SystemUtcClock:
    """Production clock returning the current UTC time."""

    def now_utc(self) -> datetime:
        """Return the current timezone-aware UTC timestamp."""

        return datetime.now(timezone.utc)
