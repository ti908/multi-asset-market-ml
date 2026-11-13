from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from urllib.error import HTTPError

import pytest

from market_prediction.providers.news.runtime import (
    SystemUtcClock,
    UrllibJsonTransport,
)


class FakeResponse:
    """Minimal context-managed urllib response."""

    def __init__(
        self,
        *,
        status: int,
        payload: object,
    ) -> None:
        self.status = status
        self._body = json.dumps(payload).encode(
            "utf-8"
        )

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        return None


def test_transport_returns_status_and_json(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(
            request,
            *,
            timeout,
    ):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["user_agent"] = (
            request.get_header(
                "User-agent"
            )
        )

        return FakeResponse(
            status=200,
            payload={
                "data": [
                    {"uuid": "article-1"}
                ]
            },
        )

    monkeypatch.setattr(
        "market_prediction.providers.news.runtime.urlopen",
        fake_urlopen,
    )

    transport = UrllibJsonTransport()

    status_code, payload = transport.get_json(
        url="https://example.com/news",
        params={
            "symbols": "AAPL",
            "page": "1",
        },
        timeout_seconds=30.0,
    )

    assert status_code == 200

    assert payload == {
        "data": [
            {"uuid": "article-1"}
        ]
    }

    assert captured["timeout"] == 30.0

    assert (
            captured["url"]
            == "https://example.com/news"
               "?symbols=AAPL&page=1"
    )

    assert captured["user_agent"] == (
        "Mozilla/5.0 "
        "(compatible; "
        "multi-asset-market-ml/1.0)"
    )


def test_transport_preserves_http_error_payload(
    monkeypatch,
) -> None:
    error_payload = {
        "error": {
            "message": "rate limit"
        }
    }

    def fake_urlopen(
        request,
        *,
        timeout,
    ):
        raise HTTPError(
            url=request.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=BytesIO(
                json.dumps(
                    error_payload
                ).encode("utf-8")
            ),
        )

    monkeypatch.setattr(
        "market_prediction.providers.news.runtime.urlopen",
        fake_urlopen,
    )

    transport = UrllibJsonTransport()

    status_code, payload = transport.get_json(
        url="https://example.com/news",
        params={"page": "1"},
        timeout_seconds=10.0,
    )

    assert status_code == 429
    assert payload == error_payload


def test_transport_rejects_invalid_json(
    monkeypatch,
) -> None:
    class InvalidJsonResponse:
        status = 200

        def read(self) -> bytes:
            return b"not-json"

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            traceback,
        ) -> None:
            return None

    monkeypatch.setattr(
        "market_prediction.providers.news.runtime.urlopen",
        lambda request, *, timeout: InvalidJsonResponse(),
    )

    transport = UrllibJsonTransport()

    with pytest.raises(
        ValueError,
        match="valid JSON",
    ):
        transport.get_json(
            url="https://example.com/news",
            params={},
            timeout_seconds=10.0,
        )


def test_system_utc_clock_returns_aware_utc() -> None:
    value = SystemUtcClock().now_utc()

    assert isinstance(value, datetime)
    assert value.tzinfo is not None
    assert value.utcoffset() is not None
    assert value.utcoffset().total_seconds() == 0

    assert (
        value.tzinfo
        == timezone.utc
    )