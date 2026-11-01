from __future__ import annotations

import pytest

from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
    NewsProviderUnavailableError,
    NewsQuotaExceededError,
)
from market_prediction.providers.news.marketaux import (
    parse_marketaux_response,
)


def test_parse_marketaux_response_returns_data_items() -> None:
    payload = {
        "meta": {
            "found": 2,
            "returned": 2,
            "limit": 50,
            "page": 1,
        },
        "data": [
            {
                "uuid": "article-1",
                "title": "First article",
            },
            {
                "uuid": "article-2",
                "title": "Second article",
            },
        ],
    }

    result = parse_marketaux_response(
        status_code=200,
        payload=payload,
    )

    assert result.items == tuple(
        payload["data"]
    )


def test_parse_marketaux_response_accepts_empty_data() -> None:
    result = parse_marketaux_response(
        status_code=200,
        payload={
            "meta": {
                "found": 0,
                "returned": 0,
                "limit": 50,
                "page": 1,
            },
            "data": [],
        },
    )

    assert result.items == ()


def test_parse_marketaux_response_rejects_non_mapping_payload() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="mapping",
    ):
        parse_marketaux_response(
            status_code=200,
            payload=None,  # type: ignore[arg-type]
        )


def test_parse_marketaux_response_rejects_missing_data() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="data",
    ):
        parse_marketaux_response(
            status_code=200,
            payload={
                "meta": {
                    "found": 0,
                    "returned": 0,
                }
            },
        )


def test_parse_marketaux_response_rejects_non_list_data() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="data",
    ):
        parse_marketaux_response(
            status_code=200,
            payload={
                "data": {
                    "uuid": "article-1",
                }
            },
        )


def test_parse_marketaux_response_rejects_non_mapping_data_item() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="data",
    ):
        parse_marketaux_response(
            status_code=200,
            payload={
                "data": [
                    {
                        "uuid": "article-1",
                    },
                    "invalid-item",
                ]
            },
        )


def test_parse_marketaux_response_maps_rate_limit_to_quota() -> None:
    with pytest.raises(
        NewsQuotaExceededError,
    ):
        parse_marketaux_response(
            status_code=429,
            payload={
                "error": {
                    "message": "API rate limit exceeded"
                }
            },
        )


@pytest.mark.parametrize(
    "status_code",
    [
        401,
        403,
    ],
)
def test_parse_marketaux_response_maps_access_errors_to_invalid_data(
    status_code: int,
) -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match=str(status_code),
    ):
        parse_marketaux_response(
            status_code=status_code,
            payload={
                "error": {
                    "message": "Authentication or access denied"
                }
            },
        )


@pytest.mark.parametrize(
    "status_code",
    [
        500,
        502,
        503,
        504,
    ],
)
def test_parse_marketaux_response_maps_server_errors(
    status_code: int,
) -> None:
    with pytest.raises(
        NewsProviderUnavailableError,
    ):
        parse_marketaux_response(
            status_code=status_code,
            payload={
                "error": {
                    "message": "Service unavailable"
                }
            },
        )


def test_parse_marketaux_response_rejects_other_http_error() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="400",
    ):
        parse_marketaux_response(
            status_code=400,
            payload={
                "error": {
                    "message": "Invalid request"
                }
            },
        )


def test_parse_marketaux_response_rejects_invalid_status_code() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="status_code",
    ):
        parse_marketaux_response(
            status_code=0,
            payload={},
        )

def test_parse_response_returns_pagination_metadata() -> None:
    payload_items = [
        {
            "uuid": "article-1",
        },
        {
            "uuid": "article-2",
        },
    ]

    response = parse_marketaux_response(
        status_code=200,
        payload={
            "meta": {
                "found": 125,
                "returned": 2,
                "limit": 50,
                "page": 1,
            },
            "data": payload_items,
        },
    )

    assert response.found == 125
    assert response.returned == 2
    assert response.limit == 50
    assert response.page == 1
    assert response.items == tuple(
        payload_items
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("found", -1),
        ("returned", -1),
        ("limit", 0),
        ("page", 0),
        ("found", True),
        ("returned", True),
        ("limit", True),
        ("page", True),
        ("found", "125"),
        ("returned", "50"),
        ("limit", "50"),
        ("page", "1"),
    ],
)
def test_parse_response_rejects_invalid_pagination_metadata(
    field: str,
    value: object,
) -> None:
    meta = {
        "found": 125,
        "returned": 50,
        "limit": 50,
        "page": 1,
    }
    meta[field] = value

    with pytest.raises(
        InvalidNewsProviderDataError,
        match=field,
    ):
        parse_marketaux_response(
            status_code=200,
            payload={
                "meta": meta,
                "data": [],
            },
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "found",
        "returned",
        "limit",
        "page",
    ],
)
def test_parse_response_rejects_missing_pagination_metadata(
    missing_field: str,
) -> None:
    meta = {
        "found": 125,
        "returned": 50,
        "limit": 50,
        "page": 1,
    }
    del meta[missing_field]

    with pytest.raises(
        InvalidNewsProviderDataError,
        match=missing_field,
    ):
        parse_marketaux_response(
            status_code=200,
            payload={
                "meta": meta,
                "data": [],
            },
        )


def test_parse_response_rejects_returned_above_limit() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="returned",
    ):
        parse_marketaux_response(
            status_code=200,
            payload={
                "meta": {
                    "found": 100,
                    "returned": 51,
                    "limit": 50,
                    "page": 1,
                },
                "data": [],
            },
        )


def test_parse_response_rejects_returned_data_mismatch() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="returned",
    ):
        parse_marketaux_response(
            status_code=200,
            payload={
                "meta": {
                    "found": 100,
                    "returned": 2,
                    "limit": 50,
                    "page": 1,
                },
                "data": [
                    {
                        "uuid": "article-1",
                    }
                ],
            },
        )


def test_parse_response_rejects_found_below_returned() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="found",
    ):
        parse_marketaux_response(
            status_code=200,
            payload={
                "meta": {
                    "found": 1,
                    "returned": 2,
                    "limit": 50,
                    "page": 1,
                },
                "data": [
                    {
                        "uuid": "article-1",
                    },
                    {
                        "uuid": "article-2",
                    },
                ],
            },
        )