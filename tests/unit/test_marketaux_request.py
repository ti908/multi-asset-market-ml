from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from market_prediction.providers.news.base import (
    NewsQuery,
)
from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
)
from market_prediction.providers.news.marketaux import (
    build_marketaux_request,
)


UTC = timezone.utc


def make_query() -> NewsQuery:
    return NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
        context_terms=(),
    )


def test_build_marketaux_request_maps_base_url() -> None:
    url, params = build_marketaux_request(
        base_url="https://api.marketaux.com/v1",
        api_token="secret-token",
        query=make_query(),
        start_at_utc=datetime(
            2025,
            10,
            30,
            0,
            0,
            tzinfo=UTC,
        ),
        end_at_utc=datetime(
            2025,
            10,
            31,
            0,
            0,
            tzinfo=UTC,
        ),
    )

    assert url == (
        "https://api.marketaux.com/v1/news/all"
    )
    assert isinstance(params, dict)


def test_build_marketaux_request_maps_search_terms() -> None:
    _, params = build_marketaux_request(
        base_url="https://api.marketaux.com/v1",
        api_token="secret-token",
        query=make_query(),
        start_at_utc=datetime(
            2025,
            10,
            30,
            tzinfo=UTC,
        ),
        end_at_utc=datetime(
            2025,
            10,
            31,
            tzinfo=UTC,
        ),
    )

    assert params["search"] == (
        '"AAPL" | "Apple Inc"'
    )


def test_build_marketaux_request_maps_window() -> None:
    _, params = build_marketaux_request(
        base_url="https://api.marketaux.com/v1",
        api_token="secret-token",
        query=make_query(),
        start_at_utc=datetime(
            2025,
            10,
            30,
            12,
            30,
            tzinfo=UTC,
        ),
        end_at_utc=datetime(
            2025,
            10,
            31,
            8,
            45,
            tzinfo=UTC,
        ),
    )

    assert params["published_after"] == (
        "2025-10-30T12:30:00Z"
    )
    assert params["published_before"] == (
        "2025-10-31T08:45:00Z"
    )


def test_build_marketaux_request_maps_language() -> None:
    _, params = build_marketaux_request(
        base_url="https://api.marketaux.com/v1",
        api_token="secret-token",
        query=make_query(),
        start_at_utc=datetime(
            2025,
            10,
            30,
            tzinfo=UTC,
        ),
        end_at_utc=datetime(
            2025,
            10,
            31,
            tzinfo=UTC,
        ),
    )

    assert params["language"] == "en"


def test_build_marketaux_request_maps_api_token() -> None:
    _, params = build_marketaux_request(
        base_url="https://api.marketaux.com/v1",
        api_token="secret-token",
        query=make_query(),
        start_at_utc=datetime(
            2025,
            10,
            30,
            tzinfo=UTC,
        ),
        end_at_utc=datetime(
            2025,
            10,
            31,
            tzinfo=UTC,
        ),
    )

    assert params["api_token"] == "secret-token"


def test_build_marketaux_request_does_not_mutate_query() -> None:
    query = make_query()

    build_marketaux_request(
        base_url="https://api.marketaux.com/v1",
        api_token="secret-token",
        query=query,
        start_at_utc=datetime(
            2025,
            10,
            30,
            tzinfo=UTC,
        ),
        end_at_utc=datetime(
            2025,
            10,
            31,
            tzinfo=UTC,
        ),
    )

    assert query.strict_terms == (
        "AAPL",
        "Apple Inc",
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "   ",
    ],
)
def test_build_marketaux_request_rejects_blank_base_url(
    base_url: str,
) -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="base_url",
    ):
        build_marketaux_request(
            base_url=base_url,
            api_token="secret-token",
            query=make_query(),
            start_at_utc=datetime(
                2025,
                10,
                30,
                tzinfo=UTC,
            ),
            end_at_utc=datetime(
                2025,
                10,
                31,
                tzinfo=UTC,
            ),
        )


@pytest.mark.parametrize(
    "api_token",
    [
        "",
        "   ",
    ],
)
def test_build_marketaux_request_rejects_blank_api_token(
    api_token: str,
) -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="api_token",
    ):
        build_marketaux_request(
            base_url="https://api.marketaux.com/v1",
            api_token=api_token,
            query=make_query(),
            start_at_utc=datetime(
                2025,
                10,
                30,
                tzinfo=UTC,
            ),
            end_at_utc=datetime(
                2025,
                10,
                31,
                tzinfo=UTC,
            ),
        )


def test_build_marketaux_request_rejects_naive_window() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="UTC",
    ):
        build_marketaux_request(
            base_url="https://api.marketaux.com/v1",
            api_token="secret-token",
            query=make_query(),
            start_at_utc=datetime(
                2025,
                10,
                30,
            ),
            end_at_utc=datetime(
                2025,
                10,
                31,
                tzinfo=UTC,
            ),
        )


def test_build_marketaux_request_rejects_non_utc_window() -> None:
    non_utc = timezone(
        timedelta(hours=9)
    )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="UTC",
    ):
        build_marketaux_request(
            base_url="https://api.marketaux.com/v1",
            api_token="secret-token",
            query=make_query(),
            start_at_utc=datetime(
                2025,
                10,
                30,
                tzinfo=non_utc,
            ),
            end_at_utc=datetime(
                2025,
                10,
                31,
                tzinfo=UTC,
            ),
        )


def test_build_marketaux_request_rejects_empty_window() -> None:
    instant = datetime(
        2025,
        10,
        30,
        tzinfo=UTC,
    )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="start_at_utc",
    ):
        build_marketaux_request(
            base_url="https://api.marketaux.com/v1",
            api_token="secret-token",
            query=make_query(),
            start_at_utc=instant,
            end_at_utc=instant,
        )

def test_build_request_includes_page_number() -> None:
    url, params = build_marketaux_request(
        base_url="https://api.marketaux.com/v1",
        api_token="secret-token",
        query=make_query(),
        start_at_utc=datetime(
            2025,
            10,
            30,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        end_at_utc=datetime(
            2025,
            10,
            31,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        page=3,
    )

    assert url == (
        "https://api.marketaux.com/v1/news/all"
    )
    assert params["page"] == "3"


@pytest.mark.parametrize(
    "page",
    [
        0,
        -1,
        True,
        1.5,
        "1",
    ],
)
def test_build_request_rejects_invalid_page(
    page: object,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
        match="page",
    ):
        build_marketaux_request(
            base_url="https://api.marketaux.com/v1",
            api_token="secret-token",
            query=make_query(),
            start_at_utc=datetime(
                2025,
                10,
                30,
                0,
                0,
                tzinfo=timezone.utc,
            ),
            end_at_utc=datetime(
                2025,
                10,
                31,
                0,
                0,
                tzinfo=timezone.utc,
            ),
            page=page,
        )