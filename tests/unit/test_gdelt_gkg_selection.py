from __future__ import annotations

import pytest

from market_prediction.providers.news.base import NewsQuery
from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
)
from market_prediction.providers.news.gdelt import (
    GdeltGkgRecord,
    parse_gkg_row,
    select_gkg_records,
)


def make_row(
    *,
    record_id: str,
    organizations: str = "",
    all_names: str = "",
) -> list[str]:
    row = [""] * 27

    row[0] = record_id
    row[1] = "20150219000000"
    row[2] = "1"
    row[3] = "example.com"
    row[4] = (
        f"https://example.com/articles/{record_id}"
    )

    row[16] = organizations
    row[23] = all_names

    return row


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


def test_select_gkg_records_returns_only_matching_records() -> None:
    records = [
        parse_gkg_row(
            make_row(
                record_id="record-1",
                organizations="Apple Inc,100",
            )
        ),
        parse_gkg_row(
            make_row(
                record_id="record-2",
                organizations="Microsoft Corporation,100",
            )
        ),
        parse_gkg_row(
            make_row(
                record_id="record-3",
                all_names="AAPL,200",
            )
        ),
    ]

    selected = select_gkg_records(
        records,
        make_query(),
    )

    assert tuple(
        record.record_id
        for record in selected
    ) == (
        "record-1",
        "record-3",
    )


def test_select_gkg_records_preserves_source_order() -> None:
    records = [
        parse_gkg_row(
            make_row(
                record_id="record-3",
                all_names="AAPL,100",
            )
        ),
        parse_gkg_row(
            make_row(
                record_id="record-1",
                organizations="Apple Inc,100",
            )
        ),
        parse_gkg_row(
            make_row(
                record_id="record-2",
                all_names="AAPL,300",
            )
        ),
    ]

    selected = select_gkg_records(
        records,
        make_query(),
    )

    assert tuple(
        record.record_id
        for record in selected
    ) == (
        "record-3",
        "record-1",
        "record-2",
    )


def test_select_gkg_records_accepts_empty_input() -> None:
    assert select_gkg_records(
        [],
        make_query(),
    ) == []


def test_select_gkg_records_does_not_match_context_only() -> None:
    records = [
        parse_gkg_row(
            make_row(
                record_id="record-1",
                all_names="gold,100;commodity,200",
            )
        )
    ]

    query = NewsQuery(
        asset="GOLD",
        query_id="gold_market",
        strict_terms=(
            "gold price",
            "gold prices",
            "bullion",
            "XAUUSD",
        ),
        context_terms=(
            "gold",
            "XAU",
            "commodity",
            "futures",
        ),
    )

    assert select_gkg_records(
        records,
        query,
    ) == []


def test_select_gkg_records_rejects_non_record_item() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="GdeltGkgRecord",
    ):
        select_gkg_records(
            [
                parse_gkg_row(
                    make_row(
                        record_id="record-1",
                        all_names="AAPL,100",
                    )
                ),
                "not-a-record",  # type: ignore[list-item]
            ],
            make_query(),
        )


def test_select_gkg_records_rejects_non_query() -> None:
    records: list[GdeltGkgRecord] = [
        parse_gkg_row(
            make_row(
                record_id="record-1",
                all_names="AAPL,100",
            )
        )
    ]

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="NewsQuery",
    ):
        select_gkg_records(
            records,
            None,  # type: ignore[arg-type]
        )