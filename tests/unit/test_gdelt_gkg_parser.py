from __future__ import annotations

from datetime import datetime, timezone

import pytest

from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
)
from market_prediction.providers.news.gdelt import (
    GdeltGkgRecord,
    parse_gkg_row,
)


def make_row() -> list[str]:
    """Return a minimal structurally valid 27-column GKG row."""

    row = [""] * 27

    row[0] = "20150219000000-0"
    row[1] = "20150219000000"
    row[2] = "1"
    row[3] = "example.com"
    row[4] = "https://example.com/article"

    return row


def test_parse_gkg_row_extracts_canonical_source_fields() -> None:
    record = parse_gkg_row(make_row())

    assert isinstance(record, GdeltGkgRecord)

    assert record.record_id == "20150219000000-0"

    assert record.published_at_utc == datetime(
        2015,
        2,
        19,
        0,
        0,
        tzinfo=timezone.utc,
    )

    assert record.source_collection_identifier == "1"
    assert record.source_common_name == "example.com"

    assert (
        record.document_identifier
        == "https://example.com/article"
    )


def test_parse_gkg_row_preserves_raw_fields() -> None:
    row = make_row()
    row[5] = "theme-data"
    row[6] = "location-data"

    record = parse_gkg_row(row)

    assert record.raw_fields == tuple(row)


def test_parse_gkg_row_rejects_wrong_column_count() -> None:
    row = make_row()[:-1]

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="27",
    ):
        parse_gkg_row(row)


def test_parse_gkg_row_rejects_empty_record_id() -> None:
    row = make_row()
    row[0] = ""

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="record",
    ):
        parse_gkg_row(row)


def test_parse_gkg_row_rejects_empty_gkg_date() -> None:
    row = make_row()
    row[1] = ""

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="date",
    ):
        parse_gkg_row(row)


def test_parse_gkg_row_rejects_malformed_gkg_date() -> None:
    row = make_row()
    row[1] = "2015-02-19"

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="date",
    ):
        parse_gkg_row(row)


def test_parse_gkg_row_rejects_impossible_gkg_date() -> None:
    row = make_row()
    row[1] = "20150230000000"

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="date",
    ):
        parse_gkg_row(row)


def test_parse_gkg_row_rejects_empty_source_name() -> None:
    row = make_row()
    row[3] = ""

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="source",
    ):
        parse_gkg_row(row)


def test_parse_gkg_row_rejects_empty_document_identifier() -> None:
    row = make_row()
    row[4] = ""

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="document",
    ):
        parse_gkg_row(row)


def test_parse_gkg_row_rejects_non_http_document_identifier() -> None:
    row = make_row()
    row[4] = "not-a-url"

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="document",
    ):
        parse_gkg_row(row)


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com/article",
        "https://example.com/article",
    ],
)
def test_parse_gkg_row_accepts_http_and_https_document_identifiers(
    value: str,
) -> None:
    row = make_row()
    row[4] = value

    record = parse_gkg_row(row)

    assert record.document_identifier == value

def test_parse_gkg_row_extracts_matching_fields() -> None:
    row = make_row()

    # GKG 2.1:
    # 8  = V2ENHANCEDTHEMES
    # 14 = V2ENHANCEDPERSONS
    # 16 = V2ENHANCEDORGANIZATIONS
    # 23 = V2.1ALLNAMES
    row[8] = "ECON_STOCKMARKET,20;ECON_EARNINGS,45"
    row[14] = "Tim Cook,100"
    row[16] = "Apple Inc,10;Nasdaq,200"
    row[23] = "Apple Inc,10;iPhone,300"

    record = parse_gkg_row(row)

    assert record.enhanced_themes == (
        "ECON_STOCKMARKET,20;ECON_EARNINGS,45"
    )
    assert record.enhanced_persons == "Tim Cook,100"
    assert record.enhanced_organizations == (
        "Apple Inc,10;Nasdaq,200"
    )
    assert record.all_names == (
        "Apple Inc,10;iPhone,300"
    )


def test_parse_gkg_row_preserves_empty_matching_fields() -> None:
    record = parse_gkg_row(make_row())

    assert record.enhanced_themes == ""
    assert record.enhanced_persons == ""
    assert record.enhanced_organizations == ""
    assert record.all_names == ""