from __future__ import annotations

import pytest

from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
)
from market_prediction.providers.news.gdelt import (
    parse_gkg_name_offsets,
)


def test_parse_gkg_name_offsets_extracts_names() -> None:
    assert parse_gkg_name_offsets(
        "Apple Inc,10;Nasdaq,200"
    ) == (
        "Apple Inc",
        "Nasdaq",
    )


def test_parse_gkg_name_offsets_preserves_source_order() -> None:
    assert parse_gkg_name_offsets(
        "Microsoft Corporation,400;"
        "Microsoft,20;"
        "Azure,900"
    ) == (
        "Microsoft Corporation",
        "Microsoft",
        "Azure",
    )


def test_parse_gkg_name_offsets_accepts_empty_field() -> None:
    assert parse_gkg_name_offsets("") == ()


def test_parse_gkg_name_offsets_rejects_whitespace_only_field() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="whitespace",
    ):
        parse_gkg_name_offsets("   ")


def test_parse_gkg_name_offsets_rejects_missing_offset() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="offset",
    ):
        parse_gkg_name_offsets(
            "Apple Inc"
        )


def test_parse_gkg_name_offsets_rejects_empty_name() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="name",
    ):
        parse_gkg_name_offsets(
            ",10"
        )


def test_parse_gkg_name_offsets_rejects_empty_offset() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="offset",
    ):
        parse_gkg_name_offsets(
            "Apple Inc,"
        )


def test_parse_gkg_name_offsets_rejects_non_integer_offset() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="offset",
    ):
        parse_gkg_name_offsets(
            "Apple Inc,ten"
        )


def test_parse_gkg_name_offsets_rejects_negative_offset() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="offset",
    ):
        parse_gkg_name_offsets(
            "Apple Inc,-1"
        )


def test_parse_gkg_name_offsets_rejects_empty_entry() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="entry",
    ):
        parse_gkg_name_offsets(
            "Apple Inc,10;;Nasdaq,200"
        )


def test_parse_gkg_name_offsets_preserves_duplicate_names() -> None:
    assert parse_gkg_name_offsets(
        "Apple Inc,10;Apple Inc,500"
    ) == (
        "Apple Inc",
        "Apple Inc",
    )


def test_parse_gkg_name_offsets_rejects_non_string() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="string",
    ):
        parse_gkg_name_offsets(
            None  # type: ignore[arg-type]
        )