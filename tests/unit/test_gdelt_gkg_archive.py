from __future__ import annotations

import io
import zipfile

import pytest

from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
)
from market_prediction.providers.news.gdelt import (
    GdeltGkgRecord,
    parse_gkg_archive,
)


def make_row(
    *,
    record_id: str = "20150219000000-0",
    date: str = "20150219000000",
    source: str = "example.com",
    url: str = "https://example.com/article",
) -> list[str]:
    """Return one structurally valid 27-column GKG row."""

    row = [""] * 27

    row[0] = record_id
    row[1] = date
    row[2] = "1"
    row[3] = source
    row[4] = url

    return row


def make_zip(
    rows: list[list[str]],
    *,
    member_name: str = "sample.gkg.csv",
) -> bytes:
    """Build an in-memory ZIP containing tab-separated GKG rows."""

    text = "\n".join(
        "\t".join(row)
        for row in rows
    )

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            member_name,
            text.encode("utf-8"),
        )

    return buffer.getvalue()


def test_parse_gkg_archive_returns_records_in_source_order() -> None:
    payload = make_zip(
        [
            make_row(
                record_id="20150219000000-0",
                url="https://example.com/first",
            ),
            make_row(
                record_id="20150219000000-1",
                url="https://example.com/second",
            ),
        ]
    )

    records = parse_gkg_archive(payload)

    assert len(records) == 2

    assert all(
        isinstance(record, GdeltGkgRecord)
        for record in records
    )

    assert [
        record.record_id
        for record in records
    ] == [
        "20150219000000-0",
        "20150219000000-1",
    ]


def test_parse_gkg_archive_rejects_empty_payload() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="empty",
    ):
        parse_gkg_archive(b"")


def test_parse_gkg_archive_rejects_non_zip_payload() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="ZIP",
    ):
        parse_gkg_archive(b"not a zip archive")


def test_parse_gkg_archive_rejects_empty_zip() -> None:
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
    ):
        pass

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="exactly one",
    ):
        parse_gkg_archive(buffer.getvalue())


def test_parse_gkg_archive_rejects_multiple_members() -> None:
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
    ) as archive:
        archive.writestr(
            "first.gkg.csv",
            b"",
        )
        archive.writestr(
            "second.gkg.csv",
            b"",
        )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="exactly one",
    ):
        parse_gkg_archive(buffer.getvalue())


def test_parse_gkg_archive_rejects_empty_member() -> None:
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
    ) as archive:
        archive.writestr(
            "sample.gkg.csv",
            b"",
        )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="no rows",
    ):
        parse_gkg_archive(buffer.getvalue())


def test_parse_gkg_archive_rejects_malformed_width_row() -> None:
    valid = make_row()

    malformed = make_row(
        record_id="20150219000000-1",
    )[:-1]

    payload = make_zip(
        [
            valid,
            malformed,
        ]
    )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="27",
    ):
        parse_gkg_archive(payload)


def test_parse_gkg_archive_rejects_invalid_required_field() -> None:
    invalid = make_row()
    invalid[3] = ""

    payload = make_zip(
        [invalid]
    )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="source",
    ):
        parse_gkg_archive(payload)


def test_parse_gkg_archive_rejects_invalid_utf8() -> None:
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
    ) as archive:
        archive.writestr(
            "sample.gkg.csv",
            b"\xff\xfe\xfa",
        )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="UTF-8",
    ):
        parse_gkg_archive(buffer.getvalue())