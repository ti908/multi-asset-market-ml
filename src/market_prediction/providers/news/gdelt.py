"""GDELT GKG 2.0 parsing primitives for news ingestion.

This module contains provider-specific parsing logic for individual
GDELT GKG 2.0 rows.

It intentionally does not perform HTTP requests, archive discovery,
ZIP decoding, asset matching, deduplication, storage, sentiment
scoring, or market-data alignment.
"""
from __future__ import annotations
import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
)
from market_prediction.providers.news.base import NewsQuery
from market_prediction.providers.news.matching import (
    MatchKind,
    QueryMatch,
    contains_term,
)
from market_prediction.news.candidates import (
    NewsCandidate,
)
from market_prediction.news.fingerprint import (
    article_identity,
)
from market_prediction.news.models import (
    PublishedAtPrecision,
    TimestampSource,
)

GKG_COLUMN_COUNT = 27
GKG_DATE_FORMAT = "%Y%m%d%H%M%S"


@dataclass(frozen=True)
class GdeltGkgRecord:
    """Provider-specific representation of one GDELT GKG 2.0 row."""

    record_id: str
    published_at_utc: datetime
    source_collection_identifier: str
    source_common_name: str
    document_identifier: str

    enhanced_themes: str
    enhanced_persons: str
    enhanced_organizations: str
    all_names: str

    raw_fields: tuple[str, ...]


def _require_non_empty(
    value: str,
    *,
    field_name: str,
) -> str:
    """Require a non-empty GKG field."""

    if not isinstance(value, str) or not value.strip():
        raise InvalidNewsProviderDataError(
            f"GDELT GKG {field_name} must be non-empty"
        )

    return value


def _parse_gkg_date(value: str) -> datetime:
    """Parse the GKG V2.1DATE field as an aware UTC datetime."""

    _require_non_empty(
        value,
        field_name="date",
    )

    try:
        parsed = datetime.strptime(
            value,
            GKG_DATE_FORMAT,
        )
    except ValueError as exc:
        raise InvalidNewsProviderDataError(
            f"Invalid GDELT GKG date: {value!r}"
        ) from exc

    return parsed.replace(
        tzinfo=timezone.utc,
    )


def _validate_document_identifier(
    value: str,
) -> str:
    """Require an HTTP(S) GKG document identifier."""

    _require_non_empty(
        value,
        field_name="document identifier",
    )

    parsed = urlparse(value)

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
    ):
        raise InvalidNewsProviderDataError(
            "GDELT GKG document identifier must be "
            "an absolute HTTP(S) URL"
        )

    return value


def parse_gkg_row(
    row: list[str] | tuple[str, ...],
) -> GdeltGkgRecord:
    """Parse one canonical 27-column GDELT GKG 2.0 row.

    The first five GKG 2.0 fields are interpreted as:

    0. GKGRECORDID
    1. V2.1DATE
    2. V2SOURCECOLLECTIONIDENTIFIER
    3. V2SOURCECOMMONNAME
    4. V2DOCUMENTIDENTIFIER

    Rows with unexpected widths or invalid required fields are rejected
    rather than repaired, truncated, or silently skipped.
    """

    if not isinstance(row, (list, tuple)):
        raise InvalidNewsProviderDataError(
            "GDELT GKG row must be a list or tuple"
        )

    if len(row) != GKG_COLUMN_COUNT:
        raise InvalidNewsProviderDataError(
            "GDELT GKG row must contain exactly "
            f"{GKG_COLUMN_COUNT} columns; "
            f"received {len(row)}"
        )

    record_id = _require_non_empty(
        row[0],
        field_name="record id",
    )

    published_at_utc = _parse_gkg_date(
        row[1],
    )

    source_collection_identifier = row[2]

    source_common_name = _require_non_empty(
        row[3],
        field_name="source common name",
    )

    document_identifier = _validate_document_identifier(
        row[4],
    )

    return GdeltGkgRecord(
        record_id=record_id,
        published_at_utc=published_at_utc,
        source_collection_identifier=(
            source_collection_identifier
        ),
        source_common_name=source_common_name,
        document_identifier=document_identifier,
        enhanced_themes=row[8],
        enhanced_persons=row[14],
        enhanced_organizations=row[16],
        all_names=row[23],
        raw_fields=tuple(row),
    )

def parse_gkg_archive(
    payload: bytes,
) -> list[GdeltGkgRecord]:
    """Decode and strictly parse one GDELT GKG 2.0 ZIP archive.

    The archive must contain exactly one member. Its contents must be
    valid UTF-8 tab-separated GKG rows.

    Any malformed row blocks the archive. Records are never silently
    repaired, truncated, skipped, or reordered.
    """

    if not isinstance(payload, bytes) or not payload:
        raise InvalidNewsProviderDataError(
            "GDELT GKG archive payload must be non-empty bytes"
        )

    try:
        archive = zipfile.ZipFile(
            io.BytesIO(payload)
        )
    except (zipfile.BadZipFile, OSError) as exc:
        raise InvalidNewsProviderDataError(
            "GDELT GKG payload is not a valid ZIP archive"
        ) from exc

    with archive:
        names = archive.namelist()

        if len(names) != 1:
            raise InvalidNewsProviderDataError(
                "GDELT GKG ZIP must contain exactly one member; "
                f"found {len(names)}"
            )

        member_name = names[0]

        try:
            member_payload = archive.read(
                member_name
            )
        except (
            KeyError,
            RuntimeError,
            zipfile.BadZipFile,
            OSError,
        ) as exc:
            raise InvalidNewsProviderDataError(
                "Unable to read GDELT GKG ZIP member"
            ) from exc

    try:
        text = member_payload.decode(
            "utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise InvalidNewsProviderDataError(
            "GDELT GKG archive member is not valid UTF-8"
        ) from exc

    rows = list(
        csv.reader(
            io.StringIO(text),
            delimiter="\t",
        )
    )

    if not rows:
        raise InvalidNewsProviderDataError(
            "GDELT GKG archive contained no rows"
        )

    records: list[GdeltGkgRecord] = []

    for row in rows:
        records.append(
            parse_gkg_row(row)
        )

    return records

def parse_gkg_name_offsets(
    value: str,
) -> tuple[str, ...]:
    """Parse a semicolon-delimited GKG name/offset field.

    GKG name fields encode entries as:

        name,character_offset;name,character_offset

    Character offsets are validated but deliberately excluded from the
    returned searchable names.
    """

    if not isinstance(value, str):
        raise InvalidNewsProviderDataError(
            "GDELT GKG name/offset field must be a string"
        )

    if value == "":
        return ()

    if not value.strip():
        raise InvalidNewsProviderDataError(
            "GDELT GKG name/offset field cannot be whitespace only"
        )

    names: list[str] = []

    for entry in value.split(";"):
        if not entry:
            raise InvalidNewsProviderDataError(
                "GDELT GKG name/offset field contains an empty entry"
            )

        try:
            name, offset_text = entry.rsplit(
                ",",
                1,
            )
        except ValueError as exc:
            raise InvalidNewsProviderDataError(
                "GDELT GKG name/offset entry is missing an offset"
            ) from exc

        if not name.strip():
            raise InvalidNewsProviderDataError(
                "GDELT GKG name/offset entry has an empty name"
            )

        if not offset_text:
            raise InvalidNewsProviderDataError(
                "GDELT GKG name/offset entry has an empty offset"
            )

        try:
            offset = int(offset_text)
        except ValueError as exc:
            raise InvalidNewsProviderDataError(
                "GDELT GKG name/offset entry has an invalid offset"
            ) from exc

        if offset < 0:
            raise InvalidNewsProviderDataError(
                "GDELT GKG name/offset entry has a negative offset"
            )

        names.append(
            name.strip()
        )

    return tuple(names)

def match_gkg_record(
    record: GdeltGkgRecord,
    query: NewsQuery,
) -> QueryMatch:
    """Match a GKG record against a news query.

    Only provider fields representing extracted organization names and
    general named entities participate in production relevance matching.

    Source names, document URLs, themes, persons, and unrelated raw GKG
    fields are deliberately excluded.
    """

    if not isinstance(record, GdeltGkgRecord):
        raise InvalidNewsProviderDataError(
            "GDELT GKG record must be a GdeltGkgRecord"
        )

    if not isinstance(query, NewsQuery):
        raise InvalidNewsProviderDataError(
            "GDELT GKG query must be a NewsQuery"
        )

    organizations = parse_gkg_name_offsets(
        record.enhanced_organizations
    )
    all_names = parse_gkg_name_offsets(
        record.all_names
    )

    searchable_names = (
        organizations
        + all_names
    )

    matched_terms = tuple(
        term
        for term in query.strict_terms
        if any(
            contains_term(name, term)
            for name in searchable_names
        )
    )

    if not matched_terms:
        return QueryMatch(
            matched=False,
            kind=MatchKind.NONE,
            matched_terms=(),
        )

    return QueryMatch(
        matched=True,
        kind=MatchKind.STRICT,
        matched_terms=matched_terms,
    )

def select_gkg_records(
    records: list[GdeltGkgRecord],
    query: NewsQuery,
) -> list[GdeltGkgRecord]:
    """Return GKG records that satisfy the configured strict query.

    Input order is preserved. This function performs relevance
    selection only; it does not deduplicate, fetch article content,
    perform sentiment scoring, or mutate provider data.
    """

    if not isinstance(records, list):
        raise InvalidNewsProviderDataError(
            "GDELT GKG records must be provided as a list"
        )

    if not isinstance(query, NewsQuery):
        raise InvalidNewsProviderDataError(
            "GDELT GKG query must be a NewsQuery"
        )

    selected: list[GdeltGkgRecord] = []

    for record in records:
        if not isinstance(record, GdeltGkgRecord):
            raise InvalidNewsProviderDataError(
                "GDELT GKG records must contain only "
                "GdeltGkgRecord instances"
            )

        match = match_gkg_record(
            record,
            query,
        )

        if match.matched:
            selected.append(record)

    return selected

def gkg_record_to_candidate(
    *,
    record: GdeltGkgRecord,
    query: NewsQuery,
    retrieved_at_utc: datetime,
) -> NewsCandidate:
    """Convert a matched GKG record into a discovery candidate.

    The GKG timestamp is treated conservatively as provider-seen time.
    It must not be interpreted as independently verified publisher
    publication time.
    """

    if not isinstance(record, GdeltGkgRecord):
        raise InvalidNewsProviderDataError(
            "GDELT candidate source must be a GdeltGkgRecord"
        )

    if not isinstance(query, NewsQuery):
        raise InvalidNewsProviderDataError(
            "GDELT candidate query must be a NewsQuery"
        )

    if (
        not isinstance(retrieved_at_utc, datetime)
        or retrieved_at_utc.tzinfo is None
        or retrieved_at_utc.utcoffset() is None
        or retrieved_at_utc.utcoffset().total_seconds() != 0
    ):
        raise InvalidNewsProviderDataError(
            "retrieved_at_utc must be a timezone-aware UTC datetime"
        )

    if retrieved_at_utc < record.published_at_utc:
        raise InvalidNewsProviderDataError(
            "retrieved_at_utc cannot be before the GDELT provider time"
        )

    match = match_gkg_record(
        record,
        query,
    )

    if not match.matched:
        raise InvalidNewsProviderDataError(
            "GDELT GKG record does not match the supplied NewsQuery"
        )

    return NewsCandidate(
        candidate_id=article_identity(
            provider="gdelt",
            provider_item_id=record.record_id,
        ),
        asset=query.asset,
        query_id=query.query_id,
        canonical_url=record.document_identifier,
        publisher=record.source_common_name,
        published_at_utc=record.published_at_utc,
        published_at_precision=(
            PublishedAtPrecision.SECOND
        ),
        provider_seen_at_utc=record.published_at_utc,
        retrieved_at_utc=retrieved_at_utc,
        provider="gdelt",
        provider_item_id=record.record_id,
        timestamp_source=(
            TimestampSource.PROVIDER_SEEN
        ),
    )

