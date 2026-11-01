from __future__ import annotations
from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from market_prediction.news.fingerprint import (
    article_identity,
)
from market_prediction.news.models import (
    PublishedAtPrecision,
    TimestampSource,
)
from market_prediction.providers.news.base import (
    NewsQuery,
)
from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
)
from market_prediction.providers.news.gdelt import (
    GdeltGkgRecord,
    gkg_record_to_candidate,
    parse_gkg_row,
)


UTC = timezone.utc


def make_record(
    *,
    record_id: str = "20150219000000-0",
    source: str = "example.com",
    url: str = "https://example.com/apple-results",
) -> GdeltGkgRecord:
    row = [""] * 27

    row[0] = record_id
    row[1] = "20150219000000"
    row[2] = "1"
    row[3] = source
    row[4] = url
    row[16] = "Apple Inc,100"

    return parse_gkg_row(row)


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


def retrieval_time() -> datetime:
    return datetime(
        2026,
        10,
        31,
        12,
        0,
        tzinfo=UTC,
    )


def test_gkg_record_to_candidate_maps_identity() -> None:
    record = make_record()

    candidate = gkg_record_to_candidate(
        record=record,
        query=make_query(),
        retrieved_at_utc=retrieval_time(),
    )

    assert candidate.candidate_id == article_identity(
        provider="gdelt",
        provider_item_id=record.record_id,
    )
    assert candidate.provider == "gdelt"
    assert (
        candidate.provider_item_id
        == record.record_id
    )


def test_gkg_record_to_candidate_maps_query() -> None:
    candidate = gkg_record_to_candidate(
        record=make_record(),
        query=make_query(),
        retrieved_at_utc=retrieval_time(),
    )

    assert candidate.asset == "AAPL"
    assert candidate.query_id == "aapl_company"


def test_gkg_record_to_candidate_maps_source_metadata() -> None:
    candidate = gkg_record_to_candidate(
        record=make_record(
            source="reuters.com",
            url=(
                "https://reuters.com/"
                "markets/apple-results"
            ),
        ),
        query=make_query(),
        retrieved_at_utc=retrieval_time(),
    )

    assert candidate.publisher == "reuters.com"
    assert candidate.canonical_url == (
        "https://reuters.com/"
        "markets/apple-results"
    )


def test_gkg_record_to_candidate_uses_gkg_timestamp_conservatively() -> None:
    record = make_record()

    candidate = gkg_record_to_candidate(
        record=record,
        query=make_query(),
        retrieved_at_utc=retrieval_time(),
    )

    assert (
        candidate.published_at_utc
        == record.published_at_utc
    )
    assert (
        candidate.provider_seen_at_utc
        == record.published_at_utc
    )
    assert (
        candidate.published_at_precision
        is PublishedAtPrecision.SECOND
    )
    assert (
        candidate.timestamp_source
        is TimestampSource.PROVIDER_SEEN
    )


def test_gkg_record_to_candidate_preserves_retrieval_time() -> None:
    retrieved = retrieval_time()

    candidate = gkg_record_to_candidate(
        record=make_record(),
        query=make_query(),
        retrieved_at_utc=retrieved,
    )

    assert candidate.retrieved_at_utc == retrieved


def test_gkg_record_to_candidate_rejects_non_matching_record() -> None:
    record = make_record()

    query = NewsQuery(
        asset="MSFT",
        query_id="msft_company",
        strict_terms=(
            "MSFT",
            "Microsoft",
            "Microsoft Corporation",
        ),
        context_terms=(),
    )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="does not match",
    ):
        gkg_record_to_candidate(
            record=record,
            query=query,
            retrieved_at_utc=retrieval_time(),
        )


def test_gkg_record_to_candidate_rejects_non_record() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="GdeltGkgRecord",
    ):
        gkg_record_to_candidate(
            record=None,  # type: ignore[arg-type]
            query=make_query(),
            retrieved_at_utc=retrieval_time(),
        )


def test_gkg_record_to_candidate_rejects_non_query() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="NewsQuery",
    ):
        gkg_record_to_candidate(
            record=make_record(),
            query=None,  # type: ignore[arg-type]
            retrieved_at_utc=retrieval_time(),
        )


def test_gkg_record_to_candidate_rejects_naive_retrieval_time() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="retrieved_at_utc",
    ):
        gkg_record_to_candidate(
            record=make_record(),
            query=make_query(),
            retrieved_at_utc=datetime(
                2026,
                10,
                31,
                12,
                0,
            ),
        )


def test_gkg_record_to_candidate_rejects_non_utc_retrieval_time() -> None:
    non_utc = timezone(
        timedelta(hours=9)
    )

    with pytest.raises(
        InvalidNewsProviderDataError,
        match="retrieved_at_utc",
    ):
        gkg_record_to_candidate(
            record=make_record(),
            query=make_query(),
            retrieved_at_utc=datetime(
                2026,
                10,
                31,
                21,
                0,
                tzinfo=non_utc,
            ),
        )


def test_gkg_record_to_candidate_rejects_retrieval_before_gkg_time() -> None:
    with pytest.raises(
        InvalidNewsProviderDataError,
        match="before",
    ):
        gkg_record_to_candidate(
            record=make_record(),
            query=make_query(),
            retrieved_at_utc=datetime(
                2015,
                2,
                18,
                tzinfo=UTC,
            ),
        )