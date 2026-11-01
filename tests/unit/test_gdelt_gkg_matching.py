from __future__ import annotations

from market_prediction.providers.news.base import NewsQuery
from market_prediction.providers.news.gdelt import (
    match_gkg_record,
    parse_gkg_row,
)
from market_prediction.providers.news.matching import (
    MatchKind,
)


def make_row() -> list[str]:
    """Return one structurally valid GKG row."""

    row = [""] * 27

    row[0] = "20150219000000-0"
    row[1] = "20150219000000"
    row[2] = "1"
    row[3] = "example.com"
    row[4] = "https://example.com/article"

    return row


def make_query(
    *,
    asset: str = "AAPL",
    query_id: str = "aapl_company",
    strict_terms: tuple[str, ...] = (
        "AAPL",
        "Apple Inc",
    ),
    context_terms: tuple[str, ...] = (),
) -> NewsQuery:
    return NewsQuery(
        asset=asset,
        query_id=query_id,
        strict_terms=strict_terms,
        context_terms=context_terms,
    )


def test_match_gkg_record_matches_enhanced_organization() -> None:
    row = make_row()
    row[16] = "Apple Inc,100;Nasdaq,300"

    result = match_gkg_record(
        parse_gkg_row(row),
        make_query(),
    )

    assert result.matched is True
    assert result.kind is MatchKind.STRICT
    assert result.matched_terms == ("Apple Inc",)


def test_match_gkg_record_matches_all_names() -> None:
    row = make_row()
    row[23] = "AAPL,100;iPhone,300"

    result = match_gkg_record(
        parse_gkg_row(row),
        make_query(),
    )

    assert result.matched is True
    assert result.kind is MatchKind.STRICT
    assert result.matched_terms == ("AAPL",)


def test_match_gkg_record_combines_supported_fields() -> None:
    row = make_row()

    row[16] = "Apple Inc,100"
    row[23] = "AAPL,300"

    result = match_gkg_record(
        parse_gkg_row(row),
        make_query(),
    )

    assert result.matched is True

    # Query configuration order remains authoritative.
    assert result.matched_terms == (
        "AAPL",
        "Apple Inc",
    )


def test_match_gkg_record_does_not_search_source_name() -> None:
    row = make_row()
    row[3] = "apple-inc-news.example.com"

    result = match_gkg_record(
        parse_gkg_row(row),
        make_query(),
    )

    assert result.matched is False
    assert result.kind is MatchKind.NONE


def test_match_gkg_record_does_not_search_document_url() -> None:
    row = make_row()
    row[4] = (
        "https://example.com/"
        "apple-inc-quarterly-results"
    )

    result = match_gkg_record(
        parse_gkg_row(row),
        make_query(),
    )

    assert result.matched is False
    assert result.kind is MatchKind.NONE


def test_match_gkg_record_does_not_search_themes() -> None:
    row = make_row()

    # Even if an arbitrary theme-like value contains an asset term,
    # themes are not part of the production name-matching surface.
    row[8] = "APPLE_INC"

    result = match_gkg_record(
        parse_gkg_row(row),
        make_query(),
    )

    assert result.matched is False
    assert result.kind is MatchKind.NONE


def test_match_gkg_record_does_not_search_persons() -> None:
    row = make_row()
    row[14] = "Apple Inc,100"

    result = match_gkg_record(
        parse_gkg_row(row),
        make_query(),
    )

    assert result.matched is False
    assert result.kind is MatchKind.NONE


def test_match_gkg_record_does_not_search_unrelated_raw_field() -> None:
    row = make_row()

    # This reproduces the class of Step 4a false-context problem:
    # an asset term elsewhere in the 27-column record must not cause
    # a production relevance match.
    row[20] = "Apple Inc"

    result = match_gkg_record(
        parse_gkg_row(row),
        make_query(),
    )

    assert result.matched is False
    assert result.kind is MatchKind.NONE


def test_context_only_gkg_name_does_not_match() -> None:
    row = make_row()
    row[23] = "BTC,100;crypto,200"

    query = make_query(
        asset="BTC_USD",
        query_id="btc_asset",
        strict_terms=("Bitcoin",),
        context_terms=(
            "BTC",
            "crypto",
            "cryptocurrency",
        ),
    )

    result = match_gkg_record(
        parse_gkg_row(row),
        query,
    )

    assert result.matched is False
    assert result.kind is MatchKind.NONE


def test_bitcoin_strict_name_matches() -> None:
    row = make_row()
    row[23] = "Bitcoin,100"

    query = make_query(
        asset="BTC_USD",
        query_id="btc_asset",
        strict_terms=("Bitcoin",),
        context_terms=(
            "BTC",
            "crypto",
            "cryptocurrency",
        ),
    )

    result = match_gkg_record(
        parse_gkg_row(row),
        query,
    )

    assert result.matched is True
    assert result.matched_terms == ("Bitcoin",)


def test_bare_gold_context_name_does_not_match() -> None:
    row = make_row()
    row[23] = "gold,100;commodity,200"

    query = make_query(
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

    result = match_gkg_record(
        parse_gkg_row(row),
        query,
    )

    assert result.matched is False


def test_gold_price_name_matches() -> None:
    row = make_row()
    row[23] = "gold price,100"

    query = make_query(
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

    result = match_gkg_record(
        parse_gkg_row(row),
        query,
    )

    assert result.matched is True
    assert result.matched_terms == ("gold price",)


def test_eurusd_strict_name_matches() -> None:
    row = make_row()
    row[23] = "EUR/USD,100"

    query = make_query(
        asset="EURUSD",
        query_id="eurusd_pair",
        strict_terms=(
            "EUR/USD",
            "EURUSD",
        ),
        context_terms=(
            "euro",
            "dollar",
            "USD",
            "forex",
            "currency",
        ),
    )

    result = match_gkg_record(
        parse_gkg_row(row),
        query,
    )

    assert result.matched is True
    assert result.matched_terms == ("EUR/USD",)


def test_eurusd_context_names_alone_do_not_match() -> None:
    row = make_row()
    row[23] = "euro,100;dollar,200;forex,300"

    query = make_query(
        asset="EURUSD",
        query_id="eurusd_pair",
        strict_terms=(
            "EUR/USD",
            "EURUSD",
        ),
        context_terms=(
            "euro",
            "dollar",
            "USD",
            "forex",
            "currency",
        ),
    )

    result = match_gkg_record(
        parse_gkg_row(row),
        query,
    )

    assert result.matched is False
    assert result.kind is MatchKind.NONE