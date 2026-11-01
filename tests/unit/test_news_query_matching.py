from __future__ import annotations

import pytest

from market_prediction.providers.news.base import NewsQuery
from market_prediction.providers.news.matching import (
    MatchKind,
    QueryMatch,
    contains_term,
    match_query,
    normalize_search_text,
)


def make_query(
    *,
    strict_terms: tuple[str, ...] = ("Apple Inc", "AAPL"),
    context_terms: tuple[str, ...] = (),
) -> NewsQuery:
    return NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=strict_terms,
        context_terms=context_terms,
    )


def test_normalize_search_text_is_case_insensitive() -> None:
    assert (
        normalize_search_text("Apple INC")
        == normalize_search_text("apple inc")
    )


def test_normalize_search_text_collapses_whitespace() -> None:
    assert (
        normalize_search_text("Apple   \n\t Inc")
        == "apple inc"
    )


@pytest.mark.parametrize(
    ("text", "term"),
    [
        ("Apple Inc reported earnings", "Apple Inc"),
        ("AAPL shares rose", "AAPL"),
        ("Bitcoin rallied", "Bitcoin"),
        ("EUR/USD moved higher", "EUR/USD"),
        ("EURUSD moved higher", "EURUSD"),
        ("gold price increased", "gold price"),
        ("XAUUSD moved lower", "XAUUSD"),
    ],
)
def test_contains_term_matches_complete_terms(
    text: str,
    term: str,
) -> None:
    assert contains_term(text, term) is True


@pytest.mark.parametrize(
    ("text", "term"),
    [
        ("pineapple sales increased", "apple"),
        ("BTCCOIN moved higher", "BTC"),
        ("XAUSDT moved higher", "XAU"),
        ("microsoftware", "Microsoft"),
    ],
)
def test_contains_term_rejects_substring_matches(
    text: str,
    term: str,
) -> None:
    assert contains_term(text, term) is False


def test_match_query_returns_strict_match() -> None:
    query = make_query()

    result = match_query(
        "Apple Inc reported quarterly earnings",
        query,
    )

    assert result == QueryMatch(
        matched=True,
        kind=MatchKind.STRICT,
        matched_terms=("Apple Inc",),
    )


def test_match_query_strict_match_is_case_insensitive() -> None:
    query = make_query()

    result = match_query(
        "APPLE INC reported quarterly earnings",
        query,
    )

    assert result.matched is True
    assert result.kind is MatchKind.STRICT
    assert result.matched_terms == ("Apple Inc",)


def test_match_query_preserves_query_term_order() -> None:
    query = make_query(
        strict_terms=(
            "AAPL",
            "Apple Inc",
        ),
    )

    result = match_query(
        "Apple Inc (AAPL) reported earnings",
        query,
    )

    assert result.matched_terms == (
        "AAPL",
        "Apple Inc",
    )


def test_context_term_alone_does_not_match() -> None:
    query = make_query(
        strict_terms=("Bitcoin",),
        context_terms=(
            "BTC",
            "crypto",
            "cryptocurrency",
        ),
    )

    result = match_query(
        "BTC moved sharply higher",
        query,
    )

    assert result == QueryMatch(
        matched=False,
        kind=MatchKind.NONE,
        matched_terms=(),
    )


def test_multiple_context_terms_alone_do_not_match() -> None:
    query = make_query(
        strict_terms=("EUR/USD", "EURUSD"),
        context_terms=(
            "euro",
            "dollar",
            "USD",
            "forex",
            "currency",
        ),
    )

    result = match_query(
        "The euro gained against the dollar "
        "in foreign exchange trading",
        query,
    )

    assert result.matched is False
    assert result.kind is MatchKind.NONE
    assert result.matched_terms == ()


def test_bare_gold_context_does_not_match() -> None:
    query = make_query(
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

    result = match_query(
        "The museum displayed an ancient gold necklace",
        query,
    )

    assert result.matched is False
    assert result.kind is MatchKind.NONE


def test_gold_price_strict_term_matches() -> None:
    query = make_query(
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

    result = match_query(
        "Gold price rises as markets react",
        query,
    )

    assert result.matched is True
    assert result.kind is MatchKind.STRICT
    assert result.matched_terms == ("gold price",)


def test_empty_search_text_returns_no_match() -> None:
    query = make_query()

    result = match_query(
        "",
        query,
    )

    assert result == QueryMatch(
        matched=False,
        kind=MatchKind.NONE,
        matched_terms=(),
    )


def test_whitespace_search_text_returns_no_match() -> None:
    query = make_query()

    result = match_query(
        "   \n\t ",
        query,
    )

    assert result.matched is False


def test_match_query_rejects_non_string_search_text() -> None:
    query = make_query()

    with pytest.raises(
        ValueError,
        match="searchable_text",
    ):
        match_query(
            None,  # type: ignore[arg-type]
            query,
        )