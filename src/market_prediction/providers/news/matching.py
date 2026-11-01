"""Provider-neutral deterministic matching for market-news queries.

This module evaluates canonical NewsQuery definitions against explicitly
selected searchable text.

It intentionally does not decide which provider-specific fields are
searchable. Concrete providers remain responsible for selecting
defensible text fields before calling these matching primitives.

Context terms are supporting metadata only and do not independently
create a match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from market_prediction.providers.news.base import NewsQuery


class MatchKind(str, Enum):
    """Classification of a deterministic query match."""

    NONE = "none"
    STRICT = "strict"


@dataclass(frozen=True)
class QueryMatch:
    """Result of evaluating one NewsQuery against searchable text."""

    matched: bool
    kind: MatchKind
    matched_terms: tuple[str, ...]


def normalize_search_text(
    value: str,
) -> str:
    """Normalize searchable text for deterministic comparison.

    Matching is case-insensitive and whitespace-insensitive. Punctuation
    is otherwise preserved so that terms such as EUR/USD remain
    distinguishable from unrelated text.
    """

    if not isinstance(value, str):
        raise ValueError(
            "searchable_text must be a string"
        )

    return re.sub(
        r"\s+",
        " ",
        value.casefold(),
    ).strip()


def contains_term(
    text: str,
    term: str,
) -> bool:
    """Return whether text contains one complete query term.

    Alphanumeric boundaries prevent short terms such as BTC or XAU from
    matching inside larger unrelated tokens.
    """

    if not isinstance(text, str):
        raise ValueError(
            "searchable_text must be a string"
        )

    if not isinstance(term, str) or not term.strip():
        raise ValueError(
            "term must be a non-empty string"
        )

    normalized_text = normalize_search_text(
        text
    )

    normalized_term = normalize_search_text(
        term
    )

    if not normalized_text:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(normalized_term)
        + r"(?![a-z0-9])"
    )

    return (
        re.search(
            pattern,
            normalized_text,
        )
        is not None
    )


def match_query(
    searchable_text: str,
    query: NewsQuery,
) -> QueryMatch:
    """Match explicit searchable text against one canonical NewsQuery.

    A strict term is sufficient to create a match.

    Context terms deliberately do not create matches by themselves.
    Broad/ambiguous matching must be represented explicitly in
    configuration before it can become production behavior.
    """

    if not isinstance(searchable_text, str):
        raise ValueError(
            "searchable_text must be a string"
        )

    if not isinstance(query, NewsQuery):
        raise ValueError(
            "query must be a NewsQuery"
        )

    if not normalize_search_text(
        searchable_text
    ):
        return QueryMatch(
            matched=False,
            kind=MatchKind.NONE,
            matched_terms=(),
        )

    matched_terms = tuple(
        term
        for term in query.strict_terms
        if contains_term(
            searchable_text,
            term,
        )
    )

    if matched_terms:
        return QueryMatch(
            matched=True,
            kind=MatchKind.STRICT,
            matched_terms=matched_terms,
        )

    return QueryMatch(
        matched=False,
        kind=MatchKind.NONE,
        matched_terms=(),
    )