from __future__ import annotations

import hashlib
from urllib.parse import (
    urlsplit,
    urlunsplit,
)
from dataclasses import dataclass

from market_prediction.news.candidates import (
    NewsCandidate,
)

from market_prediction.news.models import ArticleRecord

@dataclass(frozen=True)
class CandidateDuplicateGroup:
    """Candidates sharing one deterministic normalized-URL identity."""

    duplicate_group_id: str
    candidates: tuple[NewsCandidate, ...]


def group_candidates_by_url(
    candidates: list[NewsCandidate],
) -> list[CandidateDuplicateGroup]:
    """Group candidates by conservative normalized URL identity.

    Candidate records are retained rather than collapsed. Both group
    order and candidate order follow first appearance in the input.
    """

    if not isinstance(candidates, list):
        raise TypeError(
            "candidates must be provided as a list"
        )

    grouped: dict[
        str,
        list[NewsCandidate],
    ] = {}

    for candidate in candidates:
        if not isinstance(
            candidate,
            NewsCandidate,
        ):
            raise TypeError(
                "candidates must contain only NewsCandidate instances"
            )

        group_id = duplicate_group_identity(
            candidate.canonical_url
        )

        if group_id not in grouped:
            grouped[group_id] = []

        grouped[group_id].append(
            candidate
        )

    return [
        CandidateDuplicateGroup(
            duplicate_group_id=group_id,
            candidates=tuple(group),
        )
        for group_id, group in grouped.items()
    ]


def normalize_article_url(
    url: str,
) -> str:
    """Normalize an article URL conservatively for duplicate grouping.

    Normalization intentionally performs only transformations that are
    safe for deterministic resource identity:

    - lowercase scheme and hostname;
    - remove fragments;
    - preserve path case;
    - preserve query string and query ordering;
    - preserve explicit ports;
    - preserve trailing slashes.

    Redirect resolution, tracking-parameter removal, query sorting, and
    other aggressive canonicalization are deliberately excluded.
    """

    if not isinstance(url, str):
        raise TypeError(
            "article URL must be a string"
        )

    if not url.strip():
        raise ValueError(
            "article URL cannot be blank"
        )

    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError(
            "article URL is invalid"
        ) from exc

    scheme = parsed.scheme.lower()

    if scheme not in {
        "http",
        "https",
    }:
        raise ValueError(
            "article URL must use http or https"
        )

    if not parsed.hostname:
        raise ValueError(
            "article URL must contain a hostname"
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "article URL must not contain credentials"
        )

    hostname = parsed.hostname.lower()

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            "article URL contains an invalid port"
        ) from exc

    if ":" in hostname:
        normalized_host = (
            f"[{hostname}]"
        )
    else:
        normalized_host = hostname

    if port is not None:
        netloc = (
            f"{normalized_host}:{port}"
        )
    else:
        netloc = normalized_host

    return urlunsplit(
        (
            scheme,
            netloc,
            parsed.path,
            parsed.query,
            "",
        )
    )


def duplicate_group_identity(
    canonical_url: str,
) -> str:
    """Return deterministic SHA-256 identity for a normalized URL."""

    normalized_url = normalize_article_url(
        canonical_url
    )

    return hashlib.sha256(
        normalized_url.encode(
            "utf-8"
        )
    ).hexdigest()

def assign_duplicate_groups(
    articles: list[ArticleRecord],
) -> list[ArticleRecord]:
    """Assign deterministic URL duplicate groups.

    Records are preserved in their original order.
    No records are collapsed or removed, and input
    ArticleRecord instances are not mutated.
    """
    if not isinstance(articles, list):
        raise TypeError(
            "articles must be a list"
        )

    grouped: list[ArticleRecord] = []

    for article in articles:
        if not isinstance(article, ArticleRecord):
            raise TypeError(
                "articles must contain "
                "ArticleRecord instances"
            )

        duplicate_group_id = (
            duplicate_group_identity(
                article.canonical_url
            )
        )

        grouped.append(
            article.model_copy(
                update={
                    "duplicate_group_id": (
                        duplicate_group_id
                    )
                }
            )
        )

    return grouped