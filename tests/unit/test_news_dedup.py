from __future__ import annotations

import pytest

from market_prediction.news.dedup import (
    duplicate_group_identity,
    group_candidates_by_url,
    normalize_article_url,
)
from datetime import datetime, timezone

from market_prediction.news.candidates import NewsCandidate

from market_prediction.news.dedup import (
    assign_duplicate_groups,
)
from datetime import datetime, timezone

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)


def test_normalize_article_url_lowercases_scheme_and_host() -> None:
    assert normalize_article_url(
        "HTTPS://Example.COM/Markets/Apple"
    ) == (
        "https://example.com/Markets/Apple"
    )


def test_normalize_article_url_removes_fragment() -> None:
    assert normalize_article_url(
        "https://example.com/article#section-2"
    ) == (
        "https://example.com/article"
    )


def test_normalize_article_url_preserves_query_string() -> None:
    assert normalize_article_url(
        "https://example.com/article?utm_source=x&id=42"
    ) == (
        "https://example.com/article?utm_source=x&id=42"
    )


def test_normalize_article_url_preserves_query_order() -> None:
    first = normalize_article_url(
        "https://example.com/article?a=1&b=2"
    )
    second = normalize_article_url(
        "https://example.com/article?b=2&a=1"
    )

    assert first != second


def test_normalize_article_url_preserves_path_case() -> None:
    first = normalize_article_url(
        "https://example.com/Markets/Apple"
    )
    second = normalize_article_url(
        "https://example.com/markets/apple"
    )

    assert first != second


def test_normalize_article_url_preserves_trailing_slash() -> None:
    first = normalize_article_url(
        "https://example.com/article"
    )
    second = normalize_article_url(
        "https://example.com/article/"
    )

    assert first != second


def test_normalize_article_url_preserves_explicit_port() -> None:
    assert normalize_article_url(
        "https://Example.COM:8443/article"
    ) == (
        "https://example.com:8443/article"
    )


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "example.com/article",
        "/article",
        "ftp://example.com/article",
        "mailto:news@example.com",
    ],
)
def test_normalize_article_url_rejects_invalid_url(
    url: str,
) -> None:
    with pytest.raises(ValueError):
        normalize_article_url(url)


def test_normalize_article_url_rejects_credentials() -> None:
    with pytest.raises(ValueError):
        normalize_article_url(
            "https://user:password@example.com/article"
        )


def test_normalize_article_url_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_article_url(
            None  # type: ignore[arg-type]
        )


def test_duplicate_group_identity_is_deterministic() -> None:
    first = duplicate_group_identity(
        "https://Example.COM/article#section"
    )
    second = duplicate_group_identity(
        "https://example.com/article"
    )

    assert first == second


def test_duplicate_group_identity_differs_for_distinct_urls() -> None:
    first = duplicate_group_identity(
        "https://example.com/article-a"
    )
    second = duplicate_group_identity(
        "https://example.com/article-b"
    )

    assert first != second


def test_duplicate_group_identity_is_sha256_hex() -> None:
    value = duplicate_group_identity(
        "https://example.com/article"
    )

    assert len(value) == 64
    assert all(
        character in "0123456789abcdef"
        for character in value
    )

UTC = timezone.utc


def make_candidate(
    *,
    candidate_id: str,
    provider_item_id: str,
    canonical_url: str,
) -> NewsCandidate:
    return NewsCandidate(
        candidate_id=candidate_id,
        asset="AAPL",
        query_id="aapl_company",
        canonical_url=canonical_url,
        publisher="example.com",
        published_at_utc=datetime(
            2015,
            2,
            19,
            tzinfo=UTC,
        ),
        published_at_precision=(
            PublishedAtPrecision.SECOND
        ),
        provider_seen_at_utc=datetime(
            2015,
            2,
            19,
            tzinfo=UTC,
        ),
        retrieved_at_utc=datetime(
            2026,
            10,
            31,
            tzinfo=UTC,
        ),
        provider="gdelt",
        provider_item_id=provider_item_id,
        timestamp_source=(
            TimestampSource.PROVIDER_SEEN
        ),
    )


def test_group_candidates_by_url_groups_equivalent_urls() -> None:
    first = make_candidate(
        candidate_id="candidate-a",
        provider_item_id="record-a",
        canonical_url=(
            "HTTPS://Example.COM/article#section"
        ),
    )
    second = make_candidate(
        candidate_id="candidate-b",
        provider_item_id="record-b",
        canonical_url=(
            "https://example.com/article"
        ),
    )

    groups = group_candidates_by_url(
        [first, second]
    )

    assert len(groups) == 1

    group = groups[0]

    assert group.duplicate_group_id == (
        duplicate_group_identity(
            first.canonical_url
        )
    )

    assert tuple(
        candidate.candidate_id
        for candidate in group.candidates
    ) == (
        "candidate-a",
        "candidate-b",
    )


def test_group_candidates_preserves_distinct_candidate_identity() -> None:
    first = make_candidate(
        candidate_id="candidate-a",
        provider_item_id="record-a",
        canonical_url="https://example.com/article",
    )
    second = make_candidate(
        candidate_id="candidate-b",
        provider_item_id="record-b",
        canonical_url="https://example.com/article",
    )

    groups = group_candidates_by_url(
        [first, second]
    )

    assert (
        groups[0].candidates[0].candidate_id
        != groups[0].candidates[1].candidate_id
    )

    assert (
        groups[0].candidates[0].provider_item_id
        != groups[0].candidates[1].provider_item_id
    )


def test_group_candidates_keeps_distinct_urls_separate() -> None:
    first = make_candidate(
        candidate_id="candidate-a",
        provider_item_id="record-a",
        canonical_url="https://example.com/article-a",
    )
    second = make_candidate(
        candidate_id="candidate-b",
        provider_item_id="record-b",
        canonical_url="https://example.com/article-b",
    )

    groups = group_candidates_by_url(
        [first, second]
    )

    assert len(groups) == 2


def test_group_candidates_preserves_first_group_order() -> None:
    candidates = [
        make_candidate(
            candidate_id="candidate-b1",
            provider_item_id="record-b1",
            canonical_url="https://example.com/b",
        ),
        make_candidate(
            candidate_id="candidate-a",
            provider_item_id="record-a",
            canonical_url="https://example.com/a",
        ),
        make_candidate(
            candidate_id="candidate-b2",
            provider_item_id="record-b2",
            canonical_url="https://example.com/b#fragment",
        ),
    ]

    groups = group_candidates_by_url(
        candidates
    )

    assert tuple(
        group.candidates[0].candidate_id
        for group in groups
    ) == (
        "candidate-b1",
        "candidate-a",
    )

    assert tuple(
        candidate.candidate_id
        for candidate in groups[0].candidates
    ) == (
        "candidate-b1",
        "candidate-b2",
    )


def test_group_candidates_accepts_empty_input() -> None:
    assert group_candidates_by_url([]) == []


def test_group_candidates_rejects_non_list() -> None:
    with pytest.raises(TypeError):
        group_candidates_by_url(
            None  # type: ignore[arg-type]
        )


def test_group_candidates_rejects_non_candidate_item() -> None:
    candidate = make_candidate(
        candidate_id="candidate-a",
        provider_item_id="record-a",
        canonical_url="https://example.com/a",
    )

    with pytest.raises(TypeError):
        group_candidates_by_url(
            [
                candidate,
                "not-a-candidate",  # type: ignore[list-item]
            ]
        )

def make_article(
    *,
    article_id: str,
    canonical_url: str,
) -> ArticleRecord:
    return ArticleRecord(
        article_id=article_id,
        matched_assets=["AAPL"],
        query_id="aapl-primary",
        title="Example article",
        snippet=None,
        full_text=None,
        canonical_url=canonical_url,
        publisher="Example Publisher",
        published_at_utc=datetime(
            2025,
            10,
            30,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        published_at_precision=(
            PublishedAtPrecision.SECOND
        ),
        original_timezone="UTC",
        provider_seen_at_utc=datetime(
            2025,
            10,
            30,
            12,
            1,
            tzinfo=timezone.utc,
        ),
        retrieved_at_utc=datetime(
            2025,
            10,
            30,
            12,
            2,
            tzinfo=timezone.utc,
        ),
        provider="marketaux",
        provider_item_id=article_id,
        language="en",
        content_hash=(
            "a" * 64
        ),
        duplicate_group_id=None,
        raw_payload_reference=None,
        schema_version="1",
        timestamp_source=(
            TimestampSource.PROVIDER_PUBLICATION
        ),
        text_license_class=(
            TextLicenseClass.PROVIDER_METADATA
        ),
    )

def test_assign_duplicate_groups_same_url_same_group() -> None:
    first = make_article(
        article_id="article-1",
        canonical_url=(
            "https://example.com/story"
        ),
    )
    second = make_article(
        article_id="article-2",
        canonical_url=(
            "https://EXAMPLE.com/story#section"
        ),
    )

    result = assign_duplicate_groups(
        [first, second]
    )

    expected_group = duplicate_group_identity(
        "https://example.com/story"
    )

    assert result[0].duplicate_group_id == (
        expected_group
    )
    assert result[1].duplicate_group_id == (
        expected_group
    )


def test_assign_duplicate_groups_different_urls_different_groups() -> None:
    first = make_article(
        article_id="article-1",
        canonical_url=(
            "https://example.com/story-1"
        ),
    )
    second = make_article(
        article_id="article-2",
        canonical_url=(
            "https://example.com/story-2"
        ),
    )

    result = assign_duplicate_groups(
        [first, second]
    )

    assert (
        result[0].duplicate_group_id
        != result[1].duplicate_group_id
    )


def test_assign_duplicate_groups_preserves_article_ids() -> None:
    articles = [
        make_article(
            article_id="article-1",
            canonical_url=(
                "https://example.com/story"
            ),
        ),
        make_article(
            article_id="article-2",
            canonical_url=(
                "https://example.com/story"
            ),
        ),
    ]

    result = assign_duplicate_groups(
        articles
    )

    assert [
        article.article_id
        for article in result
    ] == [
        "article-1",
        "article-2",
    ]


def test_assign_duplicate_groups_preserves_order() -> None:
    articles = [
        make_article(
            article_id="third",
            canonical_url=(
                "https://example.com/3"
            ),
        ),
        make_article(
            article_id="first",
            canonical_url=(
                "https://example.com/1"
            ),
        ),
        make_article(
            article_id="second",
            canonical_url=(
                "https://example.com/2"
            ),
        ),
    ]

    result = assign_duplicate_groups(
        articles
    )

    assert [
        article.article_id
        for article in result
    ] == [
        "third",
        "first",
        "second",
    ]


def test_assign_duplicate_groups_does_not_mutate_input() -> None:
    article = make_article(
        article_id="article-1",
        canonical_url=(
            "https://example.com/story"
        ),
    )

    assert article.duplicate_group_id is None

    result = assign_duplicate_groups(
        [article]
    )

    assert article.duplicate_group_id is None
    assert result[0].duplicate_group_id is not None
    assert result[0] is not article


def test_assign_duplicate_groups_preserves_all_records() -> None:
    articles = [
        make_article(
            article_id="article-1",
            canonical_url=(
                "https://example.com/story"
            ),
        ),
        make_article(
            article_id="article-2",
            canonical_url=(
                "https://example.com/story"
            ),
        ),
        make_article(
            article_id="article-3",
            canonical_url=(
                "https://example.com/other"
            ),
        ),
    ]

    result = assign_duplicate_groups(
        articles
    )

    assert len(result) == 3