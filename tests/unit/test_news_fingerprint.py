"""Tests for deterministic Step 4 news identities and fingerprints."""

from __future__ import annotations

import re

from market_prediction.news.fingerprint import (
    article_identity,
    content_fingerprint,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def test_article_identity_is_sha256_hex() -> None:
    result = article_identity(
        provider="marketaux",
        provider_item_id="article-123",
    )

    assert SHA256_PATTERN.fullmatch(result)


def test_article_identity_is_deterministic() -> None:
    first = article_identity(
        provider="marketaux",
        provider_item_id="article-123",
    )

    second = article_identity(
        provider="marketaux",
        provider_item_id="article-123",
    )

    assert first == second


def test_article_identity_changes_with_provider() -> None:
    marketaux = article_identity(
        provider="marketaux",
        provider_item_id="article-123",
    )

    gdelt = article_identity(
        provider="gdelt",
        provider_item_id="article-123",
    )

    assert marketaux != gdelt


def test_article_identity_changes_with_provider_item_id() -> None:
    first = article_identity(
        provider="marketaux",
        provider_item_id="article-123",
    )

    second = article_identity(
        provider="marketaux",
        provider_item_id="article-456",
    )

    assert first != second


def test_article_identity_rejects_empty_provider() -> None:
    try:
        article_identity(
            provider="",
            provider_item_id="article-123",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected empty provider to be rejected"
        )


def test_article_identity_rejects_empty_provider_item_id() -> None:
    try:
        article_identity(
            provider="marketaux",
            provider_item_id="",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected empty provider_item_id to be rejected"
        )


def test_content_fingerprint_is_sha256_hex() -> None:
    result = content_fingerprint(
        title="Apple reports quarterly earnings",
        snippet="Revenue increased during the quarter.",
        full_text=None,
    )

    assert SHA256_PATTERN.fullmatch(result)


def test_content_fingerprint_is_deterministic() -> None:
    kwargs = {
        "title": "Apple reports quarterly earnings",
        "snippet": "Revenue increased during the quarter.",
        "full_text": None,
    }

    assert content_fingerprint(**kwargs) == content_fingerprint(
        **kwargs
    )


def test_content_fingerprint_changes_when_title_changes() -> None:
    first = content_fingerprint(
        title="Apple reports quarterly earnings",
        snippet="Revenue increased.",
        full_text=None,
    )

    second = content_fingerprint(
        title="Microsoft reports quarterly earnings",
        snippet="Revenue increased.",
        full_text=None,
    )

    assert first != second


def test_content_fingerprint_changes_when_snippet_changes() -> None:
    first = content_fingerprint(
        title="Apple reports quarterly earnings",
        snippet="Revenue increased.",
        full_text=None,
    )

    second = content_fingerprint(
        title="Apple reports quarterly earnings",
        snippet="Revenue declined.",
        full_text=None,
    )

    assert first != second


def test_content_fingerprint_changes_when_full_text_changes() -> None:
    first = content_fingerprint(
        title="Apple reports quarterly earnings",
        snippet="Revenue increased.",
        full_text="First article body.",
    )

    second = content_fingerprint(
        title="Apple reports quarterly earnings",
        snippet="Revenue increased.",
        full_text="Second article body.",
    )

    assert first != second


def test_content_fingerprint_distinguishes_none_from_empty_text() -> None:
    missing = content_fingerprint(
        title="Apple reports quarterly earnings",
        snippet=None,
        full_text=None,
    )

    present_but_empty = content_fingerprint(
        title="Apple reports quarterly earnings",
        snippet="",
        full_text=None,
    )

    assert missing != present_but_empty


def test_content_fingerprint_preserves_meaningful_whitespace() -> None:
    first = content_fingerprint(
        title="Apple reports earnings",
        snippet="Revenue increased.",
        full_text=None,
    )

    second = content_fingerprint(
        title="Apple  reports earnings",
        snippet="Revenue increased.",
        full_text=None,
    )

    assert first != second


def test_content_fingerprint_rejects_empty_title() -> None:
    try:
        content_fingerprint(
            title="",
            snippet=None,
            full_text=None,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected empty title to be rejected"
        )