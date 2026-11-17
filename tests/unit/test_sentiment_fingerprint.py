"""Tests for deterministic sentiment identities and fingerprints."""

import re

import pytest

from market_prediction.sentiment.fingerprint import (
    sentiment_identity,
    sentiment_input_text_hash,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def test_sentiment_input_text_hash_is_deterministic() -> None:
    first = sentiment_input_text_hash(
        "Apple reported record quarterly revenue."
    )
    second = sentiment_input_text_hash(
        "Apple reported record quarterly revenue."
    )

    assert first == second
    assert SHA256_PATTERN.fullmatch(first)


def test_sentiment_input_text_hash_changes_with_text() -> None:
    first = sentiment_input_text_hash(
        "Apple reported record quarterly revenue."
    )
    second = sentiment_input_text_hash(
        "Apple reported weaker quarterly revenue."
    )

    assert first != second


def test_sentiment_input_text_hash_preserves_exact_text() -> None:
    original = sentiment_input_text_hash(
        "Apple reported record revenue."
    )
    whitespace_changed = sentiment_input_text_hash(
        "Apple reported  record revenue."
    )
    case_changed = sentiment_input_text_hash(
        "apple reported record revenue."
    )

    assert original != whitespace_changed
    assert original != case_changed


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_sentiment_input_text_hash_rejects_blank_text(
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="input_text must not be blank",
    ):
        sentiment_input_text_hash(value)


def test_sentiment_input_text_hash_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        sentiment_input_text_hash(123)  # type: ignore[arg-type]


def test_sentiment_identity_is_deterministic() -> None:
    kwargs = {
        "article_id": "marketaux-article-1",
        "asset": "AAPL",
        "model_name": "ProsusAI/finbert",
        "model_revision": (
            "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
        ),
        "scoring_method": "finbert",
    }

    first = sentiment_identity(**kwargs)
    second = sentiment_identity(**kwargs)

    assert first == second
    assert SHA256_PATTERN.fullmatch(first)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("article_id", "marketaux-article-2"),
        ("asset", "MSFT"),
        ("model_name", "other/model"),
        ("model_revision", "b" * 40),
        ("scoring_method", "vader"),
    ],
)
def test_sentiment_identity_changes_when_identity_field_changes(
    field: str,
    replacement: str,
) -> None:
    values = {
        "article_id": "marketaux-article-1",
        "asset": "AAPL",
        "model_name": "ProsusAI/finbert",
        "model_revision": (
            "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
        ),
        "scoring_method": "finbert",
    }

    original = sentiment_identity(**values)

    values[field] = replacement

    changed = sentiment_identity(**values)

    assert changed != original


@pytest.mark.parametrize(
    "field",
    [
        "article_id",
        "asset",
        "model_name",
        "model_revision",
        "scoring_method",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
    ],
)
def test_sentiment_identity_rejects_blank_fields(
    field: str,
    value: str,
) -> None:
    values = {
        "article_id": "marketaux-article-1",
        "asset": "AAPL",
        "model_name": "ProsusAI/finbert",
        "model_revision": (
            "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
        ),
        "scoring_method": "finbert",
    }

    values[field] = value

    with pytest.raises(
        ValueError,
        match=f"{field} must not be blank",
    ):
        sentiment_identity(**values)