"""Tests for sentiment scorer contracts and input-text construction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from market_prediction.sentiment.scorer import (
    SentimentScores,
    SentimentScorer,
    build_sentiment_input_text,
)


def test_title_and_snippet_are_joined_deterministically() -> None:
    result = build_sentiment_input_text(
        title="Apple reports record revenue",
        snippet="Guidance was raised for the next quarter.",
    )

    assert result == (
        "Apple reports record revenue"
        "\n\n"
        "Guidance was raised for the next quarter."
    )


@pytest.mark.parametrize(
    "snippet",
    [
        None,
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_missing_or_blank_snippet_uses_title_only(
    snippet: str | None,
) -> None:
    result = build_sentiment_input_text(
        title="Apple reports record revenue",
        snippet=snippet,
    )

    assert result == "Apple reports record revenue"


def test_input_builder_preserves_exact_nonblank_text() -> None:
    result = build_sentiment_input_text(
        title="  Apple Reports Revenue  ",
        snippet="  Revenue rose 10%.  ",
    )

    assert result == (
        "  Apple Reports Revenue  "
        "\n\n"
        "  Revenue rose 10%.  "
    )


@pytest.mark.parametrize(
    "title",
    [
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_input_builder_rejects_blank_title(
    title: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="title must not be blank",
    ):
        build_sentiment_input_text(
            title=title,
            snippet="Valid snippet.",
        )


def test_input_builder_rejects_non_string_title() -> None:
    with pytest.raises(TypeError):
        build_sentiment_input_text(
            title=123,  # type: ignore[arg-type]
            snippet="Valid snippet.",
        )


def test_input_builder_rejects_non_string_snippet() -> None:
    with pytest.raises(TypeError):
        build_sentiment_input_text(
            title="Valid title",
            snippet=123,  # type: ignore[arg-type]
        )


def test_sentiment_scores_accept_valid_probabilities() -> None:
    scores = SentimentScores(
        positive_probability=0.80,
        neutral_probability=0.15,
        negative_probability=0.05,
    )

    assert scores.positive_probability == 0.80
    assert scores.neutral_probability == 0.15
    assert scores.negative_probability == 0.05


def test_sentiment_scores_are_immutable() -> None:
    scores = SentimentScores(
        positive_probability=0.80,
        neutral_probability=0.15,
        negative_probability=0.05,
    )

    with pytest.raises(FrozenInstanceError):
        scores.positive_probability = 0.50  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("positive_probability", -0.01),
        ("positive_probability", 1.01),
        ("neutral_probability", -0.01),
        ("neutral_probability", 1.01),
        ("negative_probability", -0.01),
        ("negative_probability", 1.01),
    ],
)
def test_sentiment_scores_reject_out_of_range_probability(
    field: str,
    value: float,
) -> None:
    values = {
        "positive_probability": 0.80,
        "neutral_probability": 0.15,
        "negative_probability": 0.05,
    }
    values[field] = value

    with pytest.raises(ValueError):
        SentimentScores(**values)


def test_sentiment_scores_require_probabilities_to_sum_to_one() -> None:
    with pytest.raises(
        ValueError,
        match="probabilities must sum to 1",
    ):
        SentimentScores(
            positive_probability=0.80,
            neutral_probability=0.15,
            negative_probability=0.10,
        )


def test_sentiment_scores_expose_signed_score() -> None:
    scores = SentimentScores(
        positive_probability=0.80,
        neutral_probability=0.15,
        negative_probability=0.05,
    )

    assert scores.signed_score == pytest.approx(0.75)


def test_sentiment_scores_expose_confidence() -> None:
    scores = SentimentScores(
        positive_probability=0.10,
        neutral_probability=0.70,
        negative_probability=0.20,
    )

    assert scores.confidence == pytest.approx(0.70)


def test_sentiment_scores_expose_deterministic_predicted_label() -> None:
    scores = SentimentScores(
        positive_probability=0.80,
        neutral_probability=0.15,
        negative_probability=0.05,
    )

    assert scores.predicted_label == "positive"


def test_sentiment_scores_tie_breaking_is_deterministic() -> None:
    scores = SentimentScores(
        positive_probability=0.40,
        neutral_probability=0.40,
        negative_probability=0.20,
    )

    assert scores.predicted_label == "positive"


def test_sentiment_scorer_contract_is_abstract() -> None:
    with pytest.raises(TypeError):
        SentimentScorer()  # type: ignore[abstract]