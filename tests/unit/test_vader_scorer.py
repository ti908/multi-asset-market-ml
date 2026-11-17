"""Tests for the VADER baseline sentiment scorer."""

from __future__ import annotations

import pytest

from market_prediction.sentiment.scorer import SentimentScores
from market_prediction.sentiment.vader import VaderScorer


class FakeAnalyzer:
    """Deterministic VADER analyzer test double."""

    def __init__(
        self,
        scores: dict[str, float] | None = None,
    ) -> None:
        self.scores = scores or {
            "pos": 0.70,
            "neu": 0.20,
            "neg": 0.10,
            "compound": 0.75,
        }
        self.calls: list[str] = []

    def polarity_scores(
        self,
        text: str,
    ) -> dict[str, float]:
        self.calls.append(text)
        return dict(self.scores)


def test_vader_scorer_metadata() -> None:
    scorer = VaderScorer(
        analyzer=FakeAnalyzer()
    )

    assert scorer.model_name == "vaderSentiment"
    assert scorer.model_revision == "3.3.2"
    assert scorer.scoring_method == "vader"


def test_vader_passes_exact_text_to_analyzer() -> None:
    analyzer = FakeAnalyzer()
    scorer = VaderScorer(analyzer=analyzer)

    text = "  Apple reported record revenue.\n"

    scorer.score(text)

    assert analyzer.calls == [text]


def test_vader_returns_sentiment_scores() -> None:
    scorer = VaderScorer(
        analyzer=FakeAnalyzer()
    )

    result = scorer.score(
        "Apple reported record revenue."
    )

    assert isinstance(result, SentimentScores)


def test_vader_maps_normalized_scores_directly() -> None:
    scorer = VaderScorer(
        analyzer=FakeAnalyzer(
            {
                "pos": 0.65,
                "neu": 0.25,
                "neg": 0.10,
                "compound": 0.80,
            }
        )
    )

    result = scorer.score("Financial text.")

    assert result.positive_probability == pytest.approx(
        0.65
    )
    assert result.neutral_probability == pytest.approx(
        0.25
    )
    assert result.negative_probability == pytest.approx(
        0.10
    )


def test_vader_signed_score_uses_common_contract() -> None:
    scorer = VaderScorer(
        analyzer=FakeAnalyzer(
            {
                "pos": 0.60,
                "neu": 0.25,
                "neg": 0.15,
                "compound": 0.90,
            }
        )
    )

    result = scorer.score("Financial text.")

    assert result.signed_score == pytest.approx(
        0.45
    )


def test_vader_does_not_use_compound_as_signed_score() -> None:
    scorer = VaderScorer(
        analyzer=FakeAnalyzer(
            {
                "pos": 0.60,
                "neu": 0.25,
                "neg": 0.15,
                "compound": -0.99,
            }
        )
    )

    result = scorer.score("Financial text.")

    assert result.signed_score == pytest.approx(
        0.45
    )
    assert result.signed_score != pytest.approx(
        -0.99
    )


def test_vader_prediction_comes_from_common_scores() -> None:
    scorer = VaderScorer(
        analyzer=FakeAnalyzer(
            {
                "pos": 0.10,
                "neu": 0.20,
                "neg": 0.70,
                "compound": -0.75,
            }
        )
    )

    result = scorer.score("Financial text.")

    assert result.predicted_label == "negative"
    assert result.confidence == pytest.approx(
        0.70
    )


@pytest.mark.parametrize(
    "input_text",
    [
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_vader_rejects_blank_input_text(
    input_text: str,
) -> None:
    analyzer = FakeAnalyzer()
    scorer = VaderScorer(analyzer=analyzer)

    with pytest.raises(
        ValueError,
        match="input_text must not be blank",
    ):
        scorer.score(input_text)

    assert analyzer.calls == []


def test_vader_rejects_non_string_input_text() -> None:
    analyzer = FakeAnalyzer()
    scorer = VaderScorer(analyzer=analyzer)

    with pytest.raises(TypeError):
        scorer.score(123)  # type: ignore[arg-type]

    assert analyzer.calls == []


@pytest.mark.parametrize(
    "scores",
    [
        {
            "pos": 0.5,
            "neu": 0.5,
            "compound": 0.0,
        },
        {
            "pos": 0.5,
            "neu": 0.3,
            "neg": 0.2,
        },
        {
            "positive": 0.5,
            "neutral": 0.3,
            "negative": 0.2,
            "compound": 0.0,
        },
    ],
)
def test_vader_rejects_invalid_analyzer_shape(
    scores: dict[str, float],
) -> None:
    scorer = VaderScorer(
        analyzer=FakeAnalyzer(scores)
    )

    with pytest.raises(
        ValueError,
        match="VADER analyzer output",
    ):
        scorer.score("Financial text.")


def test_vader_rejects_scores_that_do_not_sum_to_one() -> None:
    scorer = VaderScorer(
        analyzer=FakeAnalyzer(
            {
                "pos": 0.70,
                "neu": 0.20,
                "neg": 0.20,
                "compound": 0.75,
            }
        )
    )

    with pytest.raises(ValueError):
        scorer.score("Financial text.")

def test_from_pretrained_constructs_vader_analyzer(
    monkeypatch,
) -> None:
    analyzer = FakeAnalyzer()
    calls = []

    def fake_analyzer_constructor():
        calls.append(True)
        return analyzer

    monkeypatch.setattr(
        "market_prediction.sentiment.vader."
        "SentimentIntensityAnalyzer",
        fake_analyzer_constructor,
    )

    scorer = VaderScorer.from_pretrained()

    assert calls == [True]
    assert scorer.model_name == "vaderSentiment"
    assert scorer.model_revision == "3.3.2"
    assert scorer.scoring_method == "vader"


def test_from_pretrained_returns_working_scorer(
    monkeypatch,
) -> None:
    analyzer = FakeAnalyzer(
        {
            "pos": 0.70,
            "neu": 0.20,
            "neg": 0.10,
            "compound": 0.75,
        }
    )

    monkeypatch.setattr(
        "market_prediction.sentiment.vader."
        "SentimentIntensityAnalyzer",
        lambda: analyzer,
    )

    scorer = VaderScorer.from_pretrained()

    result = scorer.score(
        "Apple reported record revenue."
    )

    assert isinstance(result, SentimentScores)
    assert result.positive_probability == pytest.approx(
        0.70
    )
    assert result.neutral_probability == pytest.approx(
        0.20
    )
    assert result.negative_probability == pytest.approx(
        0.10
    )
    assert result.predicted_label == "positive"