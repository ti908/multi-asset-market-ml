"""VADER baseline implementation of the sentiment scorer contract."""

from __future__ import annotations

from typing import Any

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer,
)

from market_prediction.sentiment.scorer import (
    SentimentScorer,
    SentimentScores,
)


_REQUIRED_OUTPUT_KEYS = {
    "pos",
    "neu",
    "neg",
    "compound",
}


class VaderScorer(SentimentScorer):
    """Score text using the VADER lexical sentiment baseline."""

    model_name = "vaderSentiment"
    model_revision = "3.3.2"
    scoring_method = "vader"

    @classmethod
    def from_pretrained(
        cls,
    ) -> "VaderScorer":
        """Construct the scorer with the standard VADER analyzer."""

        analyzer = SentimentIntensityAnalyzer()

        return cls(
            analyzer=analyzer,
        )

    def __init__(
        self,
        *,
        analyzer: Any,
    ) -> None:
        """Initialize the scorer with an injected VADER analyzer."""

        self._analyzer = analyzer

    def score(
        self,
        input_text: str,
    ) -> SentimentScores:
        """Score one exact sentiment input string."""

        if not isinstance(input_text, str):
            raise TypeError(
                "input_text must be a string"
            )

        if not input_text.strip():
            raise ValueError(
                "input_text must not be blank"
            )

        scores = self._analyzer.polarity_scores(
            input_text
        )

        if (
            not isinstance(scores, dict)
            or set(scores) != _REQUIRED_OUTPUT_KEYS
        ):
            raise ValueError(
                "VADER analyzer output must contain "
                "exactly pos, neu, neg, and compound"
            )

        return SentimentScores(
            positive_probability=scores["pos"],
            neutral_probability=scores["neu"],
            negative_probability=scores["neg"],
        )