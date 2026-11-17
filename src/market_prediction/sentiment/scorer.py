"""Common sentiment scorer contracts and deterministic input construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


_LABEL_PRECEDENCE = (
    "positive",
    "neutral",
    "negative",
)


def build_sentiment_input_text(
    *,
    title: str,
    snippet: str | None,
) -> str:
    """Build the exact text representation supplied to a scorer.

    Nonblank title and snippet text is preserved exactly. A missing or
    whitespace-only snippet is omitted. No lowercasing, stripping,
    truncation, summarization, or other normalization is performed here.
    """

    if not isinstance(title, str):
        raise TypeError("title must be a string")

    if not title.strip():
        raise ValueError("title must not be blank")

    if snippet is not None and not isinstance(
        snippet,
        str,
    ):
        raise TypeError(
            "snippet must be a string or None"
        )

    if snippet is None or not snippet.strip():
        return title

    return f"{title}\n\n{snippet}"


@dataclass(frozen=True)
class SentimentScores:
    """Normalized probability output from a sentiment scorer."""

    positive_probability: float
    neutral_probability: float
    negative_probability: float

    def __post_init__(self) -> None:
        """Validate probability bounds and normalization."""

        probabilities = (
            self.positive_probability,
            self.neutral_probability,
            self.negative_probability,
        )

        for probability in probabilities:
            if not isinstance(
                probability,
                (int, float),
            ):
                raise TypeError(
                    "probabilities must be numeric"
                )

            if isinstance(probability, bool):
                raise TypeError(
                    "probabilities must be numeric"
                )

            if not math.isfinite(probability):
                raise ValueError(
                    "probabilities must be finite"
                )

            if probability < 0 or probability > 1:
                raise ValueError(
                    "probabilities must be between 0 and 1"
                )

        if not math.isclose(
            sum(probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "probabilities must sum to 1"
            )

    @property
    def signed_score(self) -> float:
        """Return P(positive) - P(negative)."""

        return (
            self.positive_probability
            - self.negative_probability
        )

    @property
    def confidence(self) -> float:
        """Return the highest class probability."""

        return max(
            self.positive_probability,
            self.neutral_probability,
            self.negative_probability,
        )

    @property
    def predicted_label(self) -> str:
        """Return the highest-probability label deterministically.

        Exact ties use the frozen precedence:
        positive, then neutral, then negative.
        """

        probabilities = {
            "positive": self.positive_probability,
            "neutral": self.neutral_probability,
            "negative": self.negative_probability,
        }

        highest = self.confidence

        for label in _LABEL_PRECEDENCE:
            if probabilities[label] == highest:
                return label

        raise RuntimeError(
            "Unable to determine predicted sentiment label"
        )


class SentimentScorer(ABC):
    """Common contract implemented by sentiment scorers."""

    @abstractmethod
    def score(
        self,
        input_text: str,
    ) -> SentimentScores:
        """Score one exact sentiment input string."""