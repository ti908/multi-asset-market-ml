"""FinBERT implementation of the sentiment scorer contract."""

from __future__ import annotations

from typing import Any

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from market_prediction.sentiment.scorer import (
    SentimentScorer,
    SentimentScores,
)


_REQUIRED_LABELS = {
    "positive",
    "neutral",
    "negative",
}


class FinBertScorer(SentimentScorer):
    """Score financial text using a FinBERT sequence classifier."""

    scoring_method = "finbert"

    @staticmethod
    def resolve_device() -> str:
        """Return the preferred available inference device."""

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str,
        model_revision: str,
        max_tokens: int,
        device: str | None = None,
    ) -> "FinBertScorer":
        """Load a pinned pretrained FinBERT model and tokenizer."""

        resolved_device = (
            device
            if device is not None
            else cls.resolve_device()
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name,
                revision=model_revision,
            )
        )

        return cls(
            model_name=model_name,
            model_revision=model_revision,
            max_tokens=max_tokens,
            device=resolved_device,
            tokenizer=tokenizer,
            model=model,
        )

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str,
        max_tokens: int,
        device: str,
        tokenizer: Any,
        model: Any,
    ) -> None:
        """Initialize a FinBERT scorer from injected model components."""

        if not isinstance(model_name, str):
            raise TypeError(
                "model_name must be a string"
            )

        if not model_name.strip():
            raise ValueError(
                "model_name must not be blank"
            )

        if not isinstance(model_revision, str):
            raise TypeError(
                "model_revision must be a string"
            )

        if not model_revision.strip():
            raise ValueError(
                "model_revision must not be blank"
            )

        if isinstance(max_tokens, bool):
            raise TypeError(
                "max_tokens must be an integer"
            )

        if not isinstance(max_tokens, int):
            raise TypeError(
                "max_tokens must be an integer"
            )

        if max_tokens < 1 or max_tokens > 512:
            raise ValueError(
                "max_tokens must be between 1 and 512"
            )

        if not isinstance(device, str):
            raise TypeError(
                "device must be a string"
            )

        if not device.strip():
            raise ValueError(
                "device must not be blank"
            )

        self.model_name = model_name
        self.model_revision = model_revision
        self.max_tokens = max_tokens
        self.device = device
        self._tokenizer = tokenizer
        self._model = model

        self._label_to_index = (
            self._build_label_mapping(
                model.config.id2label
            )
        )

        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _build_label_mapping(
        id2label: dict[int, str],
    ) -> dict[str, int]:
        """Build semantic label lookup from model metadata.

        Label indices are never assumed. They are derived from the
        classifier's id2label configuration.
        """

        if not isinstance(id2label, dict):
            raise ValueError(
                "FinBERT label mapping must be a dictionary"
            )

        label_to_index: dict[str, int] = {}

        for index, label in id2label.items():
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not isinstance(label, str)
            ):
                raise ValueError(
                    "FinBERT label mapping is invalid"
                )

            normalized_label = (
                label.strip().lower()
            )

            if normalized_label in label_to_index:
                raise ValueError(
                    "FinBERT label mapping contains "
                    "duplicate labels"
                )

            label_to_index[
                normalized_label
            ] = index

        if set(label_to_index) != _REQUIRED_LABELS:
            raise ValueError(
                "FinBERT label mapping must contain "
                "exactly positive, neutral, and negative"
            )

        if set(label_to_index.values()) != {
            0,
            1,
            2,
        }:
            raise ValueError(
                "FinBERT label mapping must contain "
                "exactly three classifier indices"
            )

        return label_to_index

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

        batch = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )

        batch = batch.to(self.device)

        with torch.inference_mode():
            output = self._model(**batch)

        logits = output.logits

        if (
            logits.ndim != 2
            or logits.shape[0] != 1
            or logits.shape[1] != 3
        ):
            raise ValueError(
                "FinBERT model must return logits "
                "with shape (1, 3)"
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0]

        positive_probability = float(
            probabilities[
                self._label_to_index["positive"]
            ].item()
        )

        neutral_probability = float(
            probabilities[
                self._label_to_index["neutral"]
            ].item()
        )

        negative_probability = float(
            probabilities[
                self._label_to_index["negative"]
            ].item()
        )

        return SentimentScores(
            positive_probability=(
                positive_probability
            ),
            neutral_probability=(
                neutral_probability
            ),
            negative_probability=(
                negative_probability
            ),
        )