"""Canonical article-level sentiment contracts for Step 5.

This module defines provider-neutral sentiment observations derived
from validated news articles.

It contains no model loading, inference, temporal aggregation,
storage, feature engineering, price alignment, or modeling logic.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import math

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class SentimentLabel(str, Enum):
    """Canonical sentiment classes."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentRecord(BaseModel):
    """Canonical article-level sentiment observation.

    The record retains article and information-time provenance so
    downstream feature engineering can enforce point-in-time safety.

    All normalized timestamps must be timezone-aware UTC values.
    """

    model_config = ConfigDict(extra="forbid")

    sentiment_id: str = Field(min_length=1)

    article_id: str = Field(min_length=1)
    asset: str = Field(min_length=1)
    query_id: str = Field(min_length=1)

    published_at_utc: datetime
    provider_seen_at_utc: datetime

    model_name: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    scoring_method: str = Field(min_length=1)

    positive_probability: float = Field(
        ge=0.0,
        le=1.0,
    )
    negative_probability: float = Field(
        ge=0.0,
        le=1.0,
    )
    neutral_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    predicted_label: SentimentLabel

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    signed_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    scored_at_utc: datetime

    input_text_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    schema_version: str = Field(min_length=1)

    @field_validator(
        "published_at_utc",
        "provider_seen_at_utc",
        "scored_at_utc",
    )
    @classmethod
    def validate_utc_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        """Reject naive and non-UTC normalized timestamps."""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "normalized timestamps must "
                "be timezone-aware"
            )

        if value.utcoffset().total_seconds() != 0:
            raise ValueError(
                "normalized timestamps must be UTC"
            )

        return value

    @model_validator(mode="after")
    def validate_sentiment_invariants(
        self,
    ) -> "SentimentRecord":
        """Validate probabilities, derived values, and chronology."""

        probabilities = {
            SentimentLabel.POSITIVE:
                self.positive_probability,
            SentimentLabel.NEGATIVE:
                self.negative_probability,
            SentimentLabel.NEUTRAL:
                self.neutral_probability,
        }

        probability_sum = sum(
            probabilities.values()
        )

        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "probabilities must sum to 1"
            )

        expected_signed_score = (
            self.positive_probability
            - self.negative_probability
        )

        if not math.isclose(
            self.signed_score,
            expected_signed_score,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "signed_score must equal "
                "positive_probability - "
                "negative_probability"
            )

        highest_probability = max(
            probabilities.values()
        )

        highest_labels = {
            label
            for label, probability
            in probabilities.items()
            if math.isclose(
                probability,
                highest_probability,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        }

        if self.predicted_label not in highest_labels:
            raise ValueError(
                "predicted_label must match "
                "highest probability"
            )

        if not math.isclose(
            self.confidence,
            highest_probability,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "confidence must equal "
                "highest probability"
            )

        if (
            self.provider_seen_at_utc
            < self.published_at_utc
        ):
            raise ValueError(
                "provider_seen_at_utc cannot precede "
                "published_at_utc"
            )

        if (
            self.scored_at_utc
            < self.provider_seen_at_utc
        ):
            raise ValueError(
                "scored_at_utc cannot precede "
                "provider_seen_at_utc"
            )

        return self
