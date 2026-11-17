"""Point-in-time temporal aggregation of article-level sentiment.

This module converts canonical SentimentRecord observations into
asset-level trailing-window sentiment features.

Eligibility is based on provider_seen_at_utc rather than publication
time so that observations cannot enter a feature window before they
were actually available from the upstream provider.

No price alignment, storage, model training, or missing-value
imputation is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
import math
import statistics

from market_prediction.sentiment.models import (
    SentimentLabel,
    SentimentRecord,
)


class SentimentWindow(IntEnum):
    """Frozen trailing sentiment windows, expressed in hours."""

    SIX_HOURS = 6
    TWENTY_FOUR_HOURS = 24
    SEVENTY_TWO_HOURS = 72


@dataclass(frozen=True)
class SentimentAggregationRecord:
    """Aggregated sentiment features for one asset and trailing window."""

    asset: str
    cutoff_at_utc: datetime
    window_hours: int

    article_count: int

    signed_score_mean: float | None
    signed_score_std: float | None
    signed_score_min: float | None
    signed_score_max: float | None

    positive_probability_mean: float | None
    neutral_probability_mean: float | None
    negative_probability_mean: float | None

    positive_label_share: float | None
    neutral_label_share: float | None
    negative_label_share: float | None

    def __post_init__(self) -> None:
        """Validate aggregation-record invariants."""

        _validate_utc_timestamp(
            self.cutoff_at_utc,
            field_name="cutoff_at_utc",
        )

        if not isinstance(self.asset, str) or not self.asset.strip():
            raise ValueError("asset must be a nonblank string")

        if (
            isinstance(self.window_hours, bool)
            or not isinstance(self.window_hours, int)
            or self.window_hours <= 0
        ):
            raise ValueError(
                "window_hours must be a positive integer"
            )

        if (
            isinstance(self.article_count, bool)
            or not isinstance(self.article_count, int)
            or self.article_count < 0
        ):
            raise ValueError(
                "article_count must be a nonnegative integer"
            )

        feature_values = (
            self.signed_score_mean,
            self.signed_score_std,
            self.signed_score_min,
            self.signed_score_max,
            self.positive_probability_mean,
            self.neutral_probability_mean,
            self.negative_probability_mean,
            self.positive_label_share,
            self.neutral_label_share,
            self.negative_label_share,
        )

        if self.article_count == 0:
            if any(
                value is not None
                for value in feature_values
            ):
                raise ValueError(
                    "empty aggregation windows must preserve "
                    "missing sentiment features"
                )
            return

        if any(
            value is None
            for value in feature_values
        ):
            raise ValueError(
                "nonempty aggregation windows require "
                "all sentiment features"
            )

        numeric_values = tuple(
            value
            for value in feature_values
            if value is not None
        )

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_values
        ):
            raise ValueError(
                "aggregation features must be finite numbers"
            )

        probability_means = (
            self.positive_probability_mean,
            self.neutral_probability_mean,
            self.negative_probability_mean,
        )

        label_shares = (
            self.positive_label_share,
            self.neutral_label_share,
            self.negative_label_share,
        )

        if any(
            value is None
            or value < 0.0
            or value > 1.0
            for value in probability_means
        ):
            raise ValueError(
                "probability means must be within [0, 1]"
            )

        if any(
            value is None
            or value < 0.0
            or value > 1.0
            for value in label_shares
        ):
            raise ValueError(
                "label shares must be within [0, 1]"
            )

        if not math.isclose(
            sum(
                value
                for value in probability_means
                if value is not None
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "mean probabilities must sum to 1"
            )

        if not math.isclose(
            sum(
                value
                for value in label_shares
                if value is not None
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "label shares must sum to 1"
            )

        if (
            self.signed_score_mean is None
            or self.signed_score_mean < -1.0
            or self.signed_score_mean > 1.0
        ):
            raise ValueError(
                "signed_score_mean must be within [-1, 1]"
            )

        if (
            self.signed_score_min is None
            or self.signed_score_min < -1.0
            or self.signed_score_min > 1.0
        ):
            raise ValueError(
                "signed_score_min must be within [-1, 1]"
            )

        if (
            self.signed_score_max is None
            or self.signed_score_max < -1.0
            or self.signed_score_max > 1.0
        ):
            raise ValueError(
                "signed_score_max must be within [-1, 1]"
            )

        if (
            self.signed_score_std is None
            or self.signed_score_std < 0.0
        ):
            raise ValueError(
                "signed_score_std must be nonnegative"
            )

        if (
            self.signed_score_min
            > self.signed_score_max
        ):
            raise ValueError(
                "signed_score_min cannot exceed "
                "signed_score_max"
            )


def aggregate_sentiments(
    sentiments: list[SentimentRecord],
    *,
    asset: str,
    cutoff_at_utc: datetime,
    window: SentimentWindow,
) -> SentimentAggregationRecord:
    """Aggregate one asset's sentiment over a trailing time window.

    The window is:

        (cutoff_at_utc - window, cutoff_at_utc]

    Eligibility uses provider_seen_at_utc so future provider
    availability cannot leak backward into an earlier feature cutoff.
    """

    if not isinstance(asset, str) or not asset.strip():
        raise ValueError("asset must be a nonblank string")

    _validate_utc_timestamp(
        cutoff_at_utc,
        field_name="cutoff_at_utc",
    )

    if not isinstance(window, SentimentWindow):
        raise TypeError(
            "window must be a SentimentWindow"
        )

    lower_boundary = (
        cutoff_at_utc
        - timedelta(hours=window.value)
    )

    eligible = [
        sentiment
        for sentiment in sentiments
        if (
            sentiment.asset == asset
            and lower_boundary
            < sentiment.provider_seen_at_utc
            <= cutoff_at_utc
        )
    ]

    if not eligible:
        return SentimentAggregationRecord(
            asset=asset,
            cutoff_at_utc=cutoff_at_utc,
            window_hours=window.value,
            article_count=0,
            signed_score_mean=None,
            signed_score_std=None,
            signed_score_min=None,
            signed_score_max=None,
            positive_probability_mean=None,
            neutral_probability_mean=None,
            negative_probability_mean=None,
            positive_label_share=None,
            neutral_label_share=None,
            negative_label_share=None,
        )

    signed_scores = [
        sentiment.signed_score
        for sentiment in eligible
    ]

    positive_probabilities = [
        sentiment.positive_probability
        for sentiment in eligible
    ]

    neutral_probabilities = [
        sentiment.neutral_probability
        for sentiment in eligible
    ]

    negative_probabilities = [
        sentiment.negative_probability
        for sentiment in eligible
    ]

    article_count = len(eligible)

    positive_count = sum(
        sentiment.predicted_label
        == SentimentLabel.POSITIVE
        for sentiment in eligible
    )

    neutral_count = sum(
        sentiment.predicted_label
        == SentimentLabel.NEUTRAL
        for sentiment in eligible
    )

    negative_count = sum(
        sentiment.predicted_label
        == SentimentLabel.NEGATIVE
        for sentiment in eligible
    )

    return SentimentAggregationRecord(
        asset=asset,
        cutoff_at_utc=cutoff_at_utc,
        window_hours=window.value,
        article_count=article_count,
        signed_score_mean=statistics.fmean(
            signed_scores
        ),
        signed_score_std=statistics.pstdev(
            signed_scores
        ),
        signed_score_min=min(signed_scores),
        signed_score_max=max(signed_scores),
        positive_probability_mean=statistics.fmean(
            positive_probabilities
        ),
        neutral_probability_mean=statistics.fmean(
            neutral_probabilities
        ),
        negative_probability_mean=statistics.fmean(
            negative_probabilities
        ),
        positive_label_share=(
            positive_count / article_count
        ),
        neutral_label_share=(
            neutral_count / article_count
        ),
        negative_label_share=(
            negative_count / article_count
        ),
    )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    """Require an aware timestamp normalized to UTC."""

    if not isinstance(value, datetime):
        raise TypeError(
            f"{field_name} must be a datetime"
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be timezone-aware"
        )

    if value.utcoffset().total_seconds() != 0:
        raise ValueError(
            f"{field_name} must be UTC"
        )
