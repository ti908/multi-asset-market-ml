from datetime import datetime, timedelta, timezone

import pytest

from market_prediction.sentiment.aggregation import (
    SentimentAggregationRecord,
    SentimentWindow,
    aggregate_sentiments,
)
from market_prediction.sentiment.models import (
    SentimentLabel,
    SentimentRecord,
)


UTC = timezone.utc
MODEL_NAME = "ProsusAI/finbert"
MODEL_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
SCORING_METHOD = "finbert"
SCHEMA_VERSION = "1"


def utc_datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
) -> datetime:
    return datetime(
        year,
        month,
        day,
        hour,
        minute,
        tzinfo=UTC,
    )


def make_sentiment(
    *,
    sentiment_id: str,
    article_id: str,
    asset: str = "AAPL",
    provider_seen_at_utc: datetime,
    positive_probability: float,
    neutral_probability: float,
    negative_probability: float,
    predicted_label: SentimentLabel,
) -> SentimentRecord:
    published_at_utc = provider_seen_at_utc - timedelta(minutes=5)

    return SentimentRecord(
        sentiment_id=sentiment_id,
        article_id=article_id,
        asset=asset,
        query_id=f"{asset.lower()}-strict",
        published_at_utc=published_at_utc,
        provider_seen_at_utc=provider_seen_at_utc,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        scoring_method=SCORING_METHOD,
        positive_probability=positive_probability,
        neutral_probability=neutral_probability,
        negative_probability=negative_probability,
        predicted_label=predicted_label,
        confidence=max(
            positive_probability,
            neutral_probability,
            negative_probability,
        ),
        signed_score=(
            positive_probability - negative_probability
        ),
        scored_at_utc=provider_seen_at_utc + timedelta(minutes=1),
        input_text_hash="a" * 64,
        schema_version=SCHEMA_VERSION,
    )


def test_sentiment_windows_are_frozen() -> None:
    assert SentimentWindow.SIX_HOURS.value == 6
    assert SentimentWindow.TWENTY_FOUR_HOURS.value == 24
    assert SentimentWindow.SEVENTY_TWO_HOURS.value == 72


def test_aggregation_record_rejects_naive_cutoff() -> None:
    with pytest.raises(ValueError):
        SentimentAggregationRecord(
            asset="AAPL",
            cutoff_at_utc=datetime(2026, 11, 17, 12),
            window_hours=6,
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


def test_empty_window_preserves_missing_sentiment() -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)

    records = aggregate_sentiments(
        [],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    assert records.article_count == 0

    assert records.signed_score_mean is None
    assert records.signed_score_std is None
    assert records.signed_score_min is None
    assert records.signed_score_max is None

    assert records.positive_probability_mean is None
    assert records.neutral_probability_mean is None
    assert records.negative_probability_mean is None

    assert records.positive_label_share is None
    assert records.neutral_label_share is None
    assert records.negative_label_share is None


def test_window_uses_provider_seen_time_not_publication_time() -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)

    sentiment = make_sentiment(
        sentiment_id="sent-1",
        article_id="article-1",
        provider_seen_at_utc=cutoff + timedelta(minutes=1),
        positive_probability=0.8,
        neutral_probability=0.1,
        negative_probability=0.1,
        predicted_label=SentimentLabel.POSITIVE,
    )

    # Publication occurred before cutoff, but provider_seen is after it.
    assert sentiment.published_at_utc < cutoff
    assert sentiment.provider_seen_at_utc > cutoff

    result = aggregate_sentiments(
        [sentiment],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    assert result.article_count == 0


def test_cutoff_is_inclusive() -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)

    sentiment = make_sentiment(
        sentiment_id="sent-1",
        article_id="article-1",
        provider_seen_at_utc=cutoff,
        positive_probability=0.8,
        neutral_probability=0.1,
        negative_probability=0.1,
        predicted_label=SentimentLabel.POSITIVE,
    )

    result = aggregate_sentiments(
        [sentiment],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    assert result.article_count == 1


def test_lower_window_boundary_is_exclusive() -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)
    lower_boundary = cutoff - timedelta(hours=6)

    sentiment = make_sentiment(
        sentiment_id="sent-1",
        article_id="article-1",
        provider_seen_at_utc=lower_boundary,
        positive_probability=0.8,
        neutral_probability=0.1,
        negative_probability=0.1,
        predicted_label=SentimentLabel.POSITIVE,
    )

    result = aggregate_sentiments(
        [sentiment],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    assert result.article_count == 0


def test_record_immediately_after_lower_boundary_is_included() -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)
    lower_boundary = cutoff - timedelta(hours=6)

    sentiment = make_sentiment(
        sentiment_id="sent-1",
        article_id="article-1",
        provider_seen_at_utc=lower_boundary + timedelta(microseconds=1),
        positive_probability=0.8,
        neutral_probability=0.1,
        negative_probability=0.1,
        predicted_label=SentimentLabel.POSITIVE,
    )

    result = aggregate_sentiments(
        [sentiment],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    assert result.article_count == 1


def test_other_assets_are_excluded() -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)

    msft = make_sentiment(
        sentiment_id="sent-msft",
        article_id="article-msft",
        asset="MSFT",
        provider_seen_at_utc=cutoff - timedelta(hours=1),
        positive_probability=0.8,
        neutral_probability=0.1,
        negative_probability=0.1,
        predicted_label=SentimentLabel.POSITIVE,
    )

    result = aggregate_sentiments(
        [msft],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    assert result.article_count == 0


def test_single_observation_has_zero_population_std() -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)

    sentiment = make_sentiment(
        sentiment_id="sent-1",
        article_id="article-1",
        provider_seen_at_utc=cutoff - timedelta(hours=1),
        positive_probability=0.8,
        neutral_probability=0.1,
        negative_probability=0.1,
        predicted_label=SentimentLabel.POSITIVE,
    )

    result = aggregate_sentiments(
        [sentiment],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    assert result.signed_score_std == pytest.approx(0.0)


def test_aggregation_calculates_expected_features() -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)

    sentiments = [
        make_sentiment(
            sentiment_id="sent-1",
            article_id="article-1",
            provider_seen_at_utc=cutoff - timedelta(hours=1),
            positive_probability=0.8,
            neutral_probability=0.1,
            negative_probability=0.1,
            predicted_label=SentimentLabel.POSITIVE,
        ),
        make_sentiment(
            sentiment_id="sent-2",
            article_id="article-2",
            provider_seen_at_utc=cutoff - timedelta(hours=2),
            positive_probability=0.1,
            neutral_probability=0.2,
            negative_probability=0.7,
            predicted_label=SentimentLabel.NEGATIVE,
        ),
        make_sentiment(
            sentiment_id="sent-3",
            article_id="article-3",
            provider_seen_at_utc=cutoff - timedelta(hours=3),
            positive_probability=0.2,
            neutral_probability=0.6,
            negative_probability=0.2,
            predicted_label=SentimentLabel.NEUTRAL,
        ),
    ]

    result = aggregate_sentiments(
        sentiments,
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    # Signed scores: 0.7, -0.6, 0.0
    assert result.article_count == 3

    assert result.signed_score_mean == pytest.approx(
        (0.7 - 0.6 + 0.0) / 3
    )
    assert result.signed_score_min == pytest.approx(-0.6)
    assert result.signed_score_max == pytest.approx(0.7)

    # Population standard deviation.
    expected_mean = (0.7 - 0.6 + 0.0) / 3
    expected_variance = (
        (0.7 - expected_mean) ** 2
        + (-0.6 - expected_mean) ** 2
        + (0.0 - expected_mean) ** 2
    ) / 3

    assert result.signed_score_std == pytest.approx(
        expected_variance ** 0.5
    )

    assert result.positive_probability_mean == pytest.approx(
        (0.8 + 0.1 + 0.2) / 3
    )
    assert result.neutral_probability_mean == pytest.approx(
        (0.1 + 0.2 + 0.6) / 3
    )
    assert result.negative_probability_mean == pytest.approx(
        (0.1 + 0.7 + 0.2) / 3
    )

    assert result.positive_label_share == pytest.approx(1 / 3)
    assert result.neutral_label_share == pytest.approx(1 / 3)
    assert result.negative_label_share == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("window", "hours"),
    [
        (SentimentWindow.SIX_HOURS, 6),
        (SentimentWindow.TWENTY_FOUR_HOURS, 24),
        (SentimentWindow.SEVENTY_TWO_HOURS, 72),
    ],
)
def test_each_frozen_window_applies_its_own_boundary(
    window: SentimentWindow,
    hours: int,
) -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)

    inside = make_sentiment(
        sentiment_id=f"inside-{hours}",
        article_id=f"inside-article-{hours}",
        provider_seen_at_utc=(
            cutoff
            - timedelta(hours=hours)
            + timedelta(seconds=1)
        ),
        positive_probability=0.8,
        neutral_probability=0.1,
        negative_probability=0.1,
        predicted_label=SentimentLabel.POSITIVE,
    )

    outside = make_sentiment(
        sentiment_id=f"outside-{hours}",
        article_id=f"outside-article-{hours}",
        provider_seen_at_utc=(
            cutoff
            - timedelta(hours=hours)
            - timedelta(seconds=1)
        ),
        positive_probability=0.1,
        neutral_probability=0.1,
        negative_probability=0.8,
        predicted_label=SentimentLabel.NEGATIVE,
    )

    result = aggregate_sentiments(
        [outside, inside],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=window,
    )

    assert result.article_count == 1
    assert result.signed_score_mean == pytest.approx(0.7)


def test_aggregation_is_independent_of_input_order() -> None:
    cutoff = utc_datetime(2026, 11, 17, 12)

    first = make_sentiment(
        sentiment_id="sent-1",
        article_id="article-1",
        provider_seen_at_utc=cutoff - timedelta(hours=1),
        positive_probability=0.8,
        neutral_probability=0.1,
        negative_probability=0.1,
        predicted_label=SentimentLabel.POSITIVE,
    )

    second = make_sentiment(
        sentiment_id="sent-2",
        article_id="article-2",
        provider_seen_at_utc=cutoff - timedelta(hours=2),
        positive_probability=0.1,
        neutral_probability=0.2,
        negative_probability=0.7,
        predicted_label=SentimentLabel.NEGATIVE,
    )

    forward = aggregate_sentiments(
        [first, second],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    reverse = aggregate_sentiments(
        [second, first],
        asset="AAPL",
        cutoff_at_utc=cutoff,
        window=SentimentWindow.SIX_HOURS,
    )

    assert forward == reverse


def test_non_utc_cutoff_is_rejected() -> None:
    non_utc = timezone(timedelta(hours=1))

    with pytest.raises(ValueError):
        aggregate_sentiments(
            [],
            asset="AAPL",
            cutoff_at_utc=datetime(
                2026,
                11,
                17,
                12,
                tzinfo=non_utc,
            ),
            window=SentimentWindow.SIX_HOURS,
        )