"""Article-level sentiment scoring service.

This module connects canonical news articles to canonical sentiment
observations without performing storage, temporal aggregation, price
alignment, feature engineering, or modeling.
"""

from __future__ import annotations

from typing import Any, Protocol

from market_prediction.news.models import ArticleRecord
from market_prediction.sentiment.fingerprint import (
    sentiment_identity,
    sentiment_input_text_hash,
)
from market_prediction.sentiment.models import SentimentRecord
from market_prediction.sentiment.scorer import (
    SentimentScorer,
    build_sentiment_input_text,
)


class UtcClock(Protocol):
    """Clock contract used to timestamp sentiment observations."""

    def now_utc(self):
        """Return the current timezone-aware UTC datetime."""
        ...


class SentimentScoringService:
    """Convert canonical news articles into sentiment observations."""

    def __init__(
        self,
        *,
        scorer: SentimentScorer | Any,
        schema_version: str,
        clock: UtcClock,
    ) -> None:
        """Initialize the article-level scoring service."""

        if not isinstance(schema_version, str):
            raise TypeError(
                "schema_version must be a string"
            )

        if not schema_version.strip():
            raise ValueError(
                "schema_version must not be blank"
            )

        self._scorer = scorer
        self._schema_version = schema_version
        self._clock = clock

    def score_article(
        self,
        article: ArticleRecord,
    ) -> list[SentimentRecord]:
        """Score one article once and emit one record per asset."""

        input_text = build_sentiment_input_text(
            title=article.title,
            snippet=article.snippet,
        )

        input_text_hash = sentiment_input_text_hash(
            input_text
        )

        scores = self._scorer.score(input_text)

        scored_at_utc = self._clock.now_utc()

        records: list[SentimentRecord] = []

        for asset in article.matched_assets:
            record = SentimentRecord(
                sentiment_id=sentiment_identity(
                    article_id=article.article_id,
                    asset=asset,
                    model_name=self._scorer.model_name,
                    model_revision=(
                        self._scorer.model_revision
                    ),
                    scoring_method=(
                        self._scorer.scoring_method
                    ),
                ),
                article_id=article.article_id,
                asset=asset,
                query_id=article.query_id,
                published_at_utc=(
                    article.published_at_utc
                ),
                provider_seen_at_utc=(
                    article.provider_seen_at_utc
                ),
                model_name=self._scorer.model_name,
                model_revision=(
                    self._scorer.model_revision
                ),
                scoring_method=(
                    self._scorer.scoring_method
                ),
                positive_probability=(
                    scores.positive_probability
                ),
                negative_probability=(
                    scores.negative_probability
                ),
                neutral_probability=(
                    scores.neutral_probability
                ),
                predicted_label=(
                    scores.predicted_label
                ),
                confidence=scores.confidence,
                signed_score=scores.signed_score,
                scored_at_utc=scored_at_utc,
                input_text_hash=input_text_hash,
                schema_version=self._schema_version,
            )

            records.append(record)

        return records

    def score_articles(
        self,
        articles: list[ArticleRecord],
    ) -> list[SentimentRecord]:
        """Score articles in deterministic input order."""

        records: list[SentimentRecord] = []

        for article in articles:
            records.extend(
                self.score_article(article)
            )

        return records