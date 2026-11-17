"""Tests for article-level sentiment scoring service."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)
from market_prediction.sentiment.fingerprint import (
    sentiment_identity,
    sentiment_input_text_hash,
)
from market_prediction.sentiment.models import (
    SentimentRecord,
)
from market_prediction.sentiment.scorer import (
    SentimentScores,
)
from market_prediction.sentiment.service import (
    SentimentScoringService,
)


UTC = timezone.utc


class FakeScorer:
    """Deterministic scorer for service-boundary tests."""

    model_name = "test-finbert"
    model_revision = "abc123"
    scoring_method = "finbert"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def score(
        self,
        input_text: str,
    ) -> SentimentScores:
        self.calls.append(input_text)

        return SentimentScores(
            positive_probability=0.70,
            neutral_probability=0.20,
            negative_probability=0.10,
        )


class FakeClock:
    """Deterministic UTC scoring clock."""

    def __init__(
        self,
        now: datetime,
    ) -> None:
        self._now = now
        self.calls = 0

    def now_utc(self) -> datetime:
        self.calls += 1
        return self._now


def make_article(
    *,
    matched_assets: list[str] | None = None,
    snippet: str | None = (
        "Revenue increased and guidance was raised."
    ),
) -> ArticleRecord:
    return ArticleRecord(
        article_id="article-001",
        matched_assets=(
            matched_assets
            if matched_assets is not None
            else ["AAPL"]
        ),
        query_id="aapl-primary",
        title="Apple reports record quarterly revenue",
        snippet=snippet,
        full_text=None,
        canonical_url=(
            "https://example.com/apple-record-revenue"
        ),
        publisher="Example News",
        published_at_utc=datetime(
            2026,
            11,
            17,
            10,
            0,
            tzinfo=UTC,
        ),
        published_at_precision=(
            PublishedAtPrecision.MINUTE
        ),
        original_timezone="UTC",
        provider_seen_at_utc=datetime(
            2026,
            11,
            17,
            10,
            5,
            tzinfo=UTC,
        ),
        retrieved_at_utc=datetime(
            2026,
            11,
            17,
            10,
            10,
            tzinfo=UTC,
        ),
        provider="marketaux",
        provider_item_id="provider-001",
        language="en",
        content_hash="a" * 64,
        duplicate_group_id=None,
        raw_payload_reference=None,
        schema_version="1",
        timestamp_source=(
            TimestampSource.PROVIDER_PUBLICATION
        ),
        text_license_class=(
            TextLicenseClass.PROVIDER_METADATA
        ),
    )


def make_service(
    *,
    scorer: FakeScorer | None = None,
    scored_at: datetime | None = None,
) -> tuple[
    SentimentScoringService,
    FakeScorer,
    FakeClock,
]:
    resolved_scorer = scorer or FakeScorer()

    clock = FakeClock(
        scored_at
        or datetime(
            2026,
            11,
            17,
            11,
            0,
            tzinfo=UTC,
        )
    )

    service = SentimentScoringService(
        scorer=resolved_scorer,
        schema_version="1",
        clock=clock,
    )

    return service, resolved_scorer, clock


def test_score_article_returns_one_record_per_asset() -> None:
    service, _, _ = make_service()

    records = service.score_article(
        make_article(
            matched_assets=["AAPL", "MSFT"]
        )
    )

    assert len(records) == 2
    assert [record.asset for record in records] == [
        "AAPL",
        "MSFT",
    ]
    assert all(
        isinstance(record, SentimentRecord)
        for record in records
    )


def test_score_article_scores_text_once_for_multi_asset_article() -> None:
    service, scorer, _ = make_service()

    service.score_article(
        make_article(
            matched_assets=["AAPL", "MSFT"]
        )
    )

    assert scorer.calls == [
        (
            "Apple reports record quarterly revenue"
            "\n\n"
            "Revenue increased and guidance was raised."
        )
    ]


def test_score_article_uses_title_only_when_snippet_missing() -> None:
    service, scorer, _ = make_service()

    service.score_article(
        make_article(snippet=None)
    )

    assert scorer.calls == [
        "Apple reports record quarterly revenue"
    ]


def test_score_article_preserves_article_provenance() -> None:
    service, _, _ = make_service()
    article = make_article()

    record = service.score_article(article)[0]

    assert record.article_id == article.article_id
    assert record.asset == "AAPL"
    assert record.query_id == article.query_id
    assert (
        record.published_at_utc
        == article.published_at_utc
    )
    assert (
        record.provider_seen_at_utc
        == article.provider_seen_at_utc
    )


def test_score_article_preserves_scorer_metadata() -> None:
    service, scorer, _ = make_service()

    record = service.score_article(
        make_article()
    )[0]

    assert record.model_name == scorer.model_name
    assert (
        record.model_revision
        == scorer.model_revision
    )
    assert (
        record.scoring_method
        == scorer.scoring_method
    )


def test_score_article_preserves_all_scores() -> None:
    service, _, _ = make_service()

    record = service.score_article(
        make_article()
    )[0]

    assert record.positive_probability == pytest.approx(
        0.70
    )
    assert record.neutral_probability == pytest.approx(
        0.20
    )
    assert record.negative_probability == pytest.approx(
        0.10
    )
    assert record.signed_score == pytest.approx(
        0.60
    )
    assert record.confidence == pytest.approx(
        0.70
    )
    assert record.predicted_label.value == "positive"


def test_score_article_uses_one_scoring_timestamp() -> None:
    scored_at = datetime(
        2026,
        11,
        17,
        11,
        30,
        tzinfo=UTC,
    )

    service, _, clock = make_service(
        scored_at=scored_at
    )

    records = service.score_article(
        make_article(
            matched_assets=["AAPL", "MSFT"]
        )
    )

    assert clock.calls == 1
    assert all(
        record.scored_at_utc == scored_at
        for record in records
    )


def test_score_article_hashes_exact_constructed_input() -> None:
    service, _, _ = make_service()
    article = make_article()

    records = service.score_article(article)

    expected_text = (
        "Apple reports record quarterly revenue"
        "\n\n"
        "Revenue increased and guidance was raised."
    )

    expected_hash = sentiment_input_text_hash(
        expected_text
    )

    assert all(
        record.input_text_hash == expected_hash
        for record in records
    )


def test_score_article_builds_deterministic_identity_per_asset() -> None:
    service, scorer, _ = make_service()
    article = make_article(
        matched_assets=["AAPL", "MSFT"]
    )

    records = service.score_article(article)

    assert records[0].sentiment_id == sentiment_identity(
        article_id=article.article_id,
        asset="AAPL",
        model_name=scorer.model_name,
        model_revision=scorer.model_revision,
        scoring_method=scorer.scoring_method,
    )

    assert records[1].sentiment_id == sentiment_identity(
        article_id=article.article_id,
        asset="MSFT",
        model_name=scorer.model_name,
        model_revision=scorer.model_revision,
        scoring_method=scorer.scoring_method,
    )

    assert (
        records[0].sentiment_id
        != records[1].sentiment_id
    )


def test_score_article_preserves_asset_order() -> None:
    service, _, _ = make_service()

    records = service.score_article(
        make_article(
            matched_assets=[
                "MSFT",
                "AAPL",
                "GOLD",
            ]
        )
    )

    assert [record.asset for record in records] == [
        "MSFT",
        "AAPL",
        "GOLD",
    ]


def test_score_article_sets_sentiment_schema_version() -> None:
    service, _, _ = make_service()

    record = service.score_article(
        make_article()
    )[0]

    assert record.schema_version == "1"


def test_service_rejects_blank_schema_version() -> None:
    scorer = FakeScorer()
    clock = FakeClock(
        datetime(
            2026,
            11,
            17,
            11,
            0,
            tzinfo=UTC,
        )
    )

    with pytest.raises(
        ValueError,
        match="schema_version must not be blank",
    ):
        SentimentScoringService(
            scorer=scorer,
            schema_version=" ",
            clock=clock,
        )


def test_service_rejects_scoring_before_provider_seen() -> None:
    service, scorer, _ = make_service(
        scored_at=datetime(
            2026,
            11,
            17,
            10,
            4,
            tzinfo=UTC,
        )
    )

    with pytest.raises(
        ValueError,
        match="scored_at_utc cannot precede",
    ):
        service.score_article(
            make_article()
        )

    assert len(scorer.calls) == 1


def test_score_articles_preserves_article_and_asset_order() -> None:
    service, _, _ = make_service()

    first = make_article(
        matched_assets=["AAPL", "MSFT"]
    )

    second = first.model_copy(
        update={
            "article_id": "article-002",
            "matched_assets": ["GOLD"],
            "query_id": "gold-primary",
            "title": "Gold rises on market uncertainty",
        }
    )

    records = service.score_articles(
        [first, second]
    )

    assert [
        (record.article_id, record.asset)
        for record in records
    ] == [
        ("article-001", "AAPL"),
        ("article-001", "MSFT"),
        ("article-002", "GOLD"),
    ]


def test_score_articles_empty_input_returns_empty_list() -> None:
    service, scorer, clock = make_service()

    records = service.score_articles([])

    assert records == []
    assert scorer.calls == []
    assert clock.calls == 0