"""Tests for validated canonical sentiment storage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from market_prediction.sentiment.fingerprint import (
    sentiment_dataset_fingerprint,
)
from market_prediction.sentiment.models import (
    SentimentLabel,
    SentimentRecord,
)
from market_prediction.sentiment.storage import (
    SentimentDatasetManifest,
    SentimentValidationGateError,
    ValidatedSentimentDataset,
    build_sentiment_manifest,
    deserialize_sentiment,
    serialize_sentiment,
    serialize_sentiments,
)

from market_prediction.sentiment.validation import (
    SentimentStatus,
    validate_sentiments,
)


UTC = timezone.utc

MODEL_NAME = "ProsusAI/finbert"
MODEL_REVISION = (
    "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
)


def make_sentiment(
    **overrides,
) -> SentimentRecord:
    """Return one valid canonical sentiment record."""

    published = datetime(
        2026,
        10,
        30,
        12,
        0,
        tzinfo=UTC,
    )

    values = {
        "sentiment_id": "a" * 64,
        "article_id": "b" * 64,
        "asset": "AAPL",
        "query_id": "aapl_company",
        "published_at_utc": published,
        "provider_seen_at_utc": (
            published + timedelta(minutes=1)
        ),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "scoring_method": "finbert",
        "positive_probability": 0.80,
        "neutral_probability": 0.15,
        "negative_probability": 0.05,
        "predicted_label": SentimentLabel.POSITIVE,
        "confidence": 0.80,
        "signed_score": 0.75,
        "scored_at_utc": (
            published + timedelta(minutes=2)
        ),
        "input_text_hash": "c" * 64,
        "schema_version": "1",
    }

    values.update(overrides)

    return SentimentRecord(**values)


def test_sentiment_round_trip_preserves_record() -> None:
    sentiment = make_sentiment()

    serialized = serialize_sentiment(sentiment)
    restored = deserialize_sentiment(serialized)

    assert restored == sentiment


def test_serialize_sentiment_returns_single_jsonl_line() -> None:
    serialized = serialize_sentiment(
        make_sentiment()
    )

    assert isinstance(serialized, str)
    assert "\n" not in serialized


def test_serialization_is_deterministic() -> None:
    sentiment = make_sentiment()

    assert (
        serialize_sentiment(sentiment)
        == serialize_sentiment(sentiment)
    )


def test_serialization_does_not_mutate_record() -> None:
    sentiment = make_sentiment()
    before = sentiment.model_dump()

    serialize_sentiment(sentiment)

    assert sentiment.model_dump() == before


def test_serialize_rejects_non_sentiment_record() -> None:
    with pytest.raises(TypeError):
        serialize_sentiment(
            {"sentiment_id": "bad"}  # type: ignore[arg-type]
        )


def test_deserialize_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        deserialize_sentiment("{not-valid-json")


def test_deserialize_rejects_non_object_json() -> None:
    with pytest.raises(ValueError):
        deserialize_sentiment(
            '["not", "a", "sentiment"]'
        )


def test_deserialize_rejects_invalid_schema() -> None:
    with pytest.raises(ValueError):
        deserialize_sentiment(
            '{"sentiment_id":"bad"}'
        )


def test_serialize_sentiments_returns_jsonl() -> None:
    first = make_sentiment()

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        input_text_hash="f" * 64,
    )

    serialized = serialize_sentiments(
        [first, second]
    )

    lines = serialized.splitlines()

    assert len(lines) == 2
    assert deserialize_sentiment(lines[0]) == first
    assert deserialize_sentiment(lines[1]) == second


def test_serialize_sentiments_preserves_input_order() -> None:
    first = make_sentiment()

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        input_text_hash="f" * 64,
    )

    serialized = serialize_sentiments(
        [second, first]
    )

    restored = [
        deserialize_sentiment(line)
        for line in serialized.splitlines()
    ]

    assert [
        item.sentiment_id
        for item in restored
    ] == [
        second.sentiment_id,
        first.sentiment_id,
    ]


def test_serialize_empty_sentiment_dataset_is_valid() -> None:
    assert serialize_sentiments([]) == ""


def test_serialize_sentiments_rejects_non_list() -> None:
    with pytest.raises(TypeError):
        serialize_sentiments(
            (make_sentiment(),)  # type: ignore[arg-type]
        )


def test_serialize_sentiments_rejects_invalid_member() -> None:
    with pytest.raises(TypeError):
        serialize_sentiments(
            [
                make_sentiment(),
                {"sentiment_id": "bad"},
            ]  # type: ignore[list-item]
        )


def test_dataset_fingerprint_is_deterministic() -> None:
    sentiments = [make_sentiment()]

    first = sentiment_dataset_fingerprint(
        sentiments
    )
    second = sentiment_dataset_fingerprint(
        sentiments
    )

    assert first == second
    assert len(first) == 64


def test_empty_dataset_has_deterministic_fingerprint() -> None:
    first = sentiment_dataset_fingerprint([])
    second = sentiment_dataset_fingerprint([])

    assert first == second
    assert len(first) == 64


def test_dataset_fingerprint_changes_with_content() -> None:
    first = make_sentiment()

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        positive_probability=0.05,
        neutral_probability=0.15,
        negative_probability=0.80,
        predicted_label=SentimentLabel.NEGATIVE,
        confidence=0.80,
        signed_score=-0.75,
    )

    assert (
        sentiment_dataset_fingerprint([first])
        != sentiment_dataset_fingerprint([second])
    )


def test_dataset_fingerprint_is_order_sensitive() -> None:
    first = make_sentiment()

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        input_text_hash="f" * 64,
    )

    assert (
        sentiment_dataset_fingerprint(
            [first, second]
        )
        != sentiment_dataset_fingerprint(
            [second, first]
        )
    )


def test_manifest_accepts_nonempty_dataset() -> None:
    manifest = SentimentDatasetManifest(
        dataset_fingerprint="a" * 64,
        sentiment_count=2,
        schema_version="1",
        created_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        scoring_method="finbert",
        assets=["AAPL", "MSFT"],
    )

    assert manifest.sentiment_count == 2
    assert manifest.assets == ["AAPL", "MSFT"]


def test_manifest_accepts_empty_dataset() -> None:
    manifest = SentimentDatasetManifest(
        dataset_fingerprint="a" * 64,
        sentiment_count=0,
        schema_version="1",
        created_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        scoring_method="finbert",
        assets=[],
    )

    assert manifest.sentiment_count == 0
    assert manifest.assets == []


def test_manifest_forbids_extra_fields() -> None:
    with pytest.raises(ValueError):
        SentimentDatasetManifest(
            dataset_fingerprint="a" * 64,
            sentiment_count=1,
            schema_version="1",
            created_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
                tzinfo=UTC,
            ),
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            scoring_method="finbert",
            assets=["AAPL"],
            unexpected="bad",
        )


def test_manifest_requires_utc_created_at() -> None:
    with pytest.raises(ValueError):
        SentimentDatasetManifest(
            dataset_fingerprint="a" * 64,
            sentiment_count=1,
            schema_version="1",
            created_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
            ),
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            scoring_method="finbert",
            assets=["AAPL"],
        )

def validate_dataset(
    sentiments: list[SentimentRecord],
):
    """Validate using the frozen FinBERT configuration."""

    return validate_sentiments(
        sentiments,
        allowed_assets={"AAPL", "MSFT"},
        allowed_query_ids={
            "aapl_company",
            "msft_company",
        },
        expected_model_name=MODEL_NAME,
        expected_model_revision=MODEL_REVISION,
        expected_scoring_method="finbert",
        expected_schema_version="1",
    )


def test_build_manifest_derives_dataset_metadata() -> None:
    sentiment = make_sentiment()

    created_at = datetime(
        2026,
        10,
        30,
        14,
        0,
        tzinfo=UTC,
    )

    manifest = build_sentiment_manifest(
        sentiments=[sentiment],
        created_at_utc=created_at,
    )

    assert manifest.dataset_fingerprint == (
        sentiment_dataset_fingerprint(
            [sentiment]
        )
    )
    assert manifest.sentiment_count == 1
    assert manifest.schema_version == "1"
    assert manifest.created_at_utc == created_at
    assert manifest.model_name == MODEL_NAME
    assert manifest.model_revision == MODEL_REVISION
    assert manifest.scoring_method == "finbert"
    assert manifest.assets == ["AAPL"]


def test_build_manifest_preserves_first_seen_asset_order() -> None:
    first = make_sentiment(
        asset="MSFT",
        query_id="msft_company",
    )

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        asset="AAPL",
        input_text_hash="f" * 64,
    )

    manifest = build_sentiment_manifest(
        sentiments=[first, second],
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    assert manifest.assets == [
        "MSFT",
        "AAPL",
    ]


def test_build_manifest_accepts_empty_dataset() -> None:
    manifest = build_sentiment_manifest(
        sentiments=[],
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
        empty_model_name=MODEL_NAME,
        empty_model_revision=MODEL_REVISION,
        empty_scoring_method="finbert",
        empty_schema_version="1",
    )

    assert manifest.sentiment_count == 0
    assert manifest.assets == []
    assert manifest.model_name == MODEL_NAME
    assert manifest.model_revision == MODEL_REVISION
    assert manifest.scoring_method == "finbert"
    assert manifest.schema_version == "1"


def test_build_manifest_rejects_mixed_model_names() -> None:
    first = make_sentiment()

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        model_name="other/model",
        input_text_hash="f" * 64,
    )

    with pytest.raises(ValueError):
        build_sentiment_manifest(
            sentiments=[first, second],
            created_at_utc=datetime(
                2026,
                10,
                30,
                14,
                0,
                tzinfo=UTC,
            ),
        )


def test_build_manifest_rejects_mixed_model_revisions() -> None:
    first = make_sentiment()

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        model_revision="d" * 40,
        input_text_hash="f" * 64,
    )

    with pytest.raises(ValueError):
        build_sentiment_manifest(
            sentiments=[first, second],
            created_at_utc=datetime(
                2026,
                10,
                30,
                14,
                0,
                tzinfo=UTC,
            ),
        )


def test_build_manifest_rejects_mixed_scoring_methods() -> None:
    first = make_sentiment()

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        scoring_method="vader",
        input_text_hash="f" * 64,
    )

    with pytest.raises(ValueError):
        build_sentiment_manifest(
            sentiments=[first, second],
            created_at_utc=datetime(
                2026,
                10,
                30,
                14,
                0,
                tzinfo=UTC,
            ),
        )


def test_build_manifest_rejects_mixed_schema_versions() -> None:
    first = make_sentiment()

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        schema_version="2",
        input_text_hash="f" * 64,
    )

    with pytest.raises(ValueError):
        build_sentiment_manifest(
            sentiments=[first, second],
            created_at_utc=datetime(
                2026,
                10,
                30,
                14,
                0,
                tzinfo=UTC,
            ),
        )


def test_validated_dataset_cannot_be_constructed_directly() -> None:
    sentiments = [make_sentiment()]
    report = validate_dataset(sentiments)

    with pytest.raises(
        SentimentValidationGateError
    ):
        ValidatedSentimentDataset(
            sentiments,
            report,
        )


def test_validated_dataset_accepts_passed_report() -> None:
    sentiments = [make_sentiment()]
    report = validate_dataset(sentiments)

    dataset = (
        ValidatedSentimentDataset.from_validation(
            sentiments,
            report,
        )
    )

    assert dataset.sentiments == sentiments
    assert dataset.report is report


def test_validated_dataset_accepts_empty_passed_dataset() -> None:
    sentiments: list[SentimentRecord] = []
    report = validate_dataset(sentiments)

    assert report.status is SentimentStatus.PASSED

    dataset = (
        ValidatedSentimentDataset.from_validation(
            sentiments,
            report,
        )
    )

    assert dataset.sentiments == []


def test_validated_dataset_rejects_blocked_report() -> None:
    sentiments = [
        make_sentiment(
            asset="UNKNOWN"
        )
    ]
    report = validate_dataset(sentiments)

    assert report.status is SentimentStatus.BLOCKED

    with pytest.raises(
        SentimentValidationGateError
    ):
        ValidatedSentimentDataset.from_validation(
            sentiments,
            report,
        )


def test_validated_dataset_rejects_report_count_mismatch() -> None:
    validated_sentiments = [
        make_sentiment()
    ]
    report = validate_dataset(
        validated_sentiments
    )

    different_sentiments = [
        make_sentiment(),
        make_sentiment(
            sentiment_id="d" * 64,
            article_id="e" * 64,
            input_text_hash="f" * 64,
        ),
    ]

    with pytest.raises(
        SentimentValidationGateError
    ):
        ValidatedSentimentDataset.from_validation(
            different_sentiments,
            report,
        )


def test_validated_dataset_returns_defensive_copy() -> None:
    sentiments = [make_sentiment()]
    report = validate_dataset(sentiments)

    dataset = (
        ValidatedSentimentDataset.from_validation(
            sentiments,
            report,
        )
    )

    returned = dataset.sentiments
    returned.clear()

    assert dataset.sentiments == sentiments

def test_validated_dataset_rejects_different_same_length_dataset() -> None:
    original = [
        make_sentiment()
    ]
    report = validate_dataset(original)

    different = [
        make_sentiment(
            sentiment_id="d" * 64,
            article_id="e" * 64,
            input_text_hash="f" * 64,
        )
    ]

    assert len(original) == len(different)

    with pytest.raises(
        SentimentValidationGateError
    ):
        ValidatedSentimentDataset.from_validation(
            different,
            report,
        )