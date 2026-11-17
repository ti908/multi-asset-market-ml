"""Tests for atomic validated sentiment dataset persistence."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_prediction.sentiment.fingerprint import (
    sentiment_dataset_fingerprint,
)
from market_prediction.sentiment.models import (
    SentimentRecord,
)
from market_prediction.sentiment.storage import (
    SentimentStore,
    SentimentValidationGateError,
    ValidatedSentimentDataset,
    deserialize_sentiment,
)
from market_prediction.sentiment.validation import (
    validate_sentiments,
)


UTC = timezone.utc

MODEL_NAME = "ProsusAI/finbert"
MODEL_REVISION = (
    "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
)
SCORING_METHOD = "finbert"
SCHEMA_VERSION = "1"


def make_sentiment(
    **overrides,
) -> SentimentRecord:
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
        "scoring_method": SCORING_METHOD,
        "positive_probability": 0.7,
        "negative_probability": 0.1,
        "neutral_probability": 0.2,
        "predicted_label": "positive",
        "confidence": 0.7,
        "signed_score": 0.6,
        "scored_at_utc": (
            published + timedelta(minutes=2)
        ),
        "input_text_hash": "c" * 64,
        "schema_version": SCHEMA_VERSION,
    }

    values.update(overrides)

    return SentimentRecord(**values)


def make_validated_dataset(
    sentiments: list[SentimentRecord],
) -> ValidatedSentimentDataset:
    allowed_assets = {
        sentiment.asset
        for sentiment in sentiments
    }

    allowed_query_ids = {
        sentiment.query_id
        for sentiment in sentiments
    }

    if not sentiments:
        allowed_assets = {
            "AAPL",
            "MSFT",
        }
        allowed_query_ids = {
            "aapl_company",
            "msft_company",
        }

    report = validate_sentiments(
        sentiments,
        allowed_assets=allowed_assets,
        allowed_query_ids=allowed_query_ids,
        expected_model_name=MODEL_NAME,
        expected_model_revision=MODEL_REVISION,
        expected_scoring_method=SCORING_METHOD,
        expected_schema_version=SCHEMA_VERSION,
    )

    return ValidatedSentimentDataset.from_validation(
        sentiments,
        report,
    )


def make_store(
    tmp_path: Path,
) -> SentimentStore:
    return SentimentStore(
        storage_dir=tmp_path,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        scoring_method=SCORING_METHOD,
        schema_version=SCHEMA_VERSION,
    )


def test_write_publishes_complete_generation(
    tmp_path: Path,
) -> None:
    sentiment = make_sentiment()
    dataset = make_validated_dataset(
        [sentiment]
    )

    created_at = datetime(
        2026,
        10,
        30,
        14,
        0,
        tzinfo=UTC,
    )

    store = make_store(tmp_path)

    manifest = store.write(
        dataset,
        created_at_utc=created_at,
    )

    pointer_path = (
        tmp_path / "current.json"
    )

    assert pointer_path.is_file()

    pointer = json.loads(
        pointer_path.read_text(
            encoding="utf-8"
        )
    )

    assert set(pointer) == {
        "generation",
    }

    generation = pointer["generation"]

    assert isinstance(
        generation,
        str,
    )
    assert generation

    generation_dir = (
        tmp_path
        / "generations"
        / generation
    )

    sentiments_path = (
        generation_dir
        / "sentiments.jsonl"
    )

    manifest_path = (
        generation_dir
        / "manifest.json"
    )

    assert sentiments_path.is_file()
    assert manifest_path.is_file()

    persisted_lines = (
        sentiments_path.read_text(
            encoding="utf-8"
        ).splitlines()
    )

    assert len(persisted_lines) == 1

    assert (
        deserialize_sentiment(
            persisted_lines[0]
        )
        == sentiment
    )

    persisted_manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        persisted_manifest[
            "dataset_fingerprint"
        ]
        == sentiment_dataset_fingerprint(
            [sentiment]
        )
    )

    assert (
        persisted_manifest[
            "sentiment_count"
        ]
        == 1
    )

    assert (
        persisted_manifest[
            "created_at_utc"
        ]
        == "2026-10-30T14:00:00Z"
    )

    assert (
        manifest.dataset_fingerprint
        == dataset.dataset_id
    )

    assert manifest.sentiment_count == 1
    assert manifest.model_name == MODEL_NAME
    assert (
        manifest.model_revision
        == MODEL_REVISION
    )
    assert (
        manifest.scoring_method
        == SCORING_METHOD
    )
    assert manifest.assets == [
        "AAPL"
    ]


def test_write_publishes_empty_validated_dataset(
    tmp_path: Path,
) -> None:
    dataset = make_validated_dataset([])

    store = make_store(tmp_path)

    manifest = store.write(
        dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    pointer = json.loads(
        (
            tmp_path / "current.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    generation_dir = (
        tmp_path
        / "generations"
        / pointer["generation"]
    )

    sentiments_path = (
        generation_dir
        / "sentiments.jsonl"
    )

    assert sentiments_path.is_file()

    assert (
        sentiments_path.read_text(
            encoding="utf-8"
        )
        == ""
    )

    assert manifest.sentiment_count == 0
    assert manifest.assets == []

    assert (
        manifest.dataset_fingerprint
        == sentiment_dataset_fingerprint([])
    )

    assert manifest.model_name == MODEL_NAME
    assert (
        manifest.model_revision
        == MODEL_REVISION
    )
    assert (
        manifest.scoring_method
        == SCORING_METHOD
    )
    assert (
        manifest.schema_version
        == SCHEMA_VERSION
    )


def test_read_returns_empty_current_generation(
    tmp_path: Path,
) -> None:
    dataset = make_validated_dataset([])

    store = make_store(tmp_path)

    written_manifest = store.write(
        dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    loaded = store.read()

    assert loaded.sentiments == []
    assert (
        loaded.manifest
        == written_manifest
    )
    assert (
        loaded.dataset_id
        == sentiment_dataset_fingerprint([])
    )


def test_failed_replacement_preserves_previous_good_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = make_sentiment()

    first_dataset = make_validated_dataset(
        [first]
    )

    store = make_store(tmp_path)

    store.write(
        first_dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    pointer_path = (
        tmp_path / "current.json"
    )

    original_pointer = json.loads(
        pointer_path.read_text(
            encoding="utf-8"
        )
    )

    original_generation = (
        original_pointer["generation"]
    )

    second = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        asset="MSFT",
        query_id="msft_company",
        input_text_hash="f" * 64,
    )

    second_dataset = (
        make_validated_dataset(
            [second]
        )
    )

    real_replace = os.replace

    def fail_pointer_publish(
        src: str | Path,
        dst: str | Path,
    ) -> None:
        if Path(dst).name == "current.json":
            raise OSError(
                "simulated pointer "
                "publication failure"
            )

        real_replace(
            src,
            dst,
        )

    monkeypatch.setattr(
        os,
        "replace",
        fail_pointer_publish,
    )

    with pytest.raises(
        OSError,
        match=(
            "simulated pointer "
            "publication failure"
        ),
    ):
        store.write(
            second_dataset,
            created_at_utc=datetime(
                2026,
                10,
                30,
                15,
                0,
                tzinfo=UTC,
            ),
        )

    current_pointer = json.loads(
        pointer_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        current_pointer["generation"]
        == original_generation
    )


def test_write_rejects_raw_sentiment_list(
    tmp_path: Path,
) -> None:
    sentiment = make_sentiment()

    store = make_store(tmp_path)

    with pytest.raises(
        SentimentValidationGateError,
        match=(
            "accepts only "
            "ValidatedSentimentDataset"
        ),
    ):
        store.write(
            [sentiment],  # type: ignore[arg-type]
            created_at_utc=datetime(
                2026,
                10,
                30,
                14,
                0,
                tzinfo=UTC,
            ),
        )

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()


def test_write_rechecks_validated_content_before_publication(
    tmp_path: Path,
) -> None:
    sentiment = make_sentiment()

    dataset = make_validated_dataset(
        [sentiment]
    )

    replacement = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        input_text_hash="f" * 64,
    )

    object.__setattr__(
        dataset,
        "_sentiments",
        [replacement],
    )

    store = make_store(tmp_path)

    with pytest.raises(
        SentimentValidationGateError,
        match=(
            "no longer matches its "
            "validation fingerprint"
        ),
    ):
        store.write(
            dataset,
            created_at_utc=datetime(
                2026,
                10,
                30,
                14,
                0,
                tzinfo=UTC,
            ),
        )

    assert not (
        tmp_path / "current.json"
    ).exists()

    assert not (
        tmp_path / "generations"
    ).exists()


def test_read_returns_current_validated_generation(
    tmp_path: Path,
) -> None:
    sentiment = make_sentiment()

    dataset = make_validated_dataset(
        [sentiment]
    )

    store = make_store(tmp_path)

    written_manifest = store.write(
        dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    loaded = store.read()

    assert loaded.sentiments == [
        sentiment
    ]

    assert (
        loaded.manifest
        == written_manifest
    )

    assert (
        loaded.dataset_id
        == dataset.dataset_id
    )

    assert (
        loaded.manifest.dataset_fingerprint
        == sentiment_dataset_fingerprint(
            loaded.sentiments
        )
    )


def test_read_rejects_malformed_current_pointer(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        tmp_path / "current.json"
    ).write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    store = make_store(tmp_path)

    with pytest.raises(
        json.JSONDecodeError,
    ):
        store.read()


def test_read_rejects_pointer_with_unexpected_fields(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        tmp_path / "current.json"
    ).write_text(
        json.dumps(
            {
                "generation": "abc123",
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    store = make_store(tmp_path)

    with pytest.raises(
        ValueError,
        match=(
            "must contain exactly "
            "the generation field"
        ),
    ):
        store.read()


def test_read_rejects_missing_generation_directory(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        tmp_path / "current.json"
    ).write_text(
        json.dumps(
            {
                "generation": (
                    "missing-generation"
                ),
            }
        ),
        encoding="utf-8",
    )

    store = make_store(tmp_path)

    with pytest.raises(
        FileNotFoundError,
    ):
        store.read()


def test_read_rejects_missing_sentiments_file(
    tmp_path: Path,
) -> None:
    generation = "generation-1"

    generation_dir = (
        tmp_path
        / "generations"
        / generation
    )

    generation_dir.mkdir(
        parents=True,
    )

    (
        tmp_path / "current.json"
    ).write_text(
        json.dumps(
            {
                "generation": generation,
            }
        ),
        encoding="utf-8",
    )

    (
        generation_dir
        / "manifest.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    store = make_store(tmp_path)

    with pytest.raises(
        FileNotFoundError,
    ):
        store.read()


def test_read_rejects_missing_manifest_file(
    tmp_path: Path,
) -> None:
    generation = "generation-1"

    generation_dir = (
        tmp_path
        / "generations"
        / generation
    )

    generation_dir.mkdir(
        parents=True,
    )

    (
        tmp_path / "current.json"
    ).write_text(
        json.dumps(
            {
                "generation": generation,
            }
        ),
        encoding="utf-8",
    )

    (
        generation_dir
        / "sentiments.jsonl"
    ).write_text(
        "",
        encoding="utf-8",
    )

    store = make_store(tmp_path)

    with pytest.raises(
        FileNotFoundError,
    ):
        store.read()

def current_generation_dir(
    tmp_path: Path,
) -> Path:
    pointer = json.loads(
        (
            tmp_path / "current.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    return (
        tmp_path
        / "generations"
        / pointer["generation"]
    )


def test_read_rejects_malformed_sentiment_json(
    tmp_path: Path,
) -> None:
    sentiment = make_sentiment()
    dataset = make_validated_dataset(
        [sentiment]
    )

    store = make_store(tmp_path)

    store.write(
        dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    sentiments_path = (
        current_generation_dir(tmp_path)
        / "sentiments.jsonl"
    )

    sentiments_path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="not valid JSON",
    ):
        store.read()


def test_read_rejects_malformed_manifest_json(
    tmp_path: Path,
) -> None:
    sentiment = make_sentiment()
    dataset = make_validated_dataset(
        [sentiment]
    )

    store = make_store(tmp_path)

    store.write(
        dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    manifest_path = (
        current_generation_dir(tmp_path)
        / "manifest.json"
    )

    manifest_path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        json.JSONDecodeError,
    ):
        store.read()


def test_read_rejects_content_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    sentiment = make_sentiment()
    dataset = make_validated_dataset(
        [sentiment]
    )

    store = make_store(tmp_path)

    store.write(
        dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    sentiments_path = (
        current_generation_dir(tmp_path)
        / "sentiments.jsonl"
    )

    changed = make_sentiment(
        sentiment_id="d" * 64,
        article_id="e" * 64,
        input_text_hash="f" * 64,
    )

    from market_prediction.sentiment.storage import (
        serialize_sentiment,
    )

    sentiments_path.write_text(
        serialize_sentiment(changed),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "fingerprint does not "
            "match its manifest"
        ),
    ):
        store.read()


def test_read_rejects_manifest_sentiment_count_mismatch(
    tmp_path: Path,
) -> None:
    sentiment = make_sentiment()
    dataset = make_validated_dataset(
        [sentiment]
    )

    store = make_store(tmp_path)

    store.write(
        dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    manifest_path = (
        current_generation_dir(tmp_path)
        / "manifest.json"
    )

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    payload["sentiment_count"] = 2

    manifest_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "sentiment count does not "
            "match its manifest"
        ),
    ):
        store.read()


@pytest.mark.parametrize(
    (
        "field_name",
        "bad_value",
        "expected_message",
    ),
    [
        (
            "schema_version",
            "999",
            "schema version does not match",
        ),
        (
            "model_name",
            "other/model",
            "model name does not match",
        ),
        (
            "model_revision",
            "d" * 40,
            "model revision does not match",
        ),
        (
            "scoring_method",
            "vader",
            "scoring method does not match",
        ),
        (
            "assets",
            ["MSFT"],
            "assets do not match",
        ),
    ],
)
def test_read_rejects_manifest_metadata_mismatch(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
    expected_message: str,
) -> None:
    sentiment = make_sentiment()
    dataset = make_validated_dataset(
        [sentiment]
    )

    store = make_store(tmp_path)

    store.write(
        dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    manifest_path = (
        current_generation_dir(tmp_path)
        / "manifest.json"
    )

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    payload[field_name] = bad_value

    manifest_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        store.read()


def test_read_rejects_empty_dataset_with_wrong_provenance(
    tmp_path: Path,
) -> None:
    dataset = make_validated_dataset([])

    store = make_store(tmp_path)

    store.write(
        dataset,
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=UTC,
        ),
    )

    manifest_path = (
        current_generation_dir(tmp_path)
        / "manifest.json"
    )

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    payload["model_name"] = "other/model"

    manifest_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="model name does not match",
    ):
        store.read()