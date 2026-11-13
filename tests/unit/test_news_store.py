"""Tests for atomic validated news dataset persistence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)

from market_prediction.news.validation import (
    validate_articles,
)

import os
import pytest

from market_prediction.news.storage import (
    NewsStore,
    NewsValidationGateError,
    ValidatedNewsDataset,
    deserialize_article,
    news_dataset_fingerprint,
    serialize_article,
)

UTC = timezone.utc


def make_article(
    **overrides,
) -> ArticleRecord:
    published = datetime(
        2026,
        10,
        30,
        12,
        0,
        tzinfo=UTC,
    )

    values = {
        "article_id": "a" * 64,
        "matched_assets": ["AAPL"],
        "query_id": "aapl_company",
        "title": "Apple reports quarterly earnings",
        "snippet": "Revenue increased during the quarter.",
        "full_text": None,
        "canonical_url": (
            "https://example.com/apple-earnings"
        ),
        "publisher": "Example News",
        "published_at_utc": published,
        "published_at_precision": (
            PublishedAtPrecision.SECOND
        ),
        "original_timezone": "UTC",
        "provider_seen_at_utc": (
            published + timedelta(minutes=1)
        ),
        "retrieved_at_utc": (
            published + timedelta(minutes=2)
        ),
        "provider": "marketaux",
        "provider_item_id": "article-123",
        "language": "en",
        "content_hash": "b" * 64,
        "duplicate_group_id": None,
        "raw_payload_reference": None,
        "schema_version": "1",
        "timestamp_source": (
            TimestampSource.PROVIDER_PUBLICATION
        ),
        "text_license_class": (
            TextLicenseClass.PROVIDER_METADATA
        ),
    }

    values.update(overrides)

    return ArticleRecord(**values)


def make_validated_dataset(
    articles: list[ArticleRecord],
) -> ValidatedNewsDataset:
    report = validate_articles(
        articles,
        allowed_assets={
            asset
            for article in articles
            for asset in article.matched_assets
        },
        allowed_query_ids={
            article.query_id
            for article in articles
        },
        allowed_languages={
            article.language
            for article in articles
        },
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=UTC,
        ),
        max_future_seconds=0,
    )

    return ValidatedNewsDataset.from_validation(
        articles,
        report,
    )


def test_write_publishes_complete_generation(
    tmp_path: Path,
) -> None:
    article = make_article()
    dataset = make_validated_dataset(
        [article]
    )

    created_at = datetime(
        2026,
        10,
        30,
        14,
        0,
        tzinfo=UTC,
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    articles_path = (
        generation_dir
        / "articles.jsonl"
    )
    manifest_path = (
        generation_dir
        / "manifest.json"
    )

    assert articles_path.is_file()
    assert manifest_path.is_file()

    persisted_lines = (
        articles_path.read_text(
            encoding="utf-8"
        ).splitlines()
    )

    assert len(persisted_lines) == 1
    assert (
        deserialize_article(
            persisted_lines[0]
        )
        == article
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
        == news_dataset_fingerprint(
            [article]
        )
    )
    assert (
        persisted_manifest[
            "article_count"
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
    assert manifest.article_count == 1
    assert manifest.providers == [
        "marketaux"
    ]
    assert manifest.assets == [
        "AAPL"
    ]

def test_failed_replacement_preserves_previous_good_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_article = make_article()

    first_dataset = make_validated_dataset(
        [first_article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    second_article = make_article(
        article_id="c" * 64,
        provider_item_id="article-456",
        canonical_url=(
            "https://example.com/"
            "microsoft-earnings"
        ),
        matched_assets=["MSFT"],
        query_id="msft_company",
        title=(
            "Microsoft reports "
            "quarterly earnings"
        ),
    )

    second_dataset = make_validated_dataset(
        [second_article]
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

    original_generation_dir = (
        tmp_path
        / "generations"
        / original_generation
    )

    assert (
        original_generation_dir
        / "articles.jsonl"
    ).is_file()

    assert (
        original_generation_dir
        / "manifest.json"
    ).is_file()

def test_write_rejects_raw_article_list(
    tmp_path: Path,
) -> None:
    article = make_article()

    store = NewsStore(
        storage_dir=tmp_path,
    )

    with pytest.raises(
        NewsValidationGateError,
        match=(
            "accepts only "
            "ValidatedNewsDataset"
        ),
    ):
        store.write(
            [article],  # type: ignore[arg-type]
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
    article = make_article()

    dataset = make_validated_dataset(
        [article]
    )

    replacement = make_article(
        article_id="d" * 64,
        provider_item_id="article-789",
        canonical_url=(
            "https://example.com/"
            "changed-after-validation"
        ),
        title=(
            "Changed after validation"
        ),
    )

    object.__setattr__(
        dataset,
        "_articles",
        [replacement],
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    with pytest.raises(
        NewsValidationGateError,
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
    article = make_article()

    dataset = make_validated_dataset(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    assert loaded.articles == [
        article
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
        == news_dataset_fingerprint(
            loaded.articles
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

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    store = NewsStore(
        storage_dir=tmp_path,
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        store.read()


def test_read_rejects_missing_articles_file(
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

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    article = make_article()

    (
        generation_dir
        / "articles.jsonl"
    ).write_text(
        serialize_article(article),
        encoding="utf-8",
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        store.read()

def test_read_rejects_malformed_article_json(
    tmp_path: Path,
) -> None:
    article = make_article()
    dataset = make_validated_dataset(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    pointer = json.loads(
        (
            tmp_path / "current.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    articles_path = (
        tmp_path
        / "generations"
        / pointer["generation"]
        / "articles.jsonl"
    )

    articles_path.write_text(
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
    article = make_article()
    dataset = make_validated_dataset(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    pointer = json.loads(
        (
            tmp_path / "current.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    manifest_path = (
        tmp_path
        / "generations"
        / pointer["generation"]
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


def test_read_rejects_article_content_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    article = make_article()
    dataset = make_validated_dataset(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    pointer = json.loads(
        (
            tmp_path / "current.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    articles_path = (
        tmp_path
        / "generations"
        / pointer["generation"]
        / "articles.jsonl"
    )

    changed_article = make_article(
        article_id="e" * 64,
        provider_item_id="changed-article",
        canonical_url=(
            "https://example.com/"
            "changed-persisted-article"
        ),
        title="Changed persisted article",
    )

    articles_path.write_text(
        serialize_article(
            changed_article
        ),
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


def test_read_rejects_manifest_article_count_mismatch(
    tmp_path: Path,
) -> None:
    article = make_article()
    dataset = make_validated_dataset(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    pointer = json.loads(
        (
            tmp_path / "current.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    manifest_path = (
        tmp_path
        / "generations"
        / pointer["generation"]
        / "manifest.json"
    )

    manifest_payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    manifest_payload[
        "article_count"
    ] = 2

    manifest_path.write_text(
        json.dumps(
            manifest_payload
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "article count does not "
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
            "providers",
            ["gdelt"],
            "providers do not match",
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
    article = make_article()
    dataset = make_validated_dataset(
        [article]
    )

    store = NewsStore(
        storage_dir=tmp_path,
    )

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

    pointer = json.loads(
        (
            tmp_path / "current.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    manifest_path = (
        tmp_path
        / "generations"
        / pointer["generation"]
        / "manifest.json"
    )

    manifest_payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    manifest_payload[
        field_name
    ] = bad_value

    manifest_path.write_text(
        json.dumps(
            manifest_payload
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        store.read()