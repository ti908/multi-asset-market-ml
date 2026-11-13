from datetime import datetime, timezone

import pytest

from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)
from market_prediction.news.storage import (
    NewsDatasetManifest,
    build_news_manifest,
    deserialize_article,
    news_dataset_fingerprint,
    serialize_article,
    serialize_articles,
)


def make_article() -> ArticleRecord:
    return ArticleRecord(
        article_id="article-1",
        matched_assets=["AAPL"],
        query_id="aapl-primary",
        title="Apple announces new product",
        snippet="Example snippet",
        full_text=None,
        canonical_url=(
            "https://example.com/apple-story"
        ),
        publisher="Example Publisher",
        published_at_utc=datetime(
            2026,
            10,
            30,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        published_at_precision=(
            PublishedAtPrecision.SECOND
        ),
        original_timezone="UTC",
        provider_seen_at_utc=datetime(
            2026,
            10,
            30,
            12,
            1,
            tzinfo=timezone.utc,
        ),
        retrieved_at_utc=datetime(
            2026,
            10,
            30,
            12,
            2,
            tzinfo=timezone.utc,
        ),
        provider="marketaux",
        provider_item_id="marketaux-1",
        language="en",
        content_hash="a" * 64,
        duplicate_group_id="b" * 64,
        raw_payload_reference=None,
        schema_version="1",
        timestamp_source=(
            TimestampSource.PROVIDER_PUBLICATION
        ),
        text_license_class=(
            TextLicenseClass.PROVIDER_METADATA
        ),
    )


def test_article_round_trip_preserves_record() -> None:
    article = make_article()

    serialized = serialize_article(article)
    restored = deserialize_article(serialized)

    assert restored == article


def test_serialize_article_returns_single_jsonl_line() -> None:
    serialized = serialize_article(
        make_article()
    )

    assert isinstance(serialized, str)
    assert "\n" not in serialized


def test_serialization_is_deterministic() -> None:
    article = make_article()

    first = serialize_article(article)
    second = serialize_article(article)

    assert first == second


def test_serialization_does_not_mutate_article() -> None:
    article = make_article()
    before = article.model_dump()

    serialize_article(article)

    assert article.model_dump() == before


def test_serialize_rejects_non_article_record() -> None:
    with pytest.raises(TypeError):
        serialize_article(
            {"article_id": "article-1"}  # type: ignore[arg-type]
        )


def test_deserialize_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        deserialize_article(
            "{not-valid-json"
        )


def test_deserialize_rejects_non_object_json() -> None:
    with pytest.raises(ValueError):
        deserialize_article(
            '["not", "an", "article"]'
        )


def test_deserialize_rejects_invalid_article_schema() -> None:
    with pytest.raises(ValueError):
        deserialize_article(
            '{"article_id":"article-1"}'
        )

def test_serialize_articles_returns_jsonl() -> None:
    first = make_article()

    second = first.model_copy(
        update={
            "article_id": "article-2",
            "provider_item_id": "marketaux-2",
            "canonical_url": (
                "https://example.com/apple-story-2"
            ),
        }
    )

    serialized = serialize_articles(
        [first, second]
    )

    lines = serialized.splitlines()

    assert len(lines) == 2
    assert deserialize_article(lines[0]) == first
    assert deserialize_article(lines[1]) == second


def test_serialize_articles_preserves_input_order() -> None:
    first = make_article()

    second = first.model_copy(
        update={
            "article_id": "article-2",
            "provider_item_id": "marketaux-2",
        }
    )

    serialized = serialize_articles(
        [second, first]
    )

    restored = [
        deserialize_article(line)
        for line in serialized.splitlines()
    ]

    assert [
        article.article_id
        for article in restored
    ] == [
        "article-2",
        "article-1",
    ]


def test_serialize_articles_is_deterministic() -> None:
    articles = [make_article()]

    assert (
        serialize_articles(articles)
        == serialize_articles(articles)
    )


def test_serialize_articles_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError):
        serialize_articles([])


def test_serialize_articles_rejects_non_list() -> None:
    with pytest.raises(TypeError):
        serialize_articles(
            (make_article(),)  # type: ignore[arg-type]
        )


def test_serialize_articles_rejects_invalid_member() -> None:
    with pytest.raises(TypeError):
        serialize_articles(
            [
                make_article(),
                {"article_id": "bad"},
            ]  # type: ignore[list-item]
        )


def test_news_dataset_fingerprint_is_deterministic() -> None:
    articles = [make_article()]

    first = news_dataset_fingerprint(
        articles
    )
    second = news_dataset_fingerprint(
        articles
    )

    assert first == second
    assert len(first) == 64


def test_news_dataset_fingerprint_changes_with_content() -> None:
    first = make_article()

    second = first.model_copy(
        update={
            "title": "Different title",
        }
    )

    assert (
        news_dataset_fingerprint([first])
        != news_dataset_fingerprint([second])
    )


def test_news_dataset_fingerprint_is_order_sensitive() -> None:
    first = make_article()

    second = first.model_copy(
        update={
            "article_id": "article-2",
            "provider_item_id": "marketaux-2",
        }
    )

    assert (
        news_dataset_fingerprint(
            [first, second]
        )
        != news_dataset_fingerprint(
            [second, first]
        )
    )

def test_manifest_accepts_valid_metadata() -> None:
    manifest = NewsDatasetManifest(
        dataset_fingerprint="a" * 64,
        article_count=2,
        schema_version="1",
        created_at_utc=datetime(
            2026,
            10,
            30,
            13,
            0,
            tzinfo=timezone.utc,
        ),
        providers=["marketaux"],
        assets=["AAPL", "MSFT"],
    )

    assert manifest.dataset_fingerprint == "a" * 64
    assert manifest.article_count == 2
    assert manifest.schema_version == "1"
    assert manifest.providers == ["marketaux"]
    assert manifest.assets == ["AAPL", "MSFT"]


def test_manifest_forbids_extra_fields() -> None:
    with pytest.raises(ValueError):
        NewsDatasetManifest(
            dataset_fingerprint="a" * 64,
            article_count=1,
            schema_version="1",
            created_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
                tzinfo=timezone.utc,
            ),
            providers=["marketaux"],
            assets=["AAPL"],
            unexpected="bad",
        )


def test_manifest_requires_64_character_fingerprint() -> None:
    with pytest.raises(ValueError):
        NewsDatasetManifest(
            dataset_fingerprint="too-short",
            article_count=1,
            schema_version="1",
            created_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
                tzinfo=timezone.utc,
            ),
            providers=["marketaux"],
            assets=["AAPL"],
        )


def test_manifest_requires_positive_article_count() -> None:
    with pytest.raises(ValueError):
        NewsDatasetManifest(
            dataset_fingerprint="a" * 64,
            article_count=0,
            schema_version="1",
            created_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
                tzinfo=timezone.utc,
            ),
            providers=["marketaux"],
            assets=["AAPL"],
        )


def test_manifest_requires_utc_created_at() -> None:
    with pytest.raises(ValueError):
        NewsDatasetManifest(
            dataset_fingerprint="a" * 64,
            article_count=1,
            schema_version="1",
            created_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
            ),
            providers=["marketaux"],
            assets=["AAPL"],
        )


def test_manifest_rejects_nonzero_timezone_offset() -> None:
    from datetime import timedelta

    with pytest.raises(ValueError):
        NewsDatasetManifest(
            dataset_fingerprint="a" * 64,
            article_count=1,
            schema_version="1",
            created_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
                tzinfo=timezone(
                    timedelta(hours=1)
                ),
            ),
            providers=["marketaux"],
            assets=["AAPL"],
        )


def test_manifest_requires_nonempty_providers() -> None:
    with pytest.raises(ValueError):
        NewsDatasetManifest(
            dataset_fingerprint="a" * 64,
            article_count=1,
            schema_version="1",
            created_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
                tzinfo=timezone.utc,
            ),
            providers=[],
            assets=["AAPL"],
        )


def test_manifest_requires_nonempty_assets() -> None:
    with pytest.raises(ValueError):
        NewsDatasetManifest(
            dataset_fingerprint="a" * 64,
            article_count=1,
            schema_version="1",
            created_at_utc=datetime(
                2026,
                10,
                30,
                13,
                0,
                tzinfo=timezone.utc,
            ),
            providers=["marketaux"],
            assets=[],
        )

def test_build_manifest_derives_dataset_metadata() -> None:
    article = make_article()

    created_at = datetime(
        2026,
        10,
        30,
        14,
        0,
        tzinfo=timezone.utc,
    )

    manifest = build_news_manifest(
        articles=[article],
        created_at_utc=created_at,
    )

    assert manifest.dataset_fingerprint == (
        news_dataset_fingerprint([article])
    )
    assert manifest.article_count == 1
    assert manifest.schema_version == "1"
    assert manifest.created_at_utc == created_at
    assert manifest.providers == ["marketaux"]
    assert manifest.assets == ["AAPL"]


def test_build_manifest_collects_unique_providers() -> None:
    first = make_article()

    second = first.model_copy(
        update={
            "article_id": "article-2",
            "provider": "other-provider",
            "provider_item_id": "other-2",
        }
    )

    manifest = build_news_manifest(
        articles=[first, second],
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert manifest.providers == [
        "marketaux",
        "other-provider",
    ]


def test_build_manifest_collects_unique_assets() -> None:
    first = make_article()

    second = first.model_copy(
        update={
            "article_id": "article-2",
            "matched_assets": [
                "MSFT",
                "AAPL",
            ],
            "provider_item_id": "marketaux-2",
        }
    )

    manifest = build_news_manifest(
        articles=[first, second],
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert manifest.assets == [
        "AAPL",
        "MSFT",
    ]


def test_build_manifest_preserves_first_seen_order() -> None:
    first = make_article().model_copy(
        update={
            "matched_assets": ["MSFT", "AAPL"],
            "provider": "provider-b",
        }
    )

    second = make_article().model_copy(
        update={
            "article_id": "article-2",
            "matched_assets": ["AAPL", "BTC_USD"],
            "provider": "provider-a",
            "provider_item_id": "item-2",
        }
    )

    manifest = build_news_manifest(
        articles=[first, second],
        created_at_utc=datetime(
            2026,
            10,
            30,
            14,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert manifest.providers == [
        "provider-b",
        "provider-a",
    ]
    assert manifest.assets == [
        "MSFT",
        "AAPL",
        "BTC_USD",
    ]


def test_build_manifest_rejects_mixed_schema_versions() -> None:
    first = make_article()

    second = first.model_copy(
        update={
            "article_id": "article-2",
            "provider_item_id": "marketaux-2",
            "schema_version": "2",
        }
    )

    with pytest.raises(ValueError):
        build_news_manifest(
            articles=[first, second],
            created_at_utc=datetime(
                2026,
                10,
                30,
                14,
                0,
                tzinfo=timezone.utc,
            ),
        )


def test_build_manifest_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError):
        build_news_manifest(
            articles=[],
            created_at_utc=datetime(
                2026,
                10,
                30,
                14,
                0,
                tzinfo=timezone.utc,
            ),
        )


def test_build_manifest_rejects_non_utc_created_at() -> None:
    with pytest.raises(ValueError):
        build_news_manifest(
            articles=[make_article()],
            created_at_utc=datetime(
                2026,
                10,
                30,
                14,
                0,
            ),
        )