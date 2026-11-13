"""Serialization primitives for persisted news records."""

from __future__ import annotations

import json

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from market_prediction.news.fingerprint import (
    news_dataset_fingerprint,
)

from market_prediction.news.models import ArticleRecord

from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from market_prediction.news.validation import (
    NewsStatus,
    NewsValidationReport,
)
import os
import uuid
from pathlib import Path
import pytest

class NewsDatasetManifest(BaseModel):
    """Metadata describing one persisted news dataset."""

    model_config = ConfigDict(
        extra="forbid"
    )

    dataset_fingerprint: str = Field(
        min_length=64,
        max_length=64,
    )
    article_count: int = Field(
        gt=0
    )
    schema_version: str = Field(
        min_length=1
    )
    created_at_utc: datetime
    providers: list[str] = Field(
        min_length=1
    )
    assets: list[str] = Field(
        min_length=1
    )

    @field_validator(
        "dataset_fingerprint"
    )
    @classmethod
    def validate_dataset_fingerprint(
        cls,
        value: str,
    ) -> str:
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(
                "dataset_fingerprint must be "
                "a hexadecimal SHA-256 digest"
            ) from exc

        return value

    @field_validator(
        "created_at_utc"
    )
    @classmethod
    def validate_created_at_utc(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "created_at_utc must be "
                "timezone-aware UTC"
            )

        if value.utcoffset().total_seconds() != 0:
            raise ValueError(
                "created_at_utc must use UTC"
            )

        return value

    @field_validator(
        "providers",
        "assets",
    )
    @classmethod
    def validate_nonempty_strings(
        cls,
        value: list[str],
    ) -> list[str]:
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    "manifest list values "
                    "must be strings"
                )

            if not item.strip():
                raise ValueError(
                    "manifest list values "
                    "must not be blank"
                )

        return value

class NewsValidationGateError(ValueError):
    """Raised when unvalidated news data reaches persistence."""


class ValidatedNewsDataset:
    """Canonical news records that have passed validation."""

    __slots__ = (
        "_articles",
        "_report",
    )

    _CREATION_TOKEN = object()

    def __init__(
        self,
        articles: list[ArticleRecord],
        report: NewsValidationReport,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not self._CREATION_TOKEN:
            raise NewsValidationGateError(
                "ValidatedNewsDataset cannot be "
                "constructed directly; use "
                "ValidatedNewsDataset.from_validation()."
            )

        object.__setattr__(
            self,
            "_articles",
            list(articles),
        )
        object.__setattr__(
            self,
            "_report",
            report,
        )

    @classmethod
    def from_validation(
        cls,
        articles: list[ArticleRecord],
        report: NewsValidationReport,
    ) -> "ValidatedNewsDataset":
        """Admit only the exact dataset represented by a PASSED report."""
        if not isinstance(articles, list):
            raise NewsValidationGateError(
                "articles must be a list"
            )

        if not isinstance(
            report,
            NewsValidationReport,
        ):
            raise NewsValidationGateError(
                "report must be a "
                "NewsValidationReport"
            )

        for article in articles:
            if not isinstance(
                article,
                ArticleRecord,
            ):
                raise NewsValidationGateError(
                    "articles must contain "
                    "ArticleRecord instances"
                )

        if report.status is not NewsStatus.PASSED:
            raise NewsValidationGateError(
                "News dataset is not "
                "persistence-eligible: "
                f"validation status is "
                f"{report.status.value}."
            )

        if report.article_count != len(articles):
            raise NewsValidationGateError(
                "Validation report article count "
                f"({report.article_count}) does not "
                "match dataset article count "
                f"({len(articles)})."
            )

        if report.dataset_fingerprint is None:
            raise NewsValidationGateError(
                "Validation report has no "
                "dataset fingerprint."
            )

        if not articles:
            raise NewsValidationGateError(
                "Validated news dataset "
                "cannot be empty."
            )

        actual_fingerprint = (
            news_dataset_fingerprint(
                articles
            )
        )

        if (
            report.dataset_fingerprint
            != actual_fingerprint
        ):
            raise NewsValidationGateError(
                "News dataset does not match "
                "the content that produced its "
                "validation report."
            )

        return cls(
            articles,
            report,
            _token=cls._CREATION_TOKEN,
        )

    @property
    def articles(self) -> list[ArticleRecord]:
        """Return a defensive copy of the validated records."""
        return list(self._articles)

    @property
    def report(self) -> NewsValidationReport:
        """Return the report that admitted this dataset."""
        return self._report

    @property
    def dataset_id(self) -> str:
        """Return the validated content identity."""
        fingerprint = (
            self._report.dataset_fingerprint
        )

        if fingerprint is None:
            raise RuntimeError(
                "ValidatedNewsDataset unexpectedly "
                "has no dataset fingerprint."
            )

        return fingerprint

class StoredNewsDataset:
    """One integrity-checked news dataset loaded from storage."""

    __slots__ = (
        "_articles",
        "_manifest",
    )

    def __init__(
        self,
        *,
        articles: list[ArticleRecord],
        manifest: NewsDatasetManifest,
    ) -> None:
        self._articles = list(articles)
        self._manifest = manifest

    @property
    def articles(self) -> list[ArticleRecord]:
        """Return a defensive copy of the persisted records."""
        return list(self._articles)

    @property
    def manifest(self) -> NewsDatasetManifest:
        """Return the persisted dataset manifest."""
        return self._manifest

    @property
    def dataset_id(self) -> str:
        """Return the persisted dataset fingerprint."""
        return self._manifest.dataset_fingerprint

class NewsStore:
    """Atomically publish validated canonical news datasets."""

    def __init__(
        self,
        *,
        storage_dir: Path,
    ) -> None:
        self._storage_dir = Path(
            storage_dir
        )

    def write(
        self,
        dataset: ValidatedNewsDataset,
        *,
        created_at_utc: datetime,
    ) -> NewsDatasetManifest:
        """Publish one complete validated news generation."""
        if not isinstance(
            dataset,
            ValidatedNewsDataset,
        ):
            raise NewsValidationGateError(
                "NewsStore.write() accepts only "
                "ValidatedNewsDataset."
            )

        articles = dataset.articles

        actual_fingerprint = (
            news_dataset_fingerprint(
                articles
            )
        )

        if (
            actual_fingerprint
            != dataset.dataset_id
        ):
            raise NewsValidationGateError(
                "Validated news dataset content "
                "no longer matches its validation "
                "fingerprint."
            )

        manifest = build_news_manifest(
            articles=articles,
            created_at_utc=created_at_utc,
        )

        if (
            manifest.dataset_fingerprint
            != dataset.dataset_id
        ):
            raise NewsValidationGateError(
                "News manifest fingerprint does "
                "not match the validated dataset."
            )

        self._publish(
            articles=articles,
            manifest=manifest,
        )

        return manifest

    def read(
        self,
    ) -> StoredNewsDataset:
        """Read and integrity-check the currently published generation."""
        pointer_path = (
            self._storage_dir
            / "current.json"
        )

        pointer = json.loads(
            pointer_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(pointer, dict):
            raise ValueError(
                "News store pointer must be "
                "a JSON object."
            )

        if set(pointer) != {
            "generation",
        }:
            raise ValueError(
                "News store pointer must contain "
                "exactly the generation field."
            )

        generation = pointer[
            "generation"
        ]

        if (
            not isinstance(generation, str)
            or not generation
        ):
            raise ValueError(
                "News store generation must "
                "be a non-empty string."
            )

        generation_dir = (
            self._storage_dir
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

        serialized_articles = (
            articles_path.read_text(
                encoding="utf-8"
            )
        )

        lines = (
            serialized_articles.splitlines()
        )

        if not lines:
            raise ValueError(
                "Stored news dataset must "
                "not be empty."
            )

        articles = [
            deserialize_article(line)
            for line in lines
        ]

        manifest_payload = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            manifest_payload,
            dict,
        ):
            raise ValueError(
                "News manifest must contain "
                "a JSON object."
            )

        manifest = (
            NewsDatasetManifest.model_validate(
                manifest_payload
            )
        )

        actual_fingerprint = (
            news_dataset_fingerprint(
                articles
            )
        )

        if (
                manifest.dataset_fingerprint
                != actual_fingerprint
        ):
            raise ValueError(
                "Stored news dataset fingerprint "
                "does not match its manifest."
            )

        if (
                manifest.article_count
                != len(articles)
        ):
            raise ValueError(
                "Stored news article count does "
                "not match its manifest."
            )

        schema_versions = {
            article.schema_version
            for article in articles
        }

        if len(schema_versions) != 1:
            raise ValueError(
                "Stored news articles do not use "
                "one schema version."
            )

        actual_schema_version = next(
            iter(schema_versions)
        )

        if (
                manifest.schema_version
                != actual_schema_version
        ):
            raise ValueError(
                "Stored news schema version does "
                "not match its manifest."
            )

        actual_providers: list[str] = []
        seen_providers: set[str] = set()

        actual_assets: list[str] = []
        seen_assets: set[str] = set()

        for article in articles:
            if (
                    article.provider
                    not in seen_providers
            ):
                seen_providers.add(
                    article.provider
                )
                actual_providers.append(
                    article.provider
                )

            for asset in article.matched_assets:
                if asset not in seen_assets:
                    seen_assets.add(asset)
                    actual_assets.append(asset)

        if (
                manifest.providers
                != actual_providers
        ):
            raise ValueError(
                "Stored news providers do not "
                "match its manifest."
            )

        if (
                manifest.assets
                != actual_assets
        ):
            raise ValueError(
                "Stored news assets do not "
                "match its manifest."
            )

        return StoredNewsDataset(
            articles=articles,
            manifest=manifest,
        )

    def _publish(
        self,
        *,
        articles: list[ArticleRecord],
        manifest: NewsDatasetManifest,
    ) -> None:
        """Publish one immutable generation and atomically expose it."""
        generations_dir = (
            self._storage_dir
            / "generations"
        )

        generations_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        generation = uuid.uuid4().hex

        generation_dir = (
            generations_dir
            / generation
        )

        generation_dir.mkdir(
            parents=False,
            exist_ok=False,
        )

        articles_path = (
            generation_dir
            / "articles.jsonl"
        )
        manifest_path = (
            generation_dir
            / "manifest.json"
        )

        articles_path.write_text(
            serialize_articles(articles),
            encoding="utf-8",
        )

        manifest_path.write_text(
            json.dumps(
                manifest.model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

        pointer = json.dumps(
            {
                "generation": generation,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        temporary_pointer = (
            self._storage_dir
            / (
                f".current-{uuid.uuid4().hex}"
                ".tmp"
            )
        )

        temporary_pointer.write_text(
            pointer,
            encoding="utf-8",
        )

        os.replace(
            temporary_pointer,
            self._storage_dir
            / "current.json",
        )

def serialize_article(
    article: ArticleRecord,
) -> str:
    """Serialize one ArticleRecord as a deterministic JSONL line."""
    if not isinstance(article, ArticleRecord):
        raise TypeError(
            "article must be an ArticleRecord"
        )

    return json.dumps(
        article.model_dump(
            mode="json"
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def deserialize_article(
    serialized: str,
) -> ArticleRecord:
    """Deserialize one JSONL line into a strict ArticleRecord."""
    if not isinstance(serialized, str):
        raise TypeError(
            "serialized must be a string"
        )

    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "serialized article is not valid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "serialized article must contain a JSON object"
        )

    try:
        return ArticleRecord.model_validate(
            payload
        )
    except ValidationError as exc:
        raise ValueError(
            "serialized article does not satisfy "
            "the ArticleRecord schema"
        ) from exc

def serialize_articles(
    articles: list[ArticleRecord],
) -> str:
    """Serialize an ordered ArticleRecord dataset as canonical JSONL."""
    if not isinstance(articles, list):
        raise TypeError(
            "articles must be a list"
        )

    if not articles:
        raise ValueError(
            "articles must not be empty"
        )

    lines: list[str] = []

    for article in articles:
        if not isinstance(article, ArticleRecord):
            raise TypeError(
                "articles must contain "
                "ArticleRecord instances"
            )

        lines.append(
            serialize_article(article)
        )

    return "\n".join(lines)

def build_news_manifest(
    *,
    articles: list[ArticleRecord],
    created_at_utc: datetime,
) -> NewsDatasetManifest:
    """Build deterministic metadata for an ordered news dataset."""
    if not isinstance(articles, list):
        raise TypeError(
            "articles must be a list"
        )

    if not articles:
        raise ValueError(
            "articles must not be empty"
        )

    for article in articles:
        if not isinstance(article, ArticleRecord):
            raise TypeError(
                "articles must contain "
                "ArticleRecord instances"
            )

    schema_versions = {
        article.schema_version
        for article in articles
    }

    if len(schema_versions) != 1:
        raise ValueError(
            "articles must use one "
            "schema_version"
        )

    providers: list[str] = []
    seen_providers: set[str] = set()

    assets: list[str] = []
    seen_assets: set[str] = set()

    for article in articles:
        if article.provider not in seen_providers:
            seen_providers.add(
                article.provider
            )
            providers.append(
                article.provider
            )

        for asset in article.matched_assets:
            if asset not in seen_assets:
                seen_assets.add(asset)
                assets.append(asset)

    return NewsDatasetManifest(
        dataset_fingerprint=(
            news_dataset_fingerprint(
                articles
            )
        ),
        article_count=len(articles),
        schema_version=articles[0].schema_version,
        created_at_utc=created_at_utc,
        providers=providers,
        assets=assets,
    )
