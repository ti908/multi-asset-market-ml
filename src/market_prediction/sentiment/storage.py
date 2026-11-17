"""Serialization primitives for persisted sentiment records."""

from __future__ import annotations

from datetime import datetime
import json

import os
from pathlib import Path
import uuid

from market_prediction.sentiment.fingerprint import (
    sentiment_dataset_fingerprint,
)
from market_prediction.sentiment.validation import (
    SentimentStatus,
    SentimentValidationReport,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from market_prediction.sentiment.models import (
    SentimentRecord,
)


class SentimentDatasetManifest(BaseModel):
    """Metadata describing one persisted sentiment dataset."""

    model_config = ConfigDict(
        extra="forbid"
    )

    dataset_fingerprint: str = Field(
        min_length=64,
        max_length=64,
    )
    sentiment_count: int = Field(
        ge=0
    )
    schema_version: str = Field(
        min_length=1
    )
    created_at_utc: datetime

    model_name: str = Field(
        min_length=1
    )
    model_revision: str = Field(
        min_length=1
    )
    scoring_method: str = Field(
        min_length=1
    )

    assets: list[str]

    @field_validator(
        "dataset_fingerprint"
    )
    @classmethod
    def validate_dataset_fingerprint(
        cls,
        value: str,
    ) -> str:
        """Require a hexadecimal SHA-256 digest."""

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
        """Require an aware timestamp normalized to UTC."""

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
        "model_name",
        "model_revision",
        "scoring_method",
        "schema_version",
    )
    @classmethod
    def validate_nonblank_string(
        cls,
        value: str,
    ) -> str:
        """Reject whitespace-only manifest metadata."""

        if not value.strip():
            raise ValueError(
                "manifest string values "
                "must not be blank"
            )

        return value

    @field_validator(
        "assets"
    )
    @classmethod
    def validate_assets(
        cls,
        value: list[str],
    ) -> list[str]:
        """Require nonblank asset names when assets are present."""

        for asset in value:
            if not isinstance(asset, str):
                raise ValueError(
                    "manifest assets must "
                    "be strings"
                )

            if not asset.strip():
                raise ValueError(
                    "manifest assets must "
                    "not be blank"
                )

        return value


def serialize_sentiment(
    sentiment: SentimentRecord,
) -> str:
    """Serialize one canonical sentiment record as one JSONL line."""

    if not isinstance(
        sentiment,
        SentimentRecord,
    ):
        raise TypeError(
            "sentiment must be a SentimentRecord"
        )

    payload = sentiment.model_dump(
        mode="json"
    )

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def deserialize_sentiment(
    serialized: str,
) -> SentimentRecord:
    """Deserialize and validate one persisted sentiment record."""

    if not isinstance(serialized, str):
        raise TypeError(
            "serialized sentiment must be a string"
        )

    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "serialized sentiment is not valid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "serialized sentiment must contain "
            "a JSON object"
        )

    try:
        return SentimentRecord.model_validate(
            payload
        )
    except ValueError as exc:
        raise ValueError(
            "serialized sentiment does not "
            "satisfy the SentimentRecord schema"
        ) from exc


def serialize_sentiments(
    sentiments: list[SentimentRecord],
) -> str:
    """Serialize a canonical sentiment dataset as deterministic JSONL.

    An empty sentiment dataset is valid and serializes to an empty
    string. No synthetic neutral observation is introduced.
    """

    if not isinstance(sentiments, list):
        raise TypeError(
            "sentiments must be a list"
        )

    lines: list[str] = []

    for sentiment in sentiments:
        if not isinstance(
            sentiment,
            SentimentRecord,
        ):
            raise TypeError(
                "sentiments must contain "
                "SentimentRecord instances"
            )

        lines.append(
            serialize_sentiment(sentiment)
        )

    return "\n".join(lines)

class SentimentValidationGateError(ValueError):
    """Raised when unvalidated sentiment data reaches persistence."""


class ValidatedSentimentDataset:
    """Canonical sentiment records that have passed validation."""

    __slots__ = (
        "_sentiments",
        "_report",
    )

    _CREATION_TOKEN = object()

    def __init__(
        self,
        sentiments: list[SentimentRecord],
        report: SentimentValidationReport,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not self._CREATION_TOKEN:
            raise SentimentValidationGateError(
                "ValidatedSentimentDataset cannot be "
                "constructed directly; use "
                "ValidatedSentimentDataset.from_validation()."
            )

        object.__setattr__(
            self,
            "_sentiments",
            list(sentiments),
        )
        object.__setattr__(
            self,
            "_report",
            report,
        )

    @classmethod
    def from_validation(
        cls,
        sentiments: list[SentimentRecord],
        report: SentimentValidationReport,
    ) -> "ValidatedSentimentDataset":
        """Admit only the exact dataset represented by a PASSED report."""

        if not isinstance(sentiments, list):
            raise SentimentValidationGateError(
                "sentiments must be a list"
            )

        if not isinstance(
            report,
            SentimentValidationReport,
        ):
            raise SentimentValidationGateError(
                "report must be a "
                "SentimentValidationReport"
            )

        for sentiment in sentiments:
            if not isinstance(
                sentiment,
                SentimentRecord,
            ):
                raise SentimentValidationGateError(
                    "sentiments must contain "
                    "SentimentRecord instances"
                )

        if report.status is not SentimentStatus.PASSED:
            raise SentimentValidationGateError(
                "Sentiment dataset is not "
                "persistence-eligible: "
                f"validation status is "
                f"{report.status.value}."
            )

        if report.sentiment_count != len(sentiments):
            raise SentimentValidationGateError(
                "Validation report sentiment count "
                f"({report.sentiment_count}) does not "
                "match dataset sentiment count "
                f"({len(sentiments)})."
            )

        if report.dataset_fingerprint is None:
            raise SentimentValidationGateError(
                "Validation report has no "
                "dataset fingerprint."
            )

        actual_fingerprint = (
            sentiment_dataset_fingerprint(
                sentiments
            )
        )

        if (
            report.dataset_fingerprint
            != actual_fingerprint
        ):
            raise SentimentValidationGateError(
                "Sentiment dataset does not match "
                "the content that produced its "
                "validation report."
            )

        return cls(
            sentiments,
            report,
            _token=cls._CREATION_TOKEN,
        )

    @property
    def sentiments(self) -> list[SentimentRecord]:
        """Return a defensive copy of the validated records."""

        return list(self._sentiments)

    @property
    def report(self) -> SentimentValidationReport:
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
                "ValidatedSentimentDataset unexpectedly "
                "has no dataset fingerprint."
            )

        return fingerprint

class StoredSentimentDataset:
    """One verified sentiment generation loaded from storage."""

    __slots__ = (
        "_sentiments",
        "_manifest",
    )

    def __init__(
        self,
        *,
        sentiments: list[SentimentRecord],
        manifest: SentimentDatasetManifest,
    ) -> None:
        self._sentiments = list(sentiments)
        self._manifest = manifest

    @property
    def sentiments(self) -> list[SentimentRecord]:
        """Return a defensive copy of stored sentiment records."""

        return list(self._sentiments)

    @property
    def manifest(self) -> SentimentDatasetManifest:
        """Return the verified persisted manifest."""

        return self._manifest

    @property
    def dataset_id(self) -> str:
        """Return the persisted dataset identity."""

        return self._manifest.dataset_fingerprint

def build_sentiment_manifest(
    *,
    sentiments: list[SentimentRecord],
    created_at_utc: datetime,
    empty_model_name: str | None = None,
    empty_model_revision: str | None = None,
    empty_scoring_method: str | None = None,
    empty_schema_version: str | None = None,
) -> SentimentDatasetManifest:
    """Build deterministic metadata for one sentiment dataset.

    Nonempty datasets derive scorer and schema metadata directly from
    their records.

    Empty datasets have no records from which provenance can be derived,
    so their expected scorer and schema metadata must be supplied
    explicitly.
    """

    if not isinstance(sentiments, list):
        raise TypeError(
            "sentiments must be a list"
        )

    for sentiment in sentiments:
        if not isinstance(
            sentiment,
            SentimentRecord,
        ):
            raise TypeError(
                "sentiments must contain "
                "SentimentRecord instances"
            )

    if (
        created_at_utc.tzinfo is None
        or created_at_utc.utcoffset() is None
    ):
        raise ValueError(
            "created_at_utc must be timezone-aware UTC"
        )

    if created_at_utc.utcoffset().total_seconds() != 0:
        raise ValueError(
            "created_at_utc must use UTC"
        )

    if sentiments:
        model_names = {
            sentiment.model_name
            for sentiment in sentiments
        }
        model_revisions = {
            sentiment.model_revision
            for sentiment in sentiments
        }
        scoring_methods = {
            sentiment.scoring_method
            for sentiment in sentiments
        }
        schema_versions = {
            sentiment.schema_version
            for sentiment in sentiments
        }

        if len(model_names) != 1:
            raise ValueError(
                "Sentiment dataset must use "
                "one model name."
            )

        if len(model_revisions) != 1:
            raise ValueError(
                "Sentiment dataset must use "
                "one model revision."
            )

        if len(scoring_methods) != 1:
            raise ValueError(
                "Sentiment dataset must use "
                "one scoring method."
            )

        if len(schema_versions) != 1:
            raise ValueError(
                "Sentiment dataset must use "
                "one schema version."
            )

        model_name = next(iter(model_names))
        model_revision = next(
            iter(model_revisions)
        )
        scoring_method = next(
            iter(scoring_methods)
        )
        schema_version = next(
            iter(schema_versions)
        )

    else:
        empty_metadata = {
            "empty_model_name": empty_model_name,
            "empty_model_revision": empty_model_revision,
            "empty_scoring_method": empty_scoring_method,
            "empty_schema_version": empty_schema_version,
        }

        for field_name, value in empty_metadata.items():
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be supplied "
                    "for an empty sentiment dataset"
                )

        model_name = empty_model_name
        model_revision = empty_model_revision
        scoring_method = empty_scoring_method
        schema_version = empty_schema_version

        assert model_name is not None
        assert model_revision is not None
        assert scoring_method is not None
        assert schema_version is not None

    assets: list[str] = []

    for sentiment in sentiments:
        if sentiment.asset not in assets:
            assets.append(
                sentiment.asset
            )

    return SentimentDatasetManifest(
        dataset_fingerprint=(
            sentiment_dataset_fingerprint(
                sentiments
            )
        ),
        sentiment_count=len(sentiments),
        schema_version=schema_version,
        created_at_utc=created_at_utc,
        model_name=model_name,
        model_revision=model_revision,
        scoring_method=scoring_method,
        assets=assets,
    )
class SentimentStore:
    """Atomic persistence for validated canonical sentiment datasets."""

    def __init__(
        self,
        *,
        storage_dir: Path,
        model_name: str,
        model_revision: str,
        scoring_method: str,
        schema_version: str,
    ) -> None:
        if not isinstance(storage_dir, Path):
            raise TypeError(
                "storage_dir must be a Path"
            )

        metadata = {
            "model_name": model_name,
            "model_revision": model_revision,
            "scoring_method": scoring_method,
            "schema_version": schema_version,
        }

        for field_name, value in metadata.items():
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be "
                    "a nonblank string"
                )

        self._storage_dir = storage_dir
        self._model_name = model_name
        self._model_revision = model_revision
        self._scoring_method = scoring_method
        self._schema_version = schema_version

    def write(
        self,
        dataset: ValidatedSentimentDataset,
        *,
        created_at_utc: datetime,
    ) -> SentimentDatasetManifest:
        """Atomically publish one validated sentiment generation."""

        if not isinstance(
            dataset,
            ValidatedSentimentDataset,
        ):
            raise SentimentValidationGateError(
                "SentimentStore accepts only "
                "ValidatedSentimentDataset instances."
            )

        sentiments = dataset.sentiments

        actual_fingerprint = (
            sentiment_dataset_fingerprint(
                sentiments
            )
        )

        if (
            actual_fingerprint
            != dataset.dataset_id
        ):
            raise SentimentValidationGateError(
                "Validated sentiment content no longer "
                "matches its validation fingerprint."
            )

        if sentiments:
            first = sentiments[0]

            if (
                first.model_name
                != self._model_name
            ):
                raise SentimentValidationGateError(
                    "Validated sentiment model name "
                    "does not match the store contract."
                )

            if (
                first.model_revision
                != self._model_revision
            ):
                raise SentimentValidationGateError(
                    "Validated sentiment model revision "
                    "does not match the store contract."
                )

            if (
                first.scoring_method
                != self._scoring_method
            ):
                raise SentimentValidationGateError(
                    "Validated sentiment scoring method "
                    "does not match the store contract."
                )

            if (
                first.schema_version
                != self._schema_version
            ):
                raise SentimentValidationGateError(
                    "Validated sentiment schema version "
                    "does not match the store contract."
                )

        manifest = build_sentiment_manifest(
            sentiments=sentiments,
            created_at_utc=created_at_utc,
            empty_model_name=self._model_name,
            empty_model_revision=(
                self._model_revision
            ),
            empty_scoring_method=(
                self._scoring_method
            ),
            empty_schema_version=(
                self._schema_version
            ),
        )

        if (
            manifest.dataset_fingerprint
            != dataset.dataset_id
        ):
            raise SentimentValidationGateError(
                "Sentiment manifest fingerprint "
                "does not match the validated dataset."
            )

        if (
            manifest.model_name
            != self._model_name
        ):
            raise SentimentValidationGateError(
                "Sentiment manifest model name "
                "does not match the store contract."
            )

        if (
            manifest.model_revision
            != self._model_revision
        ):
            raise SentimentValidationGateError(
                "Sentiment manifest model revision "
                "does not match the store contract."
            )

        if (
            manifest.scoring_method
            != self._scoring_method
        ):
            raise SentimentValidationGateError(
                "Sentiment manifest scoring method "
                "does not match the store contract."
            )

        if (
            manifest.schema_version
            != self._schema_version
        ):
            raise SentimentValidationGateError(
                "Sentiment manifest schema version "
                "does not match the store contract."
            )

        self._publish(
            sentiments=sentiments,
            manifest=manifest,
        )

        return manifest

    def read(
        self,
    ) -> StoredSentimentDataset:
        """Read and independently verify the current generation."""

        pointer_path = (
            self._storage_dir
            / "current.json"
        )

        pointer_payload = json.loads(
            pointer_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            pointer_payload,
            dict,
        ):
            raise ValueError(
                "Current sentiment pointer "
                "must contain a JSON object."
            )

        if set(pointer_payload) != {
            "generation",
        }:
            raise ValueError(
                "Current sentiment pointer must "
                "contain exactly the generation field."
            )

        generation = pointer_payload[
            "generation"
        ]

        if (
            not isinstance(generation, str)
            or not generation
        ):
            raise ValueError(
                "Current sentiment generation "
                "must be a nonblank string."
            )

        generation_dir = (
            self._storage_dir
            / "generations"
            / generation
        )

        if not generation_dir.is_dir():
            raise FileNotFoundError(
                "Current sentiment generation "
                "directory does not exist."
            )

        sentiments_path = (
            generation_dir
            / "sentiments.jsonl"
        )

        manifest_path = (
            generation_dir
            / "manifest.json"
        )

        if not sentiments_path.is_file():
            raise FileNotFoundError(
                "Current sentiment generation "
                "is missing sentiments.jsonl."
            )

        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Current sentiment generation "
                "is missing manifest.json."
            )

        serialized_sentiments = (
            sentiments_path.read_text(
                encoding="utf-8"
            )
        )

        sentiments: list[
            SentimentRecord
        ] = []

        if serialized_sentiments:
            for line in (
                serialized_sentiments.splitlines()
            ):
                sentiments.append(
                    deserialize_sentiment(
                        line
                    )
                )

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
                "Stored sentiment manifest "
                "must contain a JSON object."
            )

        manifest = (
            SentimentDatasetManifest.model_validate(
                manifest_payload
            )
        )

        actual_fingerprint = (
            sentiment_dataset_fingerprint(
                sentiments
            )
        )

        if (
            actual_fingerprint
            != manifest.dataset_fingerprint
        ):
            raise ValueError(
                "Stored sentiment fingerprint does not "
                "match its manifest."
            )

        if (
            len(sentiments)
            != manifest.sentiment_count
        ):
            raise ValueError(
                "Stored sentiment count does not "
                "match its manifest."
            )

        if (
            manifest.schema_version
            != self._schema_version
        ):
            raise ValueError(
                "Stored sentiment schema version does "
                "not match the store contract."
            )

        if (
            manifest.model_name
            != self._model_name
        ):
            raise ValueError(
                "Stored sentiment model name does "
                "not match the store contract."
            )

        if (
            manifest.model_revision
            != self._model_revision
        ):
            raise ValueError(
                "Stored sentiment model revision does "
                "not match the store contract."
            )

        if (
            manifest.scoring_method
            != self._scoring_method
        ):
            raise ValueError(
                "Stored sentiment scoring method does "
                "not match the store contract."
            )

        actual_assets: list[str] = []
        seen_assets: set[str] = set()

        for sentiment in sentiments:
            if (
                sentiment.asset
                not in seen_assets
            ):
                seen_assets.add(
                    sentiment.asset
                )
                actual_assets.append(
                    sentiment.asset
                )

            if (
                sentiment.schema_version
                != self._schema_version
            ):
                raise ValueError(
                    "Stored sentiment record schema "
                    "version does not match the "
                    "store contract."
                )

            if (
                sentiment.model_name
                != self._model_name
            ):
                raise ValueError(
                    "Stored sentiment record model "
                    "name does not match the "
                    "store contract."
                )

            if (
                sentiment.model_revision
                != self._model_revision
            ):
                raise ValueError(
                    "Stored sentiment record model "
                    "revision does not match the "
                    "store contract."
                )

            if (
                sentiment.scoring_method
                != self._scoring_method
            ):
                raise ValueError(
                    "Stored sentiment record scoring "
                    "method does not match the "
                    "store contract."
                )

        if (
            manifest.assets
            != actual_assets
        ):
            raise ValueError(
                "Stored sentiment assets do not "
                "match its manifest."
            )

        return StoredSentimentDataset(
            sentiments=sentiments,
            manifest=manifest,
        )

    def _publish(
        self,
        *,
        sentiments: list[SentimentRecord],
        manifest: SentimentDatasetManifest,
    ) -> None:
        """Publish an immutable generation then atomically expose it."""

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

        sentiments_path = (
            generation_dir
            / "sentiments.jsonl"
        )

        manifest_path = (
            generation_dir
            / "manifest.json"
        )

        sentiments_path.write_text(
            serialize_sentiments(
                sentiments
            ),
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
