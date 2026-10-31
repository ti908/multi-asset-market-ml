"""Canonical news article contracts for Step 4.

This module defines provider-neutral representations of market-relevant
news records.

It contains no provider fetching, storage, sentiment scoring, market-data
alignment, feature engineering, or modeling logic.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class PublishedAtPrecision(str, Enum):
    """Precision supplied for an article publication timestamp."""

    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DATE = "date"


class TimestampSource(str, Enum):
    """Provenance of the normalized publication timestamp."""

    PUBLISHER_PUBLICATION = "publisher_publication"
    PROVIDER_PUBLICATION = "provider_publication"
    PROVIDER_SEEN = "provider_seen"
    DATE_ONLY = "date_only"
    UNKNOWN = "unknown"


class TextLicenseClass(str, Enum):
    """Classification of text retained from an upstream source."""

    PROVIDER_METADATA = "provider_metadata"
    LICENSED_TEXT = "licensed_text"
    PUBLIC_DOMAIN = "public_domain"
    UNKNOWN = "unknown"


class ArticleRecord(BaseModel):
    """Canonical provider-neutral news article.

    The record deliberately distinguishes publication, provider-seen,
    and retrieval timestamps so downstream code cannot silently treat
    crawl or retrieval time as publication time.

    All normalized timestamps must be timezone-aware UTC values.
    """

    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(min_length=1)

    matched_assets: list[str] = Field(
        min_length=1,
    )

    query_id: str = Field(min_length=1)

    title: str = Field(min_length=1)

    snippet: str | None = None
    full_text: str | None = None

    canonical_url: str = Field(min_length=1)
    publisher: str = Field(min_length=1)

    published_at_utc: datetime
    published_at_precision: PublishedAtPrecision

    original_timezone: str | None = None

    provider_seen_at_utc: datetime
    retrieved_at_utc: datetime

    provider: str = Field(min_length=1)
    provider_item_id: str = Field(min_length=1)

    language: str = Field(min_length=1)

    content_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    duplicate_group_id: str | None = None
    raw_payload_reference: str | None = None

    schema_version: str = Field(min_length=1)

    timestamp_source: TimestampSource
    text_license_class: TextLicenseClass

    @field_validator("matched_assets")
    @classmethod
    def validate_matched_assets(
        cls,
        value: list[str],
    ) -> list[str]:
        """Require unique, non-empty asset identifiers."""

        if any(
            not isinstance(asset, str)
            or not asset.strip()
            for asset in value
        ):
            raise ValueError(
                "matched_assets must contain "
                "non-empty asset identifiers"
            )

        if len(value) != len(set(value)):
            raise ValueError(
                "matched_assets must be unique"
            )

        return value

    @field_validator(
        "published_at_utc",
        "provider_seen_at_utc",
        "retrieved_at_utc",
    )
    @classmethod
    def validate_utc_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        """Reject naive and non-UTC normalized timestamps."""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "normalized timestamps must "
                "be timezone-aware"
            )

        if value.utcoffset().total_seconds() != 0:
            raise ValueError(
                "normalized timestamps must be UTC"
            )

        return value

    @model_validator(mode="after")
    def validate_timestamp_order(
        self,
    ) -> "ArticleRecord":
        """Reject impossible information-time ordering.

        A record cannot be observed by the provider before its
        publication timestamp, and this project cannot retrieve a
        record before the provider has observed it.
        """

        if (
            self.provider_seen_at_utc
            < self.published_at_utc
        ):
            raise ValueError(
                "provider_seen_at_utc cannot precede "
                "published_at_utc"
            )

        if (
            self.retrieved_at_utc
            < self.provider_seen_at_utc
        ):
            raise ValueError(
                "retrieved_at_utc cannot precede "
                "provider_seen_at_utc"
            )

        return self