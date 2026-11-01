from __future__ import annotations

from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from market_prediction.news.models import (
    PublishedAtPrecision,
    TimestampSource,
)


class NewsCandidate(BaseModel):
    """Provider-neutral news discovery candidate.

    A candidate represents a discovered news resource that may not yet
    contain enough licensed or provider-supplied content to construct
    the canonical ArticleRecord.

    In particular, no article title, snippet, or full text is invented
    merely to satisfy the downstream article contract.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    candidate_id: str = Field(
        min_length=1,
    )
    asset: str = Field(
        min_length=1,
    )
    query_id: str = Field(
        min_length=1,
    )

    canonical_url: str = Field(
        min_length=1,
    )
    publisher: str = Field(
        min_length=1,
    )

    published_at_utc: datetime
    published_at_precision: PublishedAtPrecision
    provider_seen_at_utc: datetime
    retrieved_at_utc: datetime

    provider: str = Field(
        min_length=1,
    )
    provider_item_id: str = Field(
        min_length=1,
    )

    timestamp_source: TimestampSource

    @field_validator(
        "candidate_id",
        "asset",
        "query_id",
        "canonical_url",
        "publisher",
        "provider",
        "provider_item_id",
    )
    @classmethod
    def validate_required_text(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError(
                "required text cannot be blank"
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
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "timestamp must be timezone-aware"
            )

        if value.utcoffset().total_seconds() != 0:
            raise ValueError(
                "timestamp must be UTC"
            )

        return value

    @model_validator(
        mode="after",
    )
    def validate_timestamp_order(
        self,
    ) -> "NewsCandidate":
        if (
            self.published_at_utc
            > self.provider_seen_at_utc
        ):
            raise ValueError(
                "published_at_utc cannot be after "
                "provider_seen_at_utc"
            )

        if (
            self.provider_seen_at_utc
            > self.retrieved_at_utc
        ):
            raise ValueError(
                "provider_seen_at_utc cannot be after "
                "retrieved_at_utc"
            )

        return self