from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from market_prediction.news.fingerprint import (
    article_identity,
    content_fingerprint,
)
from market_prediction.news.models import (
    ArticleRecord,
    PublishedAtPrecision,
    TextLicenseClass,
    TimestampSource,
)

from market_prediction.providers.news.exceptions import (
    InvalidNewsProviderDataError,
    NewsProviderUnavailableError,
    NewsQuotaExceededError,
)
from urllib.parse import urlsplit
from market_prediction.providers.news.base import (
    NewsProvider,
    NewsQuery,
    validate_news_query_window,
)
from typing import Protocol
from dataclasses import dataclass
from collections.abc import Mapping
from market_prediction.news.dedup import (
    assign_duplicate_groups,
)

MARKETAUX_PROVIDER = "marketaux"
MARKETAUX_SCHEMA_VERSION = "1"

@dataclass(frozen=True)
class MarketauxResponse:
    found: int
    returned: int
    limit: int
    page: int
    items: tuple[Mapping[str, object], ...]

class MarketauxTransport(Protocol):
    """Minimal injectable HTTP transport used by MarketauxNewsProvider."""

    def get_json(
        self,
        *,
        url: str,
        params: dict[str, str],
        timeout_seconds: float,
    ) -> tuple[int, object]:
        ...

class MarketauxClock(Protocol):
    """Clock used to record when a Marketaux response was observed."""

    def now_utc(self) -> datetime:
        ...

def _required_string(
    payload: Mapping[str, object],
    field: str,
) -> str:
    """Read a required non-blank string from a Marketaux payload."""

    if field not in payload:
        raise InvalidNewsProviderDataError(
            f"Marketaux payload is missing required field: {field}"
        )

    value = payload[field]

    if not isinstance(value, str):
        raise InvalidNewsProviderDataError(
            f"Marketaux field {field} must be a string"
        )

    if not value.strip():
        raise InvalidNewsProviderDataError(
            f"Marketaux field {field} cannot be blank"
        )

    return value


def _optional_string(
    payload: Mapping[str, object],
    field: str,
) -> str | None:
    """Read an optional nullable string from a Marketaux payload."""

    value = payload.get(field)

    if value is None:
        return None

    if not isinstance(value, str):
        raise InvalidNewsProviderDataError(
            f"Marketaux field {field} must be a string or null"
        )

    return value


def _parse_marketaux_timestamp(
    value: str,
) -> datetime:
    """Parse a Marketaux publication timestamp without timezone repair."""

    if value.endswith("Z"):
        parse_value = (
            value[:-1] + "+00:00"
        )
    else:
        parse_value = value

    try:
        parsed = datetime.fromisoformat(
            parse_value
        )
    except ValueError as exc:
        raise InvalidNewsProviderDataError(
            "Marketaux field published_at "
            "must be a valid ISO-8601 timestamp"
        ) from exc

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise InvalidNewsProviderDataError(
            "Marketaux field published_at "
            "must be timezone-aware UTC"
        )

    if (
        parsed.utcoffset().total_seconds()
        != 0
    ):
        raise InvalidNewsProviderDataError(
            "Marketaux field published_at "
            "must explicitly represent UTC"
        )

    return parsed


def _validate_retrieved_at(
    value: datetime,
) -> datetime:
    """Require an explicitly UTC retrieval timestamp."""

    if not isinstance(value, datetime):
        raise InvalidNewsProviderDataError(
            "retrieved_at_utc must be a datetime"
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise InvalidNewsProviderDataError(
            "retrieved_at_utc must be timezone-aware UTC"
        )

    if (
        value.utcoffset().total_seconds()
        != 0
    ):
        raise InvalidNewsProviderDataError(
            "retrieved_at_utc must explicitly represent UTC"
        )

    return value


def normalize_marketaux_article(
    *,
    payload: Mapping[str, object],
    query: NewsQuery,
    retrieved_at_utc: datetime,
) -> ArticleRecord:
    """Normalize one Marketaux news item into ArticleRecord.

    This function performs deterministic provider-data normalization
    only. It performs no HTTP requests, sentiment scoring, article-body
    retrieval, or duplicate collapsing.
    """

    if not isinstance(
        payload,
        Mapping,
    ):
        raise InvalidNewsProviderDataError(
            "Marketaux payload must be a mapping"
        )

    if not isinstance(
        query,
        NewsQuery,
    ):
        raise InvalidNewsProviderDataError(
            "Marketaux query must be a NewsQuery"
        )

    retrieved = _validate_retrieved_at(
        retrieved_at_utc
    )

    provider_item_id = _required_string(
        payload,
        "uuid",
    )
    title = _required_string(
        payload,
        "title",
    )
    canonical_url = _required_string(
        payload,
        "url",
    )
    publisher = _required_string(
        payload,
        "source",
    )
    language = _required_string(
        payload,
        "language",
    )
    published_at_text = _required_string(
        payload,
        "published_at",
    )

    snippet = _optional_string(
        payload,
        "description",
    )

    published_at = (
        _parse_marketaux_timestamp(
            published_at_text
        )
    )

    if published_at > retrieved:
        raise InvalidNewsProviderDataError(
            "Marketaux published_at cannot be after retrieved_at_utc"
        )

    return ArticleRecord(
        article_id=article_identity(
            provider=MARKETAUX_PROVIDER,
            provider_item_id=provider_item_id,
        ),
        matched_assets=[
            query.asset
        ],
        query_id=query.query_id,
        title=title,
        snippet=snippet,
        full_text=None,
        canonical_url=canonical_url,
        publisher=publisher,
        published_at_utc=published_at,
        published_at_precision=(
            PublishedAtPrecision.SECOND
        ),
        original_timezone="UTC",
        provider_seen_at_utc=retrieved,
        retrieved_at_utc=retrieved,
        provider=MARKETAUX_PROVIDER,
        provider_item_id=provider_item_id,
        language=language,
        content_hash=content_fingerprint(
            title=title,
            snippet=snippet,
            full_text=None,
        ),
        duplicate_group_id=None,
        raw_payload_reference=None,
        schema_version=(
            MARKETAUX_SCHEMA_VERSION
        ),
        timestamp_source=(
            TimestampSource.PROVIDER_PUBLICATION
        ),
        text_license_class=(
            TextLicenseClass.PROVIDER_METADATA
        ),
    )

def parse_marketaux_response(
    *,
    status_code: int,
    payload: object,
) -> MarketauxResponse:
    if (
        not isinstance(status_code, int)
        or isinstance(status_code, bool)
        or not 100 <= status_code <= 599
    ):
        raise InvalidNewsProviderDataError(
            "status_code must be an integer "
            "between 100 and 599"
        )

    if not isinstance(payload, Mapping):
        raise InvalidNewsProviderDataError(
            "Marketaux response payload "
            "must be a mapping"
        )

    if status_code == 429:
        raise NewsQuotaExceededError(
            "Marketaux quota exceeded"
        )

    if status_code >= 500:
        raise NewsProviderUnavailableError(
            "Marketaux provider unavailable"
        )

    if status_code >= 400:
        raise InvalidNewsProviderDataError(
            "Marketaux returned invalid "
            f"HTTP status {status_code}"
        )

    if not 200 <= status_code <= 299:
        raise InvalidNewsProviderDataError(
            "Marketaux returned unexpected "
            f"HTTP status {status_code}"
        )

    if "data" not in payload:
        raise InvalidNewsProviderDataError(
            "Marketaux response missing data"
        )

    data = payload["data"]

    if not isinstance(data, list):
        raise InvalidNewsProviderDataError(
            "Marketaux response data must be a list"
        )

    items: list[Mapping[str, object]] = []

    for item in data:
        if not isinstance(item, Mapping):
            raise InvalidNewsProviderDataError(
                "Marketaux response data items "
                "must be mappings"
            )

        items.append(item)

    if "meta" not in payload:
        raise InvalidNewsProviderDataError(
            "Marketaux response missing meta"
        )

    meta = payload["meta"]

    if not isinstance(meta, Mapping):
        raise InvalidNewsProviderDataError(
            "Marketaux response meta must be a mapping"
        )

    found = _pagination_integer(
        meta,
        "found",
        minimum=0,
    )
    returned = _pagination_integer(
        meta,
        "returned",
        minimum=0,
    )
    limit = _pagination_integer(
        meta,
        "limit",
        minimum=1,
    )
    page = _pagination_integer(
        meta,
        "page",
        minimum=1,
    )

    if returned > limit:
        raise InvalidNewsProviderDataError(
            "Marketaux meta returned cannot "
            "exceed limit"
        )

    if returned != len(items):
        raise InvalidNewsProviderDataError(
            "Marketaux meta returned must match "
            "the number of data items"
        )

    if found < returned:
        raise InvalidNewsProviderDataError(
            "Marketaux meta found cannot be "
            "less than returned"
        )

    return MarketauxResponse(
        found=found,
        returned=returned,
        limit=limit,
        page=page,
        items=tuple(items),
    )

def _require_request_string(
    value: str,
    field: str,
) -> str:
    if not isinstance(value, str):
        raise InvalidNewsProviderDataError(
            f"{field} must be a string"
        )

    if not value.strip():
        raise InvalidNewsProviderDataError(
            f"{field} cannot be blank"
        )

    return value


def _require_utc_window_value(
    value: datetime,
    field: str,
) -> None:
    if not isinstance(value, datetime):
        raise InvalidNewsProviderDataError(
            f"{field} must be a datetime"
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise InvalidNewsProviderDataError(
            f"{field} must be timezone-aware UTC"
        )

    if value.utcoffset().total_seconds() != 0:
        raise InvalidNewsProviderDataError(
            f"{field} must explicitly represent UTC"
        )


def _format_marketaux_datetime(
    value: datetime,
) -> str:
    return (
        value.replace(
            tzinfo=None,
        )
        .isoformat(
            timespec="seconds",
        )
        + "Z"
    )


def build_marketaux_request(
    *,
    base_url: str,
    api_token: str,
    query: NewsQuery,
    start_at_utc: datetime,
    end_at_utc: datetime,
    page: int = 1,
) -> tuple[str, dict[str, str]]:

    """Build one deterministic Marketaux news request.

    This function performs no network activity and never logs or
    otherwise exposes the API token.
    """

    base_url = _require_request_string(
        base_url,
        "base_url",
    )
    api_token = _require_request_string(
        api_token,
        "api_token",
    )

    if not isinstance(query, NewsQuery):
        raise InvalidNewsProviderDataError(
            "query must be a NewsQuery"
        )

    _require_utc_window_value(
        start_at_utc,
        "start_at_utc",
    )
    _require_utc_window_value(
        end_at_utc,
        "end_at_utc",
    )

    try:
        validate_news_query_window(
            start_at_utc=start_at_utc,
            end_at_utc=end_at_utc,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidNewsProviderDataError(
            "start_at_utc must be earlier than end_at_utc"
        ) from exc

    parsed_base = urlsplit(
        base_url,
    )

    if (
        parsed_base.scheme not in {"http", "https"}
        or not parsed_base.hostname
    ):
        raise InvalidNewsProviderDataError(
            "base_url must be an absolute HTTP(S) URL"
        )

    endpoint = (
        base_url.rstrip("/")
        + "/news/all"
    )

    search = " | ".join(
        f'"{term}"'
        for term in query.strict_terms
    )

    if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or page < 1
    ):
        raise ValueError(
            "page must be an integer >= 1"
        )

    params = {
        "api_token": api_token,
        "search": search,
        "language": "en",
        "page": str(page),
        "published_after": (
            _format_marketaux_datetime(
                start_at_utc
            )
        ),
        "published_before": (
            _format_marketaux_datetime(
                end_at_utc
            )
        ),
    }

    return endpoint, params

class MarketauxNewsProvider(NewsProvider):
    """Single-page Marketaux implementation of the news provider contract."""

    def __init__(
            self,
            *,
            base_url: str,
            api_token: str,
            timeout_seconds: float,
            transport: MarketauxTransport,
            clock: MarketauxClock,
    ) -> None:
        self._base_url = _require_request_string(
            base_url,
            "base_url",
        )
        self._api_token = _require_request_string(
            api_token,
            "api_token",
        )

        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be greater than zero"
            )

        if transport is None:
            raise TypeError(
                "transport cannot be None"
            )

        if not callable(
            getattr(
                transport,
                "get_json",
                None,
            )
        ):
            raise TypeError(
                "transport must provide get_json"
            )

        if clock is None:
            raise TypeError(
                "clock cannot be None"
            )

        if not callable(
                getattr(
                    clock,
                    "now_utc",
                    None,
                )
        ):
            raise TypeError(
                "clock must provide now_utc"
            )

        self._timeout_seconds = float(
            timeout_seconds
        )
        self._transport = transport
        self._clock = clock

    def fetch_articles(
            self,
            *,
            query: NewsQuery,
            start_at_utc: datetime,
            end_at_utc: datetime,
    ) -> list[ArticleRecord]:
        articles: list[ArticleRecord] = []
        page = 1

        while True:
            url, params = build_marketaux_request(
                base_url=self._base_url,
                api_token=self._api_token,
                query=query,
                start_at_utc=start_at_utc,
                end_at_utc=end_at_utc,
                page=page,
            )

            try:
                status_code, payload = (
                    self._transport.get_json(
                        url=url,
                        params=params,
                        timeout_seconds=(
                            self._timeout_seconds
                        ),
                    )
                )
            except NewsProviderError:
                raise
            except Exception as exc:
                raise NewsProviderUnavailableError(
                    "Marketaux transport request "
                    "failed"
                ) from exc

            response = parse_marketaux_response(
                status_code=status_code,
                payload=payload,
            )

            if response.page != page:
                raise InvalidNewsProviderDataError(
                    "Marketaux response page does "
                    "not match requested page"
                )

            retrieved_at_utc = (
                self._clock.now_utc()
            )

            retrieved_at_utc = (
                _validate_retrieved_at(
                    retrieved_at_utc
                )
            )

            for item in response.items:
                article = normalize_marketaux_article(
                    payload=item,
                    query=query,
                    retrieved_at_utc=(
                        retrieved_at_utc
                    ),
                )

                if (
                        start_at_utc
                        <= article.published_at_utc
                        < end_at_utc
                ):
                    articles.append(article)

            if response.returned < response.limit:
                break

            page += 1

        return assign_duplicate_groups(
            articles
        )

def _pagination_integer(
    meta: Mapping[str, object],
    field: str,
    *,
    minimum: int,
) -> int:
    if field not in meta:
        raise InvalidNewsProviderDataError(
            f"Marketaux meta missing {field}"
        )

    value = meta[field]

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise InvalidNewsProviderDataError(
            f"Marketaux meta {field} must be "
            f"an integer >= {minimum}"
        )

    return value
