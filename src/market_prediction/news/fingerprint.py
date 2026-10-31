"""Deterministic identities and content fingerprints for news records.

This module keeps upstream record identity separate from textual-content
identity.

It contains no provider fetching, duplicate grouping, storage, sentiment
scoring, or market-data alignment logic.
"""

from __future__ import annotations

import hashlib


_FIELD_SEPARATOR = b"\x1f"
_NULL_MARKER = b"\x00"
_VALUE_MARKER = b"\x01"


def _require_non_empty(
    value: str,
    *,
    field_name: str,
) -> str:
    """Return a required non-empty string or raise ValueError."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )

    return value


def _update_required_text(
    digest: hashlib._Hash,
    value: str,
) -> None:
    """Add one required text value to a digest."""

    encoded = value.encode("utf-8")

    digest.update(_VALUE_MARKER)
    digest.update(
        len(encoded).to_bytes(
            8,
            byteorder="big",
            signed=False,
        )
    )
    digest.update(encoded)
    digest.update(_FIELD_SEPARATOR)


def _update_optional_text(
    digest: hashlib._Hash,
    value: str | None,
) -> None:
    """Add optional text while distinguishing None from empty text."""

    if value is None:
        digest.update(_NULL_MARKER)
        digest.update(_FIELD_SEPARATOR)
        return

    encoded = value.encode("utf-8")

    digest.update(_VALUE_MARKER)
    digest.update(
        len(encoded).to_bytes(
            8,
            byteorder="big",
            signed=False,
        )
    )
    digest.update(encoded)
    digest.update(_FIELD_SEPARATOR)


def article_identity(
    *,
    provider: str,
    provider_item_id: str,
) -> str:
    """Return deterministic SHA-256 identity for one provider record.

    The identity deliberately uses both the canonical provider name and
    the provider's own item identifier.

    It does not attempt to determine whether another provider represents
    the same underlying news story. Cross-provider duplicate grouping is
    a separate concern.
    """

    provider = _require_non_empty(
        provider,
        field_name="provider",
    )

    provider_item_id = _require_non_empty(
        provider_item_id,
        field_name="provider_item_id",
    )

    digest = hashlib.sha256()

    _update_required_text(
        digest,
        provider,
    )

    _update_required_text(
        digest,
        provider_item_id,
    )

    return digest.hexdigest()


def content_fingerprint(
    *,
    title: str,
    snippet: str | None,
    full_text: str | None,
) -> str:
    """Return deterministic SHA-256 identity for retained article text.

    Text is fingerprinted exactly as supplied by the canonical record.
    No lowercasing, whitespace collapsing, punctuation removal, stemming,
    or semantic normalization is performed here.

    Missing optional text is distinct from present-but-empty text.
    """

    title = _require_non_empty(
        title,
        field_name="title",
    )

    digest = hashlib.sha256()

    _update_required_text(
        digest,
        title,
    )

    _update_optional_text(
        digest,
        snippet,
    )

    _update_optional_text(
        digest,
        full_text,
    )

    return digest.hexdigest()