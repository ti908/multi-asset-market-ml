"""Deterministic identities and text fingerprints for sentiment scoring."""

from __future__ import annotations

import hashlib
import json


def _require_nonblank_string(
    value: str,
    *,
    field_name: str,
) -> str:
    """Require a string containing at least one non-whitespace character."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string"
        )

    if not value.strip():
        raise ValueError(
            f"{field_name} must not be blank"
        )

    return value


def sentiment_input_text_hash(
    input_text: str,
) -> str:
    """Return the SHA-256 fingerprint of the exact scorer input text.

    The text is intentionally not stripped, lowercased, normalized, or
    otherwise modified before hashing. This preserves the identity of
    the exact representation supplied to a sentiment scorer.
    """

    _require_nonblank_string(
        input_text,
        field_name="input_text",
    )

    return hashlib.sha256(
        input_text.encode("utf-8")
    ).hexdigest()


def sentiment_identity(
    *,
    article_id: str,
    asset: str,
    model_name: str,
    model_revision: str,
    scoring_method: str,
) -> str:
    """Return the deterministic identity of one sentiment observation.

    Identity includes the source article, project asset, model,
    immutable model revision, and scoring method. The exact scorer input
    text is tracked separately through ``input_text_hash``.
    """

    identity_fields = {
        "article_id": _require_nonblank_string(
            article_id,
            field_name="article_id",
        ),
        "asset": _require_nonblank_string(
            asset,
            field_name="asset",
        ),
        "model_name": _require_nonblank_string(
            model_name,
            field_name="model_name",
        ),
        "model_revision": _require_nonblank_string(
            model_revision,
            field_name="model_revision",
        ),
        "scoring_method": _require_nonblank_string(
            scoring_method,
            field_name="scoring_method",
        ),
    }

    canonical_payload = json.dumps(
        identity_fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical_payload.encode("utf-8")
    ).hexdigest()