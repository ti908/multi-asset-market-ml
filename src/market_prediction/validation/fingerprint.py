"""Deterministic fingerprints for normalized market-data frames."""

from __future__ import annotations

import hashlib

import pandas as pd


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 identity for a normalized DataFrame."""

    digest = hashlib.sha256()

    digest.update(
        "\x1f".join(str(column) for column in frame.columns).encode("utf-8")
    )
    digest.update(b"\x00")

    digest.update(
        "\x1f".join(str(dtype) for dtype in frame.dtypes).encode("utf-8")
    )
    digest.update(b"\x00")

    row_hashes = pd.util.hash_pandas_object(
        frame,
        index=False,
        categorize=True,
    )

    digest.update(row_hashes.to_numpy().tobytes())

    return digest.hexdigest()