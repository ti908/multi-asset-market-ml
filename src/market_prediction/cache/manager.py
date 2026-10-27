"""Atomic persistence for validated OHLCV market-data caches."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import quote

import pandas as pd

from market_prediction.cache.models import (
    CacheMetadata,
    CacheReadResult,
    CacheState,
    ColumnPriceBasis,
    HistoricalStability,
    PriceAdjustment,
    RestatementSummary,
    SoftAnomalySummary,
    ValidatedDataset,
    ValidationGateError,
)
from market_prediction.validation import Status
from market_prediction.validation.fingerprint import dataframe_fingerprint


class CacheManager:
    """Persist and retrieve validated market data.

    Writes use immutable generation directories. A generation becomes visible
    only when ``current.json`` is atomically replaced. Therefore a failed
    refresh cannot expose a half-published data/metadata pair.

    Successful writes are full dataset replacements, never blind appends.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        ttl_hours: int,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if ttl_hours <= 0:
            raise ValueError("ttl_hours must be greater than zero")

        self._cache_dir = Path(cache_dir)
        self._ttl = timedelta(hours=ttl_hours)
        self._now = now or (
            lambda: datetime.now(timezone.utc)
        )

    def write(
            self,
            dataset: ValidatedDataset,
            *,
            provider: str,
            fetch_timestamp: datetime,
            price_adjustment_mode: str,
            price_basis: Mapping[str, ColumnPriceBasis],
            bar_convention: str | None = None,
    ) -> CacheMetadata:
        """Atomically publish a complete validated cache generation."""

        if not isinstance(dataset, ValidatedDataset):
            raise ValidationGateError(
                "CacheManager.write() accepts only ValidatedDataset; "
                "raw DataFrames cannot be cached."
            )

        if dataset.report.status is not Status.PASSED:
            raise ValidationGateError(
                "CacheManager.write() received a dataset whose validation "
                "report is no longer PASSED."
            )

        frame = dataset.data

        if frame.empty:
            raise ValidationGateError(
                "Validated cache dataset cannot be empty."
            )

        actual_id = dataframe_fingerprint(frame)

        if actual_id != dataset.dataset_id:
            raise ValidationGateError(
                "Validated dataset content no longer matches its "
                "validation fingerprint."
            )

        fetch_timestamp = self._require_utc_timestamp(
            fetch_timestamp,
            field="fetch_timestamp",
        )

        if not provider.strip():
            raise ValueError("provider must not be empty")

        if not price_adjustment_mode.strip():
            raise ValueError(
                "price_adjustment_mode must not be empty"
            )
        if (
            bar_convention is not None
            and not bar_convention.strip()
        ):
            raise ValueError(
                "bar_convention must not be empty when provided"
            )

        asset = dataset.report.asset
        previous = self.read(asset)

        restatement = None

        if (
            previous.exists
            and previous.data is not None
            and previous.metadata is not None
            and previous.metadata.provider == provider
        ):
            restatement = self._detect_restatement(
                previous.data,
                frame,
            )

        stamps = pd.to_datetime(
            frame["timestamp"],
            utc=True,
        )

        metadata = CacheMetadata(
            asset=asset,
            provider=provider,
            fetch_timestamp=fetch_timestamp,
            date_start=stamps.min().date(),
            date_end=stamps.max().date(),
            row_count=len(frame),
            price_adjustment_mode=price_adjustment_mode,
            validation_status=dataset.report.status,
            soft_anomalies=self._summarize_soft_anomalies(
                dataset
            ),
            price_basis=dict(price_basis),
            dataset_id=actual_id,
            restatement=restatement,
            bar_convention=bar_convention,
        )

        self._publish(
            asset=asset,
            frame=frame,
            metadata=metadata,
        )

        return metadata

    def read(
        self,
        asset: str,
    ) -> CacheReadResult:
        """Return the explicit state, data and metadata for an asset."""

        asset_dir = self._asset_dir(asset)
        pointer_path = asset_dir / "current.json"

        if not pointer_path.exists():
            return CacheReadResult(
                state=CacheState.MISSING,
                data=None,
                metadata=None,
            )

        try:
            pointer = self._read_json(pointer_path)

            generation = pointer["generation"]

            if (
                not isinstance(generation, str)
                or not generation
                or "/" in generation
                or "\\" in generation
            ):
                raise ValueError(
                    "Invalid cache generation pointer"
                )

            generation_dir = (
                asset_dir
                / "generations"
                / generation
            )

            data_path = generation_dir / "data.parquet"
            metadata_path = generation_dir / "metadata.json"

            if not data_path.is_file():
                raise FileNotFoundError(
                    "Cached Parquet generation is missing"
                )

            if not metadata_path.is_file():
                raise FileNotFoundError(
                    "Cached metadata generation is missing"
                )

            metadata = self._metadata_from_dict(
                self._read_json(metadata_path)
            )

            if metadata.asset != asset:
                raise ValueError(
                    "Cached metadata asset does not match requested asset"
                )

            frame = pd.read_parquet(
                data_path,
                engine="pyarrow",
            )

            if len(frame) != metadata.row_count:
                raise ValueError(
                    "Cached row count does not match metadata"
                )

            actual_id = dataframe_fingerprint(frame)

            if actual_id != metadata.dataset_id:
                raise ValueError(
                    "Cached dataset fingerprint does not match metadata"
                )

            stamps = pd.to_datetime(
                frame["timestamp"],
                utc=True,
            )

            if frame.empty:
                raise ValueError(
                    "Cached dataset is unexpectedly empty"
                )

            if stamps.min().date() != metadata.date_start:
                raise ValueError(
                    "Cached start date does not match metadata"
                )

            if stamps.max().date() != metadata.date_end:
                raise ValueError(
                    "Cached end date does not match metadata"
                )

            now = self._require_utc_timestamp(
                self._now(),
                field="current time",
            )

            age = now - metadata.fetch_timestamp

            state = (
                CacheState.FRESH
                if age <= self._ttl
                else CacheState.STALE
            )

            return CacheReadResult(
                state=state,
                data=frame,
                metadata=metadata,
            )

        except Exception as exc:
            return CacheReadResult(
                state=CacheState.CORRUPTED,
                data=None,
                metadata=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _publish(
        self,
        *,
        asset: str,
        frame: pd.DataFrame,
        metadata: CacheMetadata,
    ) -> None:
        asset_dir = self._asset_dir(asset)
        generations_dir = asset_dir / "generations"

        generations_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        generation = uuid.uuid4().hex
        generation_dir = generations_dir / generation

        generation_dir.mkdir()

        data_tmp = generation_dir / "data.parquet.tmp"
        metadata_tmp = generation_dir / "metadata.json.tmp"

        data_final = generation_dir / "data.parquet"
        metadata_final = generation_dir / "metadata.json"

        pointer_tmp = (
            asset_dir
            / f".current-{generation}.json.tmp"
        )
        pointer_final = asset_dir / "current.json"

        pointer_published = False

        try:
            frame.to_parquet(
                data_tmp,
                engine="pyarrow",
                index=False,
            )

            self._fsync_file(data_tmp)

            metadata_tmp.write_text(
                json.dumps(
                    self._metadata_to_dict(metadata),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            self._fsync_file(metadata_tmp)

            os.replace(
                data_tmp,
                data_final,
            )
            os.replace(
                metadata_tmp,
                metadata_final,
            )

            self._fsync_directory(generation_dir)

            pointer_tmp.write_text(
                json.dumps(
                    {
                        "generation": generation,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            self._fsync_file(pointer_tmp)

            os.replace(
                pointer_tmp,
                pointer_final,
            )
            pointer_published = True

            self._fsync_directory(asset_dir)

        except Exception:
            if not pointer_published:
                pointer_tmp.unlink(
                    missing_ok=True
                )
                shutil.rmtree(
                    generation_dir,
                    ignore_errors=True,
                )
            raise

    @staticmethod
    def _summarize_soft_anomalies(
        dataset: ValidatedDataset,
    ) -> tuple[SoftAnomalySummary, ...]:
        counts = Counter(
            issue.rule
            for issue in dataset.report.soft_anomalies
        )

        return tuple(
            SoftAnomalySummary(
                rule=rule,
                count=counts[rule],
            )
            for rule in sorted(counts)
        )

    @staticmethod
    def _detect_restatement(
        previous: pd.DataFrame,
        fresh: pd.DataFrame,
    ) -> RestatementSummary | None:
        compare_columns = (
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        )

        old = previous.copy(deep=True)
        new = fresh.copy(deep=True)

        old["timestamp"] = pd.to_datetime(
            old["timestamp"],
            utc=True,
        )
        new["timestamp"] = pd.to_datetime(
            new["timestamp"],
            utc=True,
        )

        old = old.set_index("timestamp")
        new = new.set_index("timestamp")

        overlap = old.index.intersection(
            new.index
        ).sort_values()

        if overlap.empty:
            return None

        changed_columns: set[str] = set()
        changed_rows: set[pd.Timestamp] = set()

        for column in compare_columns:
            if (
                column not in old.columns
                or column not in new.columns
            ):
                continue

            old_values = old.loc[
                overlap,
                column,
            ]
            new_values = new.loc[
                overlap,
                column,
            ]

            equal = (
                old_values.eq(new_values)
                | (
                    old_values.isna()
                    & new_values.isna()
                )
            )

            if not bool(equal.all()):
                changed_columns.add(column)

                changed_rows.update(
                    overlap[
                        ~equal.to_numpy()
                    ]
                )

        return RestatementSummary(
            detected=bool(changed_rows),
            overlap_start=overlap.min().date(),
            overlap_end=overlap.max().date(),
            changed_rows=len(changed_rows),
            changed_columns=tuple(
                sorted(changed_columns)
            ),
        )

    @staticmethod
    def _metadata_to_dict(
        metadata: CacheMetadata,
    ) -> dict:
        restatement = None

        if metadata.restatement is not None:
            restatement = {
                "detected": metadata.restatement.detected,
                "overlap_start": (
                    metadata.restatement.overlap_start.isoformat()
                ),
                "overlap_end": (
                    metadata.restatement.overlap_end.isoformat()
                ),
                "changed_rows": metadata.restatement.changed_rows,
                "changed_columns": list(
                    metadata.restatement.changed_columns
                ),
            }

        return {
            "asset": metadata.asset,
            "provider": metadata.provider,
            "fetch_timestamp": (
                metadata.fetch_timestamp.isoformat()
            ),
            "date_start": metadata.date_start.isoformat(),
            "date_end": metadata.date_end.isoformat(),
            "row_count": metadata.row_count,
            "price_adjustment_mode": (
                metadata.price_adjustment_mode
            ),
            "validation_status": (
                metadata.validation_status.value
            ),
            "soft_anomalies": [
                {
                    "rule": item.rule,
                    "count": item.count,
                }
                for item in metadata.soft_anomalies
            ],
            "price_basis": {
                column: {
                    "adjustment": basis.adjustment.value,
                    "historical_stability": (
                        basis.historical_stability.value
                    ),
                }
                for column, basis in sorted(
                    metadata.price_basis.items()
                )
            },
            "dataset_id": metadata.dataset_id,
            "restatement": restatement,
            "bar_convention": metadata.bar_convention,
        }

    @staticmethod
    def _metadata_from_dict(
        raw: dict,
    ) -> CacheMetadata:
        required = {
            "asset",
            "provider",
            "fetch_timestamp",
            "date_start",
            "date_end",
            "row_count",
            "price_adjustment_mode",
            "validation_status",
            "soft_anomalies",
            "price_basis",
            "dataset_id",
            "restatement",
        }

        optional = {
            "bar_convention",
        }

        keys = set(raw)

        if (
            not required.issubset(keys)
            or not keys.issubset(required | optional)
        ):
            raise ValueError(
                "Cached metadata has unexpected schema"
            )

        status = Status(
            raw["validation_status"]
        )

        if status is not Status.PASSED:
            raise ValueError(
                "Persisted cache metadata is not PASSED"
            )

        soft_anomalies = tuple(
            SoftAnomalySummary(
                rule=item["rule"],
                count=int(item["count"]),
            )
            for item in raw["soft_anomalies"]
        )

        price_basis = {
            column: ColumnPriceBasis(
                adjustment=PriceAdjustment(
                    item["adjustment"]
                ),
                historical_stability=HistoricalStability(
                    item["historical_stability"]
                ),
            )
            for column, item in raw[
                "price_basis"
            ].items()
        }

        restatement_raw = raw["restatement"]
        restatement = None

        if restatement_raw is not None:
            restatement = RestatementSummary(
                detected=bool(
                    restatement_raw["detected"]
                ),
                overlap_start=date.fromisoformat(
                    restatement_raw["overlap_start"]
                ),
                overlap_end=date.fromisoformat(
                    restatement_raw["overlap_end"]
                ),
                changed_rows=int(
                    restatement_raw["changed_rows"]
                ),
                changed_columns=tuple(
                    restatement_raw[
                        "changed_columns"
                    ]
                ),
            )

        fetch_timestamp = datetime.fromisoformat(
            raw["fetch_timestamp"]
        )

        fetch_timestamp = CacheManager._require_utc_timestamp(
            fetch_timestamp,
            field="fetch_timestamp",
        )

        return CacheMetadata(
            asset=str(raw["asset"]),
            provider=str(raw["provider"]),
            fetch_timestamp=fetch_timestamp,
            date_start=date.fromisoformat(
                raw["date_start"]
            ),
            date_end=date.fromisoformat(
                raw["date_end"]
            ),
            row_count=int(raw["row_count"]),
            price_adjustment_mode=str(
                raw["price_adjustment_mode"]
            ),
            validation_status=status,
            soft_anomalies=soft_anomalies,
            price_basis=price_basis,
            dataset_id=str(raw["dataset_id"]),
            restatement=restatement,
            bar_convention=(
                None
                if raw.get("bar_convention") is None
                else str(raw["bar_convention"])
            ),
        )

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict:
        raw = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(raw, dict):
            raise ValueError(
                f"{path.name} must contain a JSON object"
            )

        return raw

    def _asset_dir(
        self,
        asset: str,
    ) -> Path:
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(
                "asset must be a non-empty string"
            )

        safe_asset = quote(
            asset,
            safe="-_.=",
        )

        return self._cache_dir / safe_asset

    @staticmethod
    def _require_utc_timestamp(
        value: datetime,
        *,
        field: str,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                f"{field} must be timezone-aware"
            )

        return value.astimezone(
            timezone.utc
        )

    @staticmethod
    def _fsync_file(
        path: Path,
    ) -> None:
        with path.open("rb") as handle:
            os.fsync(
                handle.fileno()
            )

    @staticmethod
    def _fsync_directory(
        path: Path,
    ) -> None:
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY,
            )
        except OSError:
            return

        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)