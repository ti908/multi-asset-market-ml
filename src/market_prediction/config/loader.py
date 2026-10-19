"""Typed configuration loading for the market prediction project.

This module is intentionally limited to configuration concerns.

Responsibilities:
- Read YAML configuration files.
- Parse YAML safely.
- Validate configuration structure and allowed values.
- Return strongly typed Pydantic objects.
- Produce clear errors when configuration is invalid.

It contains no provider, ingestion, caching, or modeling logic.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Literal

import yaml

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    model_validator,
)


class ConfigLoadError(Exception):
    """Raised when a project configuration file cannot be loaded or validated."""


class AssetClass(str, Enum):
    """Asset classes supported by version 1 of the project."""

    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"


class ProviderName(str, Enum):
    """Market-data providers supported by the current configuration contract."""

    YFINANCE = "yfinance"
    STOOQ = "stooq"
    DUKASCOPY = "dukascopy"


class AssetConfig(BaseModel):
    """Configuration for one market asset."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    asset_class: AssetClass
    currency: str = Field(min_length=1)
    calendar: str = Field(min_length=1)
    primary_provider: ProviderName
    backup_provider: ProviderName


class AssetsConfig(BaseModel):
    """Top-level asset registry."""

    model_config = ConfigDict(extra="forbid")

    assets: dict[str, AssetConfig] = Field(min_length=1)


class HistoryConfig(BaseModel):
    """Historical market-data settings."""

    model_config = ConfigDict(extra="forbid")

    start_date: date
    interval: str = Field(min_length=1)


class CacheConfig(BaseModel):
    """Cache behavior settings."""

    model_config = ConfigDict(extra="forbid")

    ttl_hours: int = Field(gt=0)
    format: Literal["parquet"]


class RetryConfig(BaseModel):
    """Retry and exponential-backoff settings."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(gt=0)
    initial_delay_seconds: float = Field(gt=0)
    backoff_multiplier: float = Field(gt=0)


class AssetValidationConfig(BaseModel):
    """Validation and anomaly-detection settings for one asset class."""

    model_config = ConfigDict(extra="forbid")

    large_daily_move_threshold: float = Field(gt=0)
    volume_spike_multiplier: float = Field(gt=1)
    volume_rolling_window: int = Field(gt=1)
    max_missing_sessions: int = Field(ge=0)
    check_zero_volume: StrictBool
    calendar: str = Field(min_length=1)
    closed_dates: list[date] = Field(default_factory=list)


class AssetClassValidationConfig(BaseModel):
    """Validation settings for every supported asset class."""

    model_config = ConfigDict(extra="forbid")

    equity: AssetValidationConfig
    crypto: AssetValidationConfig
    forex: AssetValidationConfig
    commodity: AssetValidationConfig

    @model_validator(mode="after")
    def validate_calendars(self) -> "AssetClassValidationConfig":
        """Ensure each asset class uses its intended calendar strategy."""

        expected_calendars = {
            "equity": "XNYS",
            "crypto": "24/7",
            "forex": "24/5",
            "commodity": "COMEX",
        }

        for asset_class, expected_calendar in expected_calendars.items():
            settings = getattr(self, asset_class)

            if settings.calendar != expected_calendar:
                raise ValueError(
                    f"{asset_class} calendar must be "
                    f"'{expected_calendar}', received "
                    f"'{settings.calendar}'"
                )

        return self


class ValidationConfig(BaseModel):
    """Pipeline-wide data-validation settings."""

    model_config = ConfigDict(extra="forbid")

    asset_classes: AssetClassValidationConfig


class PathConfig(BaseModel):
    """Project data locations."""

    model_config = ConfigDict(extra="forbid")

    raw: Path
    cache: Path
    processed: Path
    synthetic: Path


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    history: HistoryConfig
    cache: CacheConfig
    retry: RetryConfig
    validation: ValidationConfig
    paths: PathConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its top-level mapping.

    Args:
        path:
            Path to the YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ConfigLoadError:
            If the file does not exist, cannot be read, contains malformed
            YAML, is empty, or does not contain a mapping at the top level.
    """
    if not path.exists():
        raise ConfigLoadError(
            f"Configuration file not found: '{path}'."
        )

    if not path.is_file():
        raise ConfigLoadError(
            f"Configuration path is not a file: '{path}'."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ConfigLoadError(
            f"Malformed YAML in configuration file '{path}': {exc}"
        ) from exc

    except OSError as exc:
        raise ConfigLoadError(
            f"Unable to read configuration file '{path}': {exc}"
        ) from exc

    if data is None:
        raise ConfigLoadError(
            f"Configuration file '{path}' is empty."
        )

    if not isinstance(data, dict):
        raise ConfigLoadError(
            f"Configuration file '{path}' must contain "
            "a YAML mapping at the top level."
        )

    return data


def _format_validation_error(
    path: Path,
    error: ValidationError,
) -> str:
    """Convert Pydantic validation errors into clear project-specific messages."""
    messages: list[str] = []

    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        message = item["msg"]

        invalid_input = item.get("input")

        if invalid_input is not None:
            messages.append(
                f"{location}: {message} "
                f"(received: {invalid_input!r})"
            )
        else:
            messages.append(
                f"{location}: {message}"
            )

    details = "; ".join(messages)

    return (
        f"Invalid configuration in '{path}': "
        f"{details}"
    )


def load_assets_config(path: str | Path) -> AssetsConfig:
    """Load and validate the asset registry.

    Args:
        path:
            Path to assets.yaml.

    Returns:
        Validated AssetsConfig object.

    Raises:
        ConfigLoadError:
            If the file cannot be loaded or its contents violate the
            asset configuration contract.
    """
    config_path = Path(path)
    raw_config = _read_yaml(config_path)

    try:
        return AssetsConfig.model_validate(raw_config)

    except ValidationError as exc:
        raise ConfigLoadError(
            _format_validation_error(config_path, exc)
        ) from exc


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate pipeline-wide configuration.

    Args:
        path:
            Path to pipeline.yaml.

    Returns:
        Validated PipelineConfig object.

    Raises:
        ConfigLoadError:
            If the file cannot be loaded or its contents violate the
            pipeline configuration contract.
    """
    config_path = Path(path)
    raw_config = _read_yaml(config_path)

    try:
        return PipelineConfig.model_validate(raw_config)

    except ValidationError as exc:
        raise ConfigLoadError(
            _format_validation_error(config_path, exc)
        ) from exc


def load_project_config(
    assets_path: str | Path,
    pipeline_path: str | Path,
) -> tuple[AssetsConfig, PipelineConfig]:
    """Load both project configuration files.

    This is the public convenience function for callers that need the
    complete project configuration.

    Args:
        assets_path:
            Path to assets.yaml.

        pipeline_path:
            Path to pipeline.yaml.

    Returns:
        A tuple containing:
            (AssetsConfig, PipelineConfig)
    """
    assets_config = load_assets_config(assets_path)
    pipeline_config = load_pipeline_config(pipeline_path)

    return assets_config, pipeline_config