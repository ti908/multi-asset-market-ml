"""Unit tests for project configuration loading."""

from datetime import date
from pathlib import Path

import pytest

from market_prediction.config.loader import (
    AssetClass,
    ConfigLoadError,
    ProviderName,
    load_assets_config,
    load_pipeline_config,
)


VALID_ASSETS_YAML = """
assets:
  AAPL:
    display_name: "Apple"
    ticker: "AAPL"
    asset_class: "equity"
    currency: "USD"
    calendar: "XNYS"
    primary_provider: "yfinance"
    backup_provider: "stooq"

  MSFT:
    display_name: "Microsoft"
    ticker: "MSFT"
    asset_class: "equity"
    currency: "USD"
    calendar: "XNYS"
    primary_provider: "yfinance"
    backup_provider: "stooq"

  BTC_USD:
    display_name: "Bitcoin"
    ticker: "BTC-USD"
    asset_class: "crypto"
    currency: "USD"
    calendar: "24/7"
    primary_provider: "yfinance"
    backup_provider: "stooq"

  EURUSD:
    display_name: "EUR/USD"
    ticker: "EURUSD=X"
    asset_class: "forex"
    currency: "USD"
    calendar: "24/5"
    primary_provider: "yfinance"
    backup_provider: "dukascopy"

  GOLD:
    display_name: "Gold Futures"
    ticker: "GC=F"
    asset_class: "commodity"
    currency: "USD"
    calendar: "COMEX"
    primary_provider: "yfinance"
    backup_provider: "stooq"
"""


VALID_PIPELINE_YAML = """
history:
  start_date: "2015-01-01"
  interval: "1d"

cache:
  ttl_hours: 24
  format: "parquet"

retry:
  max_attempts: 3
  initial_delay_seconds: 2.0
  backoff_multiplier: 2.0

validation:
  asset_classes:
    equity:
      large_daily_move_threshold: 0.20
      volume_spike_multiplier: 5.0
      volume_rolling_window: 20
      max_missing_sessions: 2
      check_zero_volume: true
      calendar: XNYS
      closed_dates: []

    crypto:
      large_daily_move_threshold: 0.40
      volume_spike_multiplier: 8.0
      volume_rolling_window: 20
      max_missing_sessions: 1
      check_zero_volume: true
      calendar: 24/7
      closed_dates: []

    forex:
      large_daily_move_threshold: 0.05
      volume_spike_multiplier: 5.0
      volume_rolling_window: 20
      max_missing_sessions: 2
      check_zero_volume: false
      calendar: 24/5
      closed_dates: []

    commodity:
      large_daily_move_threshold: 0.15
      volume_spike_multiplier: 6.0
      volume_rolling_window: 20
      max_missing_sessions: 2
      check_zero_volume: true
      calendar: COMEX
      closed_dates: []

paths:
  raw: "data/raw"
  cache: "data/cache"
  processed: "data/processed"
  synthetic: "data/synthetic"
"""


def write_config(
    tmp_path: Path,
    filename: str,
    contents: str,
) -> Path:
    """Create a temporary YAML configuration file for a test."""
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")
    return path


def test_valid_assets_config_loads_into_typed_objects(
    tmp_path: Path,
) -> None:
    """A valid asset registry should load into typed objects."""
    path = write_config(
        tmp_path,
        "assets.yaml",
        VALID_ASSETS_YAML,
    )

    config = load_assets_config(path)

    assert len(config.assets) == 5

    aapl = config.assets["AAPL"]

    assert aapl.display_name == "Apple"
    assert aapl.ticker == "AAPL"
    assert aapl.asset_class is AssetClass.EQUITY
    assert aapl.currency == "USD"
    assert aapl.calendar == "XNYS"
    assert aapl.primary_provider is ProviderName.YFINANCE
    assert aapl.backup_provider is ProviderName.STOOQ

    assert config.assets["BTC_USD"].asset_class is AssetClass.CRYPTO
    assert config.assets["EURUSD"].asset_class is AssetClass.FOREX
    assert config.assets["GOLD"].asset_class is AssetClass.COMMODITY


def test_valid_pipeline_config_loads_into_typed_objects(
    tmp_path: Path,
) -> None:
    """A valid pipeline configuration should be parsed correctly."""
    path = write_config(
        tmp_path,
        "pipeline.yaml",
        VALID_PIPELINE_YAML,
    )

    config = load_pipeline_config(path)

    assert config.history.start_date == date(2015, 1, 1)
    assert config.history.interval == "1d"

    assert config.cache.ttl_hours == 24
    assert config.cache.format == "parquet"

    assert config.retry.max_attempts == 3
    assert config.retry.initial_delay_seconds == 2.0
    assert config.retry.backoff_multiplier == 2.0

    equity_validation = config.validation.asset_classes.equity
    assert equity_validation.large_daily_move_threshold == 0.20
    assert equity_validation.volume_spike_multiplier == 5.0
    assert equity_validation.volume_rolling_window == 20
    assert equity_validation.max_missing_sessions == 2
    assert equity_validation.check_zero_volume is True
    assert equity_validation.calendar == "XNYS"
    assert equity_validation.closed_dates == []

    assert config.paths.raw == Path("data/raw")
    assert config.paths.cache == Path("data/cache")
    assert config.paths.processed == Path("data/processed")
    assert config.paths.synthetic == Path("data/synthetic")


def test_missing_required_asset_field_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A missing ticker should identify both asset and field."""
    invalid_yaml = """
assets:
  AAPL:
    display_name: "Apple"
    asset_class: "equity"
    currency: "USD"
    calendar: "XNYS"
    primary_provider: "yfinance"
    backup_provider: "stooq"
"""

    path = write_config(
        tmp_path,
        "assets.yaml",
        invalid_yaml,
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        load_assets_config(path)

    message = str(exc_info.value)

    assert "AAPL" in message
    assert "ticker" in message
    assert "Field required" in message


def test_missing_required_pipeline_field_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A missing TTL must fail rather than silently use a default."""
    invalid_yaml = """
history:
  start_date: "2015-01-01"
  interval: "1d"

cache:
  format: "parquet"

retry:
  max_attempts: 3
  initial_delay_seconds: 2.0
  backoff_multiplier: 2.0

validation:
  large_daily_move_threshold: 0.30
  allow_zero_volume: true

paths:
  raw: "data/raw"
  cache: "data/cache"
  processed: "data/processed"
  synthetic: "data/synthetic"
"""

    path = write_config(
        tmp_path,
        "pipeline.yaml",
        invalid_yaml,
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        load_pipeline_config(path)

    message = str(exc_info.value)

    assert "cache.ttl_hours" in message
    assert "Field required" in message


def test_malformed_yaml_raises_clear_error(
    tmp_path: Path,
) -> None:
    """Syntactically invalid YAML should produce a clear error."""
    malformed_yaml = """
assets:
  AAPL:
    ticker: "AAPL"
    asset_class: [equity
"""

    path = write_config(
        tmp_path,
        "assets.yaml",
        malformed_yaml,
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        load_assets_config(path)

    message = str(exc_info.value)

    assert "Malformed YAML" in message
    assert "assets.yaml" in message


def test_unknown_asset_class_is_rejected(
    tmp_path: Path,
) -> None:
    """Unsupported asset classes must not be silently accepted."""
    invalid_yaml = VALID_ASSETS_YAML.replace(
        'asset_class: "equity"',
        'asset_class: "collectible"',
        1,
    )

    path = write_config(
        tmp_path,
        "assets.yaml",
        invalid_yaml,
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        load_assets_config(path)

    message = str(exc_info.value)

    assert "AAPL" in message
    assert "asset_class" in message
    assert "collectible" in message


def test_unknown_provider_is_rejected(
    tmp_path: Path,
) -> None:
    """Unsupported provider names must not be silently accepted."""
    invalid_yaml = VALID_ASSETS_YAML.replace(
        'primary_provider: "yfinance"',
        'primary_provider: "unknown_provider"',
        1,
    )

    path = write_config(
        tmp_path,
        "assets.yaml",
        invalid_yaml,
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        load_assets_config(path)

    message = str(exc_info.value)

    assert "AAPL" in message
    assert "primary_provider" in message
    assert "unknown_provider" in message


def test_missing_config_file_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A missing configuration file should identify its path."""
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(ConfigLoadError) as exc_info:
        load_assets_config(missing_path)

    message = str(exc_info.value)

    assert "Configuration file not found" in message
    assert "does_not_exist.yaml" in message


def test_empty_yaml_file_is_rejected(
    tmp_path: Path,
) -> None:
    """An empty YAML file should not be treated as valid config."""
    path = write_config(
        tmp_path,
        "assets.yaml",
        "",
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        load_assets_config(path)

    assert "is empty" in str(exc_info.value)


def test_unexpected_asset_field_is_rejected(
    tmp_path: Path,
) -> None:
    """Unknown fields should fail to catch configuration typos."""
    invalid_yaml = VALID_ASSETS_YAML.replace(
        '    currency: "USD"',
        '    currency: "USD"\n    accidental_field: "oops"',
        1,
    )

    path = write_config(
        tmp_path,
        "assets.yaml",
        invalid_yaml,
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        load_assets_config(path)

    message = str(exc_info.value)

    assert "AAPL" in message
    assert "accidental_field" in message

def test_unsupported_cache_format_is_rejected(tmp_path):
    pipeline_yaml = VALID_PIPELINE_YAML.replace(
        'format: "parquet"',
        'format: "csv"',
    )

    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        pipeline_yaml,
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(pipeline_path)