"""Step 3d configuration regression tests."""
from copy import deepcopy
import pytest
import yaml
from market_prediction.config import ConfigLoadError, load_pipeline_config

@pytest.fixture
def payload():
    with open("config/pipeline.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def write(tmp_path, payload):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path

def test_asset_class_validation_config_loads():
    cfg = load_pipeline_config("config/pipeline.yaml")
    assert cfg.validation.asset_classes.forex.large_daily_move_threshold == .05
    assert cfg.validation.asset_classes.crypto.check_zero_volume
    assert cfg.validation.asset_classes.commodity.calendar == "COMEX"

@pytest.mark.parametrize("field,value", [
    ("large_daily_move_threshold", 0),
    ("volume_spike_multiplier", 1),
    ("volume_rolling_window", 1),
    ("max_missing_sessions", -1),
])
def test_bad_threshold_rejected(payload, tmp_path, field, value):
    payload["validation"]["asset_classes"]["equity"][field] = value
    with pytest.raises(ConfigLoadError):
        load_pipeline_config(write(tmp_path, payload))

def test_missing_asset_class_rejected(payload, tmp_path):
    del payload["validation"]["asset_classes"]["forex"]
    with pytest.raises(ConfigLoadError):
        load_pipeline_config(write(tmp_path, payload))

def test_old_global_thresholds_rejected(payload, tmp_path):
    payload["validation"]["large_daily_move_threshold"] = .30
    with pytest.raises(ConfigLoadError):
        load_pipeline_config(write(tmp_path, payload))

def test_incorrect_calendar_rejected(payload, tmp_path):
    payload["validation"]["asset_classes"]["commodity"]["calendar"] = "XNYS"
    with pytest.raises(ConfigLoadError):
        load_pipeline_config(write(tmp_path, payload))

def test_non_boolean_zero_volume_rejected(payload, tmp_path):
    payload["validation"]["asset_classes"]["forex"]["check_zero_volume"] = "yes"
    with pytest.raises(ConfigLoadError):
        load_pipeline_config(write(tmp_path, payload))
