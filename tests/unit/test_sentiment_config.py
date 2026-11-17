from pathlib import Path

import pytest
from pydantic import ValidationError

from market_prediction.config.loader import (
    SentimentConfig,
    SentimentInputTextStrategy,
    SentimentScorerName,
    load_pipeline_config,
)


def valid_sentiment_config() -> dict:
    """Return one valid sentiment configuration mapping."""
    return {
        "primary_scorer": "finbert",
        "baseline_scorer": "vader",
        "input_text_strategy": "title_and_snippet",
        "schema_version": "1",
        "finbert": {
            "model_name": "ProsusAI/finbert",
            "model_revision": (
                "0123456789abcdef"
                "0123456789abcdef"
                "01234567"
            ),
            "max_tokens": 512,
        },
    }


def test_sentiment_config_accepts_valid_configuration():
    config = SentimentConfig.model_validate(
        valid_sentiment_config()
    )

    assert (
        config.primary_scorer
        is SentimentScorerName.FINBERT
    )
    assert (
        config.baseline_scorer
        is SentimentScorerName.VADER
    )
    assert (
        config.input_text_strategy
        is SentimentInputTextStrategy.TITLE_AND_SNIPPET
    )
    assert config.schema_version == "1"

    assert (
        config.finbert.model_name
        == "ProsusAI/finbert"
    )
    assert len(config.finbert.model_revision) == 40
    assert config.finbert.max_tokens == 512


def test_sentiment_config_forbids_unknown_fields():
    raw = valid_sentiment_config()
    raw["unexpected"] = True

    with pytest.raises(ValidationError):
        SentimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("primary_scorer", "unknown"),
        ("baseline_scorer", "unknown"),
        ("input_text_strategy", "full_text"),
    ],
)
def test_sentiment_config_rejects_unknown_enum_values(
    field,
    value,
):
    raw = valid_sentiment_config()
    raw[field] = value

    with pytest.raises(ValidationError):
        SentimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "ProsusAI",
    ],
)
def test_finbert_model_name_rejects_invalid_values(
    value,
):
    raw = valid_sentiment_config()
    raw["finbert"]["model_name"] = value

    with pytest.raises(ValidationError):
        SentimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "main",
        "not-a-sha",
        "abc123",
        "g" * 40,
        "a" * 39,
        "a" * 41,
    ],
)
def test_finbert_revision_requires_full_commit_sha(
    value,
):
    raw = valid_sentiment_config()
    raw["finbert"]["model_revision"] = value

    with pytest.raises(ValidationError):
        SentimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        513,
    ],
)
def test_finbert_max_tokens_rejects_out_of_range(
    value,
):
    raw = valid_sentiment_config()
    raw["finbert"]["max_tokens"] = value

    with pytest.raises(ValidationError):
        SentimentConfig.model_validate(raw)


def test_finbert_max_tokens_rejects_boolean():
    raw = valid_sentiment_config()
    raw["finbert"]["max_tokens"] = True

    with pytest.raises(ValidationError):
        SentimentConfig.model_validate(raw)


def test_sentiment_schema_version_cannot_be_blank():
    raw = valid_sentiment_config()
    raw["schema_version"] = " "

    with pytest.raises(ValidationError):
        SentimentConfig.model_validate(raw)


def test_primary_and_baseline_scorers_must_differ():
    raw = valid_sentiment_config()
    raw["baseline_scorer"] = "finbert"

    with pytest.raises(ValidationError):
        SentimentConfig.model_validate(raw)


def test_pipeline_config_loads_sentiment_section():
    config = load_pipeline_config(
        Path("config/pipeline.yaml")
    )

    assert config.sentiment is not None
    assert (
        config.sentiment.primary_scorer
        is SentimentScorerName.FINBERT
    )
    assert (
        config.sentiment.baseline_scorer
        is SentimentScorerName.VADER
    )
    assert (
        config.sentiment.input_text_strategy
        is SentimentInputTextStrategy.TITLE_AND_SNIPPET
    )
    assert (
        config.sentiment.finbert.model_name
        == "ProsusAI/finbert"
    )
    assert config.sentiment.finbert.max_tokens == 512