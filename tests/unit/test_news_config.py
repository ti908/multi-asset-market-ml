"""Tests for Step 4 news configuration contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from market_prediction.config import (
    ConfigLoadError,
    load_pipeline_config,
    load_project_config,
)


@pytest.fixture
def payload() -> dict:
    """Return the current pipeline configuration as a mapping."""

    with open(
        "config/pipeline.yaml",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)

def test_all_project_assets_require_news_queries(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    del config_payload["news"]["queries"]["EURUSD"]

    pipeline_path = write_config(
        tmp_path,
        config_payload,
    )

    with pytest.raises(
        ConfigLoadError,
        match="missing news queries",
    ):
        load_project_config(
            "config/assets.yaml",
            pipeline_path,
        )


def test_unknown_news_asset_rejected(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["queries"]["UNKNOWN"] = [
        {
            "query_id": "unknown_asset",
            "strict_terms": [
                "unknown",
            ],
            "context_terms": [],
        }
    ]

    pipeline_path = write_config(
        tmp_path,
        config_payload,
    )

    with pytest.raises(
        ConfigLoadError,
        match="unknown assets",
    ):
        load_project_config(
            "config/assets.yaml",
            pipeline_path,
        )

def write_config(
    tmp_path,
    payload: dict,
):
    """Write a temporary pipeline configuration."""

    path = tmp_path / "pipeline.yaml"

    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return path


def news_config() -> dict:
    """Return one valid news configuration."""

    return {
        "providers": {
            "gdelt": {
                "enabled": True,
                "base_url": (
                    "https://data.gdeltproject.org/"
                    "gdeltv2"
                ),
                "request_delay_seconds": 1.0,
            },
            "marketaux": {
                "enabled": True,
                "base_url": (
                    "https://api.marketaux.com/v1"
                ),
                "api_key_env": "MARKETAUX_API_TOKEN",
                "request_delay_seconds": 1.0,
            },
        },
        "queries": {
            "AAPL": [
                {
                    "query_id": "aapl_company",
                    "strict_terms": [
                        "AAPL",
                        "Apple Inc",
                    ],
                    "context_terms": [],
                }
            ],
            "MSFT": [
                {
                    "query_id": "msft_company",
                    "strict_terms": [
                        "MSFT",
                        "Microsoft",
                        "Microsoft Corporation",
                    ],
                    "context_terms": [],
                }
            ],
            "BTC_USD": [
                {
                    "query_id": "btc_asset",
                    "strict_terms": [
                        "Bitcoin",
                    ],
                    "context_terms": [
                        "BTC",
                        "crypto",
                        "cryptocurrency",
                    ],
                }
            ],
            "EURUSD": [
                {
                    "query_id": "eurusd_pair",
                    "strict_terms": [
                        "EUR/USD",
                        "EURUSD",
                    ],
                    "context_terms": [
                        "euro",
                        "dollar",
                        "USD",
                        "forex",
                        "currency",
                    ],
                }
            ],
            "GOLD": [
                {
                    "query_id": "gold_market",
                    "strict_terms": [
                        "gold price",
                        "gold prices",
                        "bullion",
                        "XAUUSD",
                    ],
                    "context_terms": [
                        "gold",
                        "XAU",
                        "commodity",
                        "futures",
                    ],
                }
            ],
        },
        "availability": {
            "equity": {
                "strategy": "session_cutoff",
            },
            "crypto": {
                "strategy": "utc_cutoff",
            },
            "forex": {
                "strategy": "forex_24_5_cutoff",
            },
            "commodity": {
                "strategy": "session_cutoff",
            },
        },
        "validation": {
            "allowed_languages": [
                "en",
            ],
            "require_title": True,
            "require_canonical_url": True,
            "max_future_seconds": 0,
        },
    }


def add_news(
    payload: dict,
) -> dict:
    """Return payload with valid news configuration."""

    result = deepcopy(payload)
    result["news"] = news_config()

    return result


def test_valid_news_config_loads(
    payload,
    tmp_path,
) -> None:
    config = load_pipeline_config(
        write_config(
            tmp_path,
            add_news(payload),
        )
    )

    assert config.news.providers.gdelt.enabled is True

    assert (
        config.news.providers.marketaux.api_key_env
        == "MARKETAUX_API_TOKEN"
    )

    assert (
        config.news.queries["AAPL"][0].query_id
        == "aapl_company"
    )

    assert (
        config.news.availability.forex.strategy
        == "forex_24_5_cutoff"
    )


def test_news_config_rejects_unknown_fields(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["unexpected"] = True

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )


def test_news_provider_rejects_negative_delay(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["providers"]["gdelt"][
        "request_delay_seconds"
    ] = -1

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )


def test_marketaux_requires_api_key_env(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["providers"]["marketaux"][
        "api_key_env"
    ] = ""

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )


def test_query_requires_query_id(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["queries"]["AAPL"][0][
        "query_id"
    ] = ""

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )


def test_query_requires_strict_terms(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["queries"]["AAPL"][0][
        "strict_terms"
    ] = []

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )


def test_query_rejects_empty_term(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["queries"]["AAPL"][0][
        "strict_terms"
    ] = [
        "AAPL",
        "",
    ]

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )


def test_query_ids_must_be_globally_unique(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["queries"]["MSFT"][0][
        "query_id"
    ] = "aapl_company"

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )



    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )


@pytest.mark.parametrize(
    "asset_class",
    [
        "equity",
        "crypto",
        "forex",
        "commodity",
    ],
)
def test_every_asset_class_requires_availability_rule(
    payload,
    tmp_path,
    asset_class: str,
) -> None:
    config_payload = add_news(payload)

    del config_payload["news"]["availability"][
        asset_class
    ]

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )


def test_allowed_languages_cannot_be_empty(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["validation"][
        "allowed_languages"
    ] = []

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )


def test_future_tolerance_cannot_be_negative(
    payload,
    tmp_path,
) -> None:
    config_payload = add_news(payload)

    config_payload["news"]["validation"][
        "max_future_seconds"
    ] = -1

    with pytest.raises(ConfigLoadError):
        load_pipeline_config(
            write_config(
                tmp_path,
                config_payload,
            )
        )