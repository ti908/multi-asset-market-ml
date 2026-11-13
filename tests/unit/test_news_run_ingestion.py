from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from market_prediction.news.run_ingestion import (
    build_news_ingestion,
)
from market_prediction.providers.news.marketaux import (
    MarketauxNewsProvider,
)
from market_prediction.providers.news.runtime import (
    SystemUtcClock,
    UrllibJsonTransport,
)
from market_prediction.news.orchestrator import (
    NewsIngestionRunResult,
)
from market_prediction.news.validation import (
    NewsStatus,
)
from market_prediction.providers.news.base import (
    NewsQuery,
)


def test_build_news_ingestion_constructs_production_dependencies(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "MARKETAUX_API_TOKEN",
        "test-token",
    )

    orchestrator, queries = build_news_ingestion(
        Path("config")
    )

    assert isinstance(
        orchestrator._provider,
        MarketauxNewsProvider,
    )

    assert isinstance(
        orchestrator._provider._transport,
        UrllibJsonTransport,
    )

    assert isinstance(
        orchestrator._provider._clock,
        SystemUtcClock,
    )

    assert tuple(
        query.asset
        for query in queries
    ) == (
        "AAPL",
        "MSFT",
        "BTC_USD",
        "EURUSD",
        "GOLD",
    )

    assert tuple(
        query.query_id
        for query in queries
    ) == (
        "aapl_company",
        "msft_company",
        "btc_asset",
        "eurusd_pair",
        "gold_market",
    )

    assert orchestrator._allowed_assets == {
        "AAPL",
        "MSFT",
        "BTC_USD",
        "EURUSD",
        "GOLD",
    }

    assert orchestrator._allowed_query_ids == {
        "aapl_company",
        "msft_company",
        "btc_asset",
        "eurusd_pair",
        "gold_market",
    }

    assert orchestrator._allowed_languages == {
        "en",
    }

    assert (
        orchestrator._store._storage_dir
        == Path("data/raw/news/canonical")
    )


def test_build_news_ingestion_requires_marketaux_token(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "MARKETAUX_API_TOKEN",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="MARKETAUX_API_TOKEN",
    ):
        build_news_ingestion(
            Path("config")
        )

def test_parser_defaults_to_config_directory() -> None:
    from market_prediction.news.run_ingestion import (
        build_parser,
    )

    args = build_parser().parse_args(
        [
            "--start",
            "2026-11-01T00:00:00Z",
            "--end",
            "2026-11-08T00:00:00Z",
        ]
    )

    assert args.config_dir == Path("config")


def test_main_runs_complete_query_set_once(
    monkeypatch,
    tmp_path,
) -> None:
    import market_prediction.news.run_ingestion as cli

    observed = {
        "calls": 0,
    }

    queries = (
        NewsQuery(
            asset="AAPL",
            query_id="aapl_company",
            strict_terms=("AAPL",),
        ),
        NewsQuery(
            asset="MSFT",
            query_id="msft_company",
            strict_terms=("MSFT",),
        ),
    )

    class FakeOrchestrator:
        def run_queries(
            self,
            *,
            queries,
            start_at_utc,
            end_at_utc,
        ):
            observed["calls"] += 1
            observed["queries"] = queries
            observed["start"] = start_at_utc
            observed["end"] = end_at_utc

            return NewsIngestionRunResult(
                article_count=7,
                validation_status=(
                    NewsStatus.PASSED
                ),
                dataset_id="a" * 64,
            )

    def fake_build_news_ingestion(
        config_dir,
    ):
        observed["config_dir"] = config_dir

        return (
            FakeOrchestrator(),
            queries,
        )

    monkeypatch.setattr(
        cli,
        "build_news_ingestion",
        fake_build_news_ingestion,
    )

    exit_code = cli.main(
        [
            "--config-dir",
            str(tmp_path),
            "--start",
            "2026-11-01T00:00:00Z",
            "--end",
            "2026-11-08T00:00:00Z",
        ]
    )

    assert exit_code == 0

    assert observed["calls"] == 1

    assert observed["config_dir"] == tmp_path

    assert isinstance(
        observed["queries"],
        list,
    )

    assert observed["queries"] == list(
        queries
    )

    assert observed["start"] == datetime(
        2026,
        11,
        1,
        tzinfo=timezone.utc,
    )

    assert observed["end"] == datetime(
        2026,
        11,
        8,
        tzinfo=timezone.utc,
    )


def test_main_prints_success_result(
    monkeypatch,
    capsys,
) -> None:
    import market_prediction.news.run_ingestion as cli

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=("AAPL",),
    )

    class FakeOrchestrator:
        def run_queries(
            self,
            **kwargs,
        ):
            return NewsIngestionRunResult(
                article_count=12,
                validation_status=(
                    NewsStatus.PASSED
                ),
                dataset_id="b" * 64,
            )

    monkeypatch.setattr(
        cli,
        "build_news_ingestion",
        lambda config_dir: (
            FakeOrchestrator(),
            (query,),
        ),
    )

    exit_code = cli.main(
        [
            "--start",
            "2026-11-01T00:00:00Z",
            "--end",
            "2026-11-08T00:00:00Z",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "12" in output
    assert NewsStatus.PASSED.value in output
    assert "b" * 64 in output


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-11-01T00:00:00",
        "2026-11-01T01:00:00+01:00",
        "not-a-timestamp",
    ],
)
def test_main_rejects_invalid_utc_timestamp(
    timestamp,
    capsys,
) -> None:
    import market_prediction.news.run_ingestion as cli

    exit_code = cli.main(
        [
            "--start",
            timestamp,
            "--end",
            "2026-11-08T00:00:00Z",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.err


def test_main_reports_construction_error(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    import market_prediction.news.run_ingestion as cli

    def fail_build(
        config_dir,
    ):
        raise ValueError(
            "MARKETAUX_API_TOKEN is not configured"
        )

    monkeypatch.setattr(
        cli,
        "build_news_ingestion",
        fail_build,
    )

    exit_code = cli.main(
        [
            "--config-dir",
            str(tmp_path),
            "--start",
            "2026-11-01T00:00:00Z",
            "--end",
            "2026-11-08T00:00:00Z",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert (
        "MARKETAUX_API_TOKEN"
        in captured.err
    )


def test_main_reports_orchestration_error(
    monkeypatch,
    capsys,
) -> None:
    import market_prediction.news.run_ingestion as cli

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=("AAPL",),
    )

    class FakeOrchestrator:
        def run_queries(
            self,
            **kwargs,
        ):
            raise ValueError(
                "start_at_utc must be before end_at_utc."
            )

    monkeypatch.setattr(
        cli,
        "build_news_ingestion",
        lambda config_dir: (
            FakeOrchestrator(),
            (query,),
        ),
    )

    exit_code = cli.main(
        [
            "--start",
            "2026-11-08T00:00:00Z",
            "--end",
            "2026-11-01T00:00:00Z",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert (
        "start_at_utc must be before end_at_utc"
        in captured.err
    )

def test_main_reports_news_provider_error(
    monkeypatch,
    capsys,
) -> None:
    import market_prediction.news.run_ingestion as cli

    from market_prediction.providers.news.exceptions import (
        InvalidNewsProviderDataError,
    )

    query = NewsQuery(
        asset="AAPL",
        query_id="aapl_company",
        strict_terms=("AAPL",),
    )

    class FakeOrchestrator:
        def run_queries(
            self,
            **kwargs,
        ):
            raise InvalidNewsProviderDataError(
                "Marketaux returned invalid HTTP status 403"
            )

    monkeypatch.setattr(
        cli,
        "build_news_ingestion",
        lambda config_dir: (
            FakeOrchestrator(),
            (query,),
        ),
    )

    exit_code = cli.main(
        [
            "--start",
            "2026-10-30T00:00:00Z",
            "--end",
            "2026-10-31T00:00:00Z",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2

    assert (
        "Marketaux returned invalid HTTP status 403"
        in captured.err
    )