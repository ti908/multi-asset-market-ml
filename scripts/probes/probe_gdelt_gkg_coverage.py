"""Bounded Step 4a coverage probe for historical GDELT GKG 2.0.

This investigation script:
- reads GDELT's public master file list;
- selects GKG files for explicitly sampled UTC dates;
- processes each ZIP entirely in memory;
- calculates strict and broad asset-query coverage;
- reports source diversity and duplicate URLs;
- writes nothing to the project cache or filesystem;
- performs no sentiment scoring.

It is NOT production ingestion code.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MASTER_FILE_LIST_URL = (
    "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
)

TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 0.25
MAX_ATTEMPTS = 3

DEFAULT_SAMPLE_DATES = (
    "20150220",
    "20160615",
    "20180615",
    "20200319",
    "20220615",
    "20240615",
    "20260615",
)


@dataclass(frozen=True)
class QueryRule:
    strict_phrases: tuple[str, ...]
    broad_primary: tuple[str, ...]
    broad_context: tuple[str, ...]


QUERY_RULES = {
    "AAPL": QueryRule(
        strict_phrases=(
            "apple inc",
            "apple inc.",
        ),
        broad_primary=(
            "apple",
            "aapl",
        ),
        broad_context=(
            "iphone",
            "ipad",
            "macbook",
            "macintosh",
            "cupertino",
            "technology",
            "stock",
            "shares",
            "nasdaq",
            "company",
        ),
    ),
    "MSFT": QueryRule(
        strict_phrases=(
            "microsoft corporation",
        ),
        broad_primary=(
            "microsoft",
            "msft",
        ),
        broad_context=(
            "windows",
            "azure",
            "xbox",
            "software",
            "technology",
            "stock",
            "shares",
            "nasdaq",
            "company",
        ),
    ),
    "BTC_USD": QueryRule(
        strict_phrases=(
            "bitcoin",
        ),
        broad_primary=(
            "bitcoin",
            "btc",
        ),
        broad_context=(
            "crypto",
            "cryptocurrency",
            "blockchain",
            "digital currency",
            "digital asset",
            "exchange",
            "coin",
        ),
    ),
    "EURUSD": QueryRule(
        strict_phrases=(
            "eur/usd",
            "eurusd",
        ),
        broad_primary=(
            "euro",
        ),
        broad_context=(
            "dollar",
            "usd",
            "forex",
            "currency",
            "exchange rate",
            "foreign exchange",
        ),
    ),
    "GOLD": QueryRule(
        strict_phrases=(
            "gold price",
            "gold prices",
            "gold bullion",
        ),
        broad_primary=(
            "gold",
            "xau",
        ),
        broad_context=(
            "commodity",
            "commodities",
            "futures",
            "bullion",
            "ounce",
            "ounces",
            "precious metal",
            "price",
            "market",
        ),
    ),
}


@dataclass
class MatchStats:
    matched_rows: int = 0
    urls: set[str] = field(default_factory=set)
    sources: Counter[str] = field(
        default_factory=Counter
    )
    intervals: set[str] = field(default_factory=set)
    samples: list[dict[str, str]] = field(
        default_factory=list
    )

    def add(
        self,
        *,
        interval: str,
        record_id: str,
        source: str,
        url: str,
        matched_as: str,
    ) -> None:
        self.matched_rows += 1

        if url:
            self.urls.add(url)

        if source:
            self.sources[source] += 1

        self.intervals.add(interval)

        if len(self.samples) < 3:
            self.samples.append(
                {
                    "record_id": record_id,
                    "source": source,
                    "url": url,
                    "matched_as": matched_as,
                }
            )


def normalize(value: str) -> str:
    value = value.casefold()

    value = re.sub(
        r"[_\-/]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def contains_phrase(
    text: str,
    phrase: str,
) -> bool:
    normalized_phrase = normalize(phrase)

    if not normalized_phrase:
        return False

    # Word-like boundaries reduce accidental substring
    # matches such as "btc" inside an unrelated token.
    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(normalized_phrase)
        + r"(?![a-z0-9])"
    )

    return re.search(
        pattern,
        text,
    ) is not None


def matches_strict(
    text: str,
    rule: QueryRule,
) -> bool:
    return any(
        contains_phrase(text, phrase)
        for phrase in rule.strict_phrases
    )


def matches_broad(
    text: str,
    rule: QueryRule,
) -> bool:
    primary = any(
        contains_phrase(text, phrase)
        for phrase in rule.broad_primary
    )

    context = any(
        contains_phrase(text, phrase)
        for phrase in rule.broad_context
    )

    return primary and context


def fetch_bytes(
    url: str,
) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent":
                "multi-asset-market-ml-step4a-probe/1.0"
        },
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        try:
            with urlopen(
                request,
                timeout=TIMEOUT_SECONDS,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"HTTP {response.status}"
                    )

                return response.read()

        except (
            HTTPError,
            URLError,
            TimeoutError,
        ) as exc:
            last_error = exc

            if attempt == MAX_ATTEMPTS:
                break

            delay = 2 ** (attempt - 1)

            print(
                f"    retry {attempt}/{MAX_ATTEMPTS} "
                f"after {type(exc).__name__}: {exc}"
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Unable to fetch {url}: {last_error}"
    )


def load_master_file_list() -> str:
    print(
        "Loading GDELT master file list..."
    )

    payload = fetch_bytes(
        MASTER_FILE_LIST_URL
    )

    text = payload.decode(
        "utf-8",
        errors="replace",
    )

    print(
        "Master-list rows:",
        len(text.splitlines()),
    )

    return text


def gkg_urls_for_date(
    master_text: str,
    date_prefix: str,
) -> list[str]:
    urls: list[str] = []

    for line in master_text.splitlines():
        parts = line.strip().split()

        if not parts:
            continue

        url = parts[-1]

        if (
            f"/{date_prefix}" in url
            and ".gkg.csv.zip" in url
        ):
            urls.append(url)

    return sorted(set(urls))


def rows_from_gkg_zip(
    payload: bytes,
) -> list[list[str]]:
    with zipfile.ZipFile(
        io.BytesIO(payload)
    ) as archive:
        names = archive.namelist()

        if len(names) != 1:
            raise RuntimeError(
                "Expected one GKG member; "
                f"found {names}"
            )

        with archive.open(
            names[0]
        ) as fh:
            text = io.TextIOWrapper(
                fh,
                encoding="utf-8",
                errors="replace",
            ).read()

    return list(
        csv.reader(
            io.StringIO(text),
            delimiter="\t",
        )
    )


def process_date(
    date_prefix: str,
    urls: list[str],
) -> None:
    print()
    print("=" * 78)
    print("DATE:", date_prefix)
    print("GKG FILES DISCOVERED:", len(urls))

    strict_stats = {
        asset: MatchStats()
        for asset in QUERY_RULES
    }

    broad_stats = {
        asset: MatchStats()
        for asset in QUERY_RULES
    }

    successful_files = 0
    failed_files = 0
    total_rows = 0
    malformed_rows = 0

    for index, url in enumerate(
        urls,
        start=1,
    ):
        filename = url.rsplit(
            "/",
            1,
        )[-1]

        interval = filename[:14]

        print(
            f"[{index:02d}/{len(urls):02d}] "
            f"{interval}",
            end="",
            flush=True,
        )

        try:
            payload = fetch_bytes(url)
            rows = rows_from_gkg_zip(
                payload
            )
        except Exception as exc:
            failed_files += 1

            print(
                " ERROR:",
                f"{type(exc).__name__}: {exc}",
            )

            continue

        successful_files += 1
        total_rows += len(rows)

        print(
            f" rows={len(rows)}"
        )

        for row in rows:
            if len(row) != 27:
                malformed_rows += 1
                continue

            record_id = row[0]
            source = row[3]
            document_identifier = row[4]

            # Step 4a exploratory matching deliberately
            # examines the complete GKG metadata record.
            # Production matching rules are NOT being
            # decided here.
            searchable = normalize(
                "\t".join(row)
            )

            for asset, rule in (
                QUERY_RULES.items()
            ):
                if matches_strict(
                    searchable,
                    rule,
                ):
                    strict_stats[asset].add(
                        interval=interval,
                        record_id=record_id,
                        source=source,
                        url=document_identifier,
                        matched_as="strict",
                    )

                if matches_broad(
                    searchable,
                    rule,
                ):
                    broad_stats[asset].add(
                        interval=interval,
                        record_id=record_id,
                        source=source,
                        url=document_identifier,
                        matched_as="broad",
                    )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    print()
    print("DAY SUMMARY")
    print("Successful files:", successful_files)
    print("Failed files:", failed_files)
    print("Total GKG rows:", total_rows)
    print("Malformed-width rows:", malformed_rows)

    if urls:
        print(
            "Archive-file completeness:",
            f"{successful_files}/{len(urls)}",
        )

    print()
    print(
        "ASSET COVERAGE "
        "(STRICT vs BROAD)"
    )

    for asset in QUERY_RULES:
        strict = strict_stats[asset]
        broad = broad_stats[asset]

        strict_duplicate_rows = max(
            strict.matched_rows
            - len(strict.urls),
            0,
        )

        broad_duplicate_rows = max(
            broad.matched_rows
            - len(broad.urls),
            0,
        )

        print("-" * 78)
        print("ASSET:", asset)

        print(
            "STRICT:",
            f"rows={strict.matched_rows}",
            f"unique_urls={len(strict.urls)}",
            f"intervals={len(strict.intervals)}",
            f"sources={len(strict.sources)}",
            f"duplicate_rows≈{strict_duplicate_rows}",
        )

        print(
            "BROAD:",
            f"rows={broad.matched_rows}",
            f"unique_urls={len(broad.urls)}",
            f"intervals={len(broad.intervals)}",
            f"sources={len(broad.sources)}",
            f"duplicate_rows≈{broad_duplicate_rows}",
        )

        if strict.sources:
            print(
                "STRICT TOP SOURCES:",
                strict.sources.most_common(5),
            )

        if broad.sources:
            print(
                "BROAD TOP SOURCES:",
                broad.sources.most_common(5),
            )

        for sample in strict.samples:
            print(
                "STRICT SAMPLE:",
                sample,
            )

        for sample in broad.samples:
            print(
                "BROAD SAMPLE:",
                sample,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded read-only GDELT GKG "
            "historical coverage probe."
        )
    )

    parser.add_argument(
        "--date",
        action="append",
        dest="dates",
        help=(
            "UTC date in YYYYMMDD format. "
            "May be repeated. Defaults to "
            "the Step 4a sample dates."
        ),
    )

    return parser.parse_args()


def validate_dates(
    dates: list[str],
) -> None:
    for value in dates:
        if not re.fullmatch(
            r"\d{8}",
            value,
        ):
            raise ValueError(
                f"Invalid date {value!r}; "
                "expected YYYYMMDD"
            )


def main() -> None:
    args = parse_args()

    dates = (
        args.dates
        if args.dates
        else list(DEFAULT_SAMPLE_DATES)
    )

    validate_dates(dates)

    print(
        "GDELT GKG 2.0 "
        "BOUNDED COVERAGE PROBE"
    )
    print(
        "Dates:",
        ", ".join(dates),
    )
    print(
        "Writes project/cache data: NO"
    )

    master_text = (
        load_master_file_list()
    )

    for date_prefix in dates:
        urls = gkg_urls_for_date(
            master_text,
            date_prefix,
        )

        if not urls:
            print()
            print("=" * 78)
            print(
                "DATE:",
                date_prefix,
            )
            print(
                "GKG FILES DISCOVERED: 0"
            )
            print(
                "No archive files found "
                "for this sampled date."
            )
            continue

        process_date(
            date_prefix,
            urls,
        )

    print()
    print("=" * 78)
    print("PROBE COMPLETE")
    print(
        "Counts are exploratory metadata "
        "matches, not sentiment features."
    )
    print(
        "No production cache or config "
        "was modified."
    )


if __name__ == "__main__":
    main()