"""Read-only Step 4a probe for the historical GDELT GKG 2.0 archive.

Purpose:
- verify that historical GKG 2.0 files are discoverable;
- select one bounded file from the beginning of the archive;
- inspect real GKG records;
- test simple asset-term presence;
- write nothing to the project cache or local disk.

This is investigation code only, not production ingestion code.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import Counter
from urllib.request import Request, urlopen


MASTER_FILE_LIST_URL = (
    "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
)

TARGET_DATE_PREFIX = "20150219"
TIMEOUT_SECONDS = 60

ASSET_TERMS = {
    "AAPL": (
        "apple inc",
        "apple computer",
    ),
    "MSFT": (
        "microsoft",
        "microsoft corporation",
    ),
    "BTC_USD": (
        "bitcoin",
        "btc",
    ),
    "EURUSD": (
        "eur/usd",
        "eurusd",
        "euro dollar",
    ),
    "GOLD": (
        "gold price",
        "gold prices",
        "gold market",
        "gold bullion",
    ),
}


def fetch_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent":
                "multi-asset-market-ml-step4a-probe/1.0"
        },
    )

    with urlopen(
        request,
        timeout=TIMEOUT_SECONDS,
    ) as response:
        print(
            "HTTP:",
            response.status,
            url,
        )
        return response.read()


def find_earliest_gkg_url(
    master_text: str,
) -> str:
    candidates: list[str] = []

    for line in master_text.splitlines():
        parts = line.strip().split()

        if not parts:
            continue

        url = parts[-1]

        if (
            TARGET_DATE_PREFIX in url
            and ".gkg.csv.zip" in url
        ):
            candidates.append(url)

    if not candidates:
        raise RuntimeError(
            "No GKG archive files found for "
            f"{TARGET_DATE_PREFIX}"
        )

    return sorted(candidates)[0]


def decode_gkg_zip(
    payload: bytes,
) -> tuple[str, str]:
    with zipfile.ZipFile(
        io.BytesIO(payload)
    ) as archive:
        names = archive.namelist()

        if len(names) != 1:
            raise RuntimeError(
                "Expected exactly one file in GKG ZIP; "
                f"found {names}"
            )

        member = names[0]

        with archive.open(member) as fh:
            text = io.TextIOWrapper(
                fh,
                encoding="utf-8",
                errors="replace",
            ).read()

    return member, text


def normalize_text(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        value.casefold(),
    )


def main() -> None:
    print("GDELT GKG 2.0 HISTORICAL ARCHIVE PROBE")
    print("=" * 72)

    master_payload = fetch_bytes(
        MASTER_FILE_LIST_URL
    )
    master_text = master_payload.decode(
        "utf-8",
        errors="replace",
    )

    print(
        "MASTER FILE LIST LINES:",
        len(master_text.splitlines()),
    )

    gkg_url = find_earliest_gkg_url(
        master_text
    )

    print(
        "SELECTED GKG FILE:",
        gkg_url,
    )

    archive_payload = fetch_bytes(
        gkg_url
    )

    print(
        "COMPRESSED BYTES:",
        len(archive_payload),
    )

    member, text = decode_gkg_zip(
        archive_payload
    )

    print(
        "ZIP MEMBER:",
        member,
    )

    rows = list(
        csv.reader(
            io.StringIO(text),
            delimiter="\t",
        )
    )

    print(
        "ROWS:",
        len(rows),
    )

    if not rows:
        raise RuntimeError(
            "Historical GKG file contained no rows"
        )

    widths = Counter(
        len(row)
        for row in rows
    )

    print(
        "COLUMN WIDTH COUNTS:",
        dict(sorted(widths.items())),
    )

    first = rows[0]

    print(
        "FIRST RECORD COLUMN COUNT:",
        len(first),
    )

    print("\nFIRST RECORD — SELECTED RAW FIELDS")

    # GKG 2.0 fields begin with:
    # 0 GKGRECORDID
    # 1 V2.1DATE
    # 2 V2SOURCECOLLECTIONIDENTIFIER
    # 3 V2SOURCECOMMONNAME
    # 4 V2DOCUMENTIDENTIFIER
    selected_indices = {
        "record_id": 0,
        "date": 1,
        "source_collection": 2,
        "source_common_name": 3,
        "document_identifier": 4,
    }

    for name, index in selected_indices.items():
        value = (
            first[index]
            if index < len(first)
            else "<missing>"
        )
        print(
            f"{name}: {value}"
        )

    print("\nASSET TERM MATCHES IN THIS SINGLE 15-MINUTE FILE")

    match_counts: dict[str, int] = {}
    sample_matches: dict[
        str,
        list[dict[str, str]]
    ] = {}

    for asset, terms in ASSET_TERMS.items():
        matches = []

        for row in rows:
            searchable = normalize_text(
                "\t".join(row)
            )

            matched_terms = [
                term
                for term in terms
                if term in searchable
            ]

            if not matched_terms:
                continue

            matches.append(
                {
                    "record_id": (
                        row[0]
                        if len(row) > 0
                        else ""
                    ),
                    "date": (
                        row[1]
                        if len(row) > 1
                        else ""
                    ),
                    "source": (
                        row[3]
                        if len(row) > 3
                        else ""
                    ),
                    "document_identifier": (
                        row[4]
                        if len(row) > 4
                        else ""
                    ),
                    "matched_terms": ",".join(
                        matched_terms
                    ),
                }
            )

        match_counts[asset] = len(matches)
        sample_matches[asset] = matches[:3]

    for asset in ASSET_TERMS:
        print(
            f"{asset}: {match_counts[asset]}"
        )

        for sample in sample_matches[asset]:
            print(
                "  SAMPLE:",
                sample,
            )

    print("\nIMPORTANT INTERPRETATION")
    print(
        "These counts represent one historical GKG "
        "update file only."
    )
    print(
        "Zero matches do NOT establish zero daily "
        "coverage for an asset."
    )
    print(
        "Keyword matches are discovery evidence only; "
        "they are not yet relevance-filtered articles."
    )
    print(
        "No data was written to the project cache."
    )


if __name__ == "__main__":
    main()