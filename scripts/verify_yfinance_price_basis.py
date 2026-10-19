"""Manual verification of Yahoo Finance price adjustment behavior.

This script intentionally makes a network request and therefore MUST NOT
be executed as part of the automated test suite.

It examines AAPL around the 4-for-1 stock split effective 2020-08-31 and
compares yfinance history() with auto_adjust=False and auto_adjust=True.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


TICKER = "AAPL"
START = "2020-08-27"
END = "2020-09-03"


def fetch(auto_adjust: bool) -> pd.DataFrame:
    """Fetch the AAPL split window using the requested adjustment mode."""
    return yf.Ticker(TICKER).history(
        start=START,
        end=END,
        interval="1d",
        auto_adjust=auto_adjust,
        actions=True,
        raise_errors=True,
    )


def main() -> None:
    """Print raw and automatically adjusted AAPL observations."""
    unadjusted = fetch(auto_adjust=False)
    adjusted = fetch(auto_adjust=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 6)

    print("\n=== auto_adjust=False ===")
    print(unadjusted)

    print("\n=== auto_adjust=True ===")
    print(adjusted)

    print("\n=== auto_adjust=False selected columns ===")
    columns = [
        column
        for column in [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Dividends",
            "Stock Splits",
        ]
        if column in unadjusted.columns
    ]

    print(unadjusted[columns])

    print("\n=== Close vs Adj Close ===")
    if "Adj Close" in unadjusted.columns:
        comparison = unadjusted[["Close", "Adj Close"]].copy()
        comparison["Adj/Close"] = (
            comparison["Adj Close"] / comparison["Close"]
        )
        print(comparison)
    else:
        print("Adj Close was not returned.")

    print("\n=== Split event ===")
    if "Stock Splits" in unadjusted.columns:
        print(unadjusted.loc[unadjusted["Stock Splits"] != 0, "Stock Splits"])
    else:
        print("Stock Splits column was not returned.")


if __name__ == "__main__":
    main()