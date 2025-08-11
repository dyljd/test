#!/usr/bin/env python3
"""CLI for computing expected order quantity and occurrence rate.

This script reads a CSV file containing order history with columns
``date``, ``product_id`` and ``quantity``.  For each product it computes:

* Average quantity per order row within a given time window.
* Occurrence rate defined as ``days with orders / total days`` in the
  selected window.

The results are printed to stdout and written to a CSV file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute expected order quantity and occurrence rate",
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument(
        "--output",
        help="Path for output CSV (default: <input_stem>_occurrence_results.csv)",
    )
    parser.add_argument(
        "--window",
        default="30d",
        help="Relative window size such as 30d,60d,90d (used if --start/--end not set)",
    )
    parser.add_argument(
        "--anchor",
        default="data_max",
        choices=["data_max", "today"],
        help="Anchor date for relative window",
    )
    parser.add_argument("--start", help="Explicit window start YYYY-MM-DD")
    parser.add_argument("--end", help="Explicit window end YYYY-MM-DD")

    args = parser.parse_args()
    if bool(args.start) ^ bool(args.end):
        parser.error("--start and --end must be provided together")
    return args


def _parse_window(window_str: str) -> int:
    """Parse window string like ``"30d"`` into integer days."""
    if not window_str.endswith("d"):
        raise ValueError("window must be specified like '30d'")
    try:
        return int(window_str[:-1])
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError("window must start with an integer") from exc


def determine_window(args: argparse.Namespace, df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Determine start and end dates for the analysis window.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    df:
        Dataframe of orders with a ``date`` column parsed to ``datetime``.

    Returns
    -------
    Tuple[pd.Timestamp, pd.Timestamp]
        Start and end date for the window (inclusive).
    """
    if args.start and args.end:
        start = pd.to_datetime(args.start, format="%Y-%m-%d", errors="raise")
        end = pd.to_datetime(args.end, format="%Y-%m-%d", errors="raise")
    else:
        days = _parse_window(args.window)
        if args.anchor == "data_max":
            anchor_date = df["date"].max()
        else:
            anchor_date = pd.Timestamp.today().normalize()
        end = anchor_date
        start = anchor_date - pd.Timedelta(days=days - 1)
    if start > end:
        raise ValueError("start date is after end date")
    return start.normalize(), end.normalize()


def compute_run_rate(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Compute run-rate statistics for each product within the window.

    Parameters
    ----------
    df:
        Dataframe of orders with columns ``date``, ``product_id``, ``quantity``.
    start:
        Window start date (inclusive).
    end:
        Window end date (inclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``product_id``, ``avg_quantity``,
        ``occurrence_rate``, ``days_with_orders``, ``total_days``,
        ``total_quantity`` and ``num_orders`` sorted as required.
    """
    mask = (df["date"] >= start) & (df["date"] <= end)
    window_df = df.loc[mask].copy()
    total_days = (end - start).days + 1

    if window_df.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "avg_quantity",
                "occurrence_rate",
                "days_with_orders",
                "total_days",
                "total_quantity",
                "num_orders",
            ]
        )

    window_df["date"] = window_df["date"].dt.normalize()

    grouped = window_df.groupby("product_id")
    avg_quantity = grouped["quantity"].mean()
    total_quantity = grouped["quantity"].sum()
    num_orders = grouped["quantity"].count()

    daily = window_df.groupby(["product_id", "date"])["quantity"].sum()
    days_with_orders = daily[daily > 0].groupby("product_id").size()

    result = pd.DataFrame({
        "avg_quantity": avg_quantity,
        "total_quantity": total_quantity,
        "num_orders": num_orders,
        "days_with_orders": days_with_orders,
    }).fillna({"days_with_orders": 0})
    result["days_with_orders"] = result["days_with_orders"].astype(int)
    result["occurrence_rate"] = result["days_with_orders"] / total_days
    result["total_days"] = total_days
    result = result.reset_index()
    result = result[
        [
            "product_id",
            "avg_quantity",
            "occurrence_rate",
            "days_with_orders",
            "total_days",
            "total_quantity",
            "num_orders",
        ]
    ]
    result = result.sort_values(
        ["occurrence_rate", "avg_quantity"], ascending=[False, False]
    )
    return result


def main() -> None:
    """Entry point for the CLI."""
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_occurrence_results.csv")
    )

    df = pd.read_csv(input_path)
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except Exception as exc:  # pragma: no cover - input validation
        raise ValueError(f"Failed to parse date column: {exc}") from exc

    df["product_id"] = df["product_id"].astype(str)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="raise")

    start, end = determine_window(args, df)
    total_days = (end - start).days + 1
    result = compute_run_rate(df, start, end)

    print(
        f"Analyzed window: {start.date()} to {end.date()} ({total_days} days)"
    )
    if result.empty:
        print("No data in the selected window.")
    else:
        print(result.to_string(index=False))
    result.to_csv(output_path, index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
