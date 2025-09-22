"""Command line entry points for the AutoCLV toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from customer_base_audit.foundation import CustomerDataMartBuilder, PeriodGranularity


def _load_transactions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):  # pragma: no cover - defensive
        raise ValueError("Expected a list of transactions in the input file")
    return [dict(item) for item in payload]


def build_data_mart_cli(argv: list[str] | None = None) -> int:
    """Build order and customer-period aggregations from a JSON file."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to JSON file with raw transactions")
    parser.add_argument(
        "--period",
        dest="periods",
        action="append",
        choices=[item.value for item in PeriodGranularity],
        help="Period granularities to compute (defaults to quarter and year).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for writing the aggregated data mart as JSON.",
    )

    args = parser.parse_args(argv)
    periods = (
        tuple(PeriodGranularity(period) for period in args.periods)
        if args.periods
        else None
    )
    builder = CustomerDataMartBuilder(period_granularities=periods)
    transactions = _load_transactions(args.input)
    mart = builder.build(transactions)

    if args.output:
        payload = mart.as_dict()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    else:  # stdout fallback enables piping in shell usage.
        json.dump(mart.as_dict(), fp=sys.stdout, indent=2, sort_keys=True)
        print()

    return 0


def main() -> None:
    raise SystemExit(build_data_mart_cli())


if __name__ == "__main__":  # pragma: no cover
    main()
