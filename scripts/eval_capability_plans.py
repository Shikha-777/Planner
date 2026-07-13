#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskdecomp.capability_eval import read_jsonl, score_predictions, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument(
        "--predicted-only",
        action="store_true",
        help="Score only gold cases that have prediction rows. Useful for LIMIT smoke runs.",
    )
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    prediction_rows = read_jsonl(args.predictions)
    result = score_predictions(gold_rows, prediction_rows, predicted_only=args.predicted_only)
    write_json(args.metrics_out, result)
    write_csv(args.csv_out, result["cases"])
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
