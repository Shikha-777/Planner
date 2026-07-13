#!/usr/bin/env python3
"""Reselect API-Bank Level-3 from an existing current-ensemble JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import eval_apibank_level3_current_ensemble as l3
import eval_apibank_lv12_current_ensemble as current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--level3-gt", required=True)
    parser.add_argument("--existing-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--max-error-details", type=int, default=80)
    args = parser.parse_args()

    os.chdir(args.api_bank_root)
    l3_args = argparse.Namespace(
        api_bank_root=args.api_bank_root,
        input_json=args.input_json,
        level3_gt=args.level3_gt,
        output=args.output,
        score_output=args.score_output,
        limit=0,
        offset=0,
        chronology_mode="state",
        verifier_model="none",
        max_error_details=args.max_error_details,
    )
    records = l3.load_records(l3_args)
    old_rows = [json.loads(line) for line in Path(args.existing_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != len(old_rows):
        raise RuntimeError(f"record count mismatch: {len(records)} records vs {len(old_rows)} existing rows")

    for record, old in zip(records, old_rows):
        candidates = []
        for item in (old.get("ensemble") or {}).get("candidates") or []:
            text = item.get("raw_text") or item.get("pred") or ""
            result = {
                "raw_text": text,
                "pred": item.get("pred") or text,
                "latency_ms": 0,
                "generated_tokens": 0,
            }
            candidates.append(current.make_candidate(str(item.get("source")), result, record["tools"], item.get("error", "")))
        record["candidates"] = candidates

    l3.select_records(records, l3_args)
    l3.write_and_score(records, l3_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
