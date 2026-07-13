#!/usr/bin/env python3
"""Rescore existing ToolACE API-Bank predictions with the paper API-call metric."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import eval_apibank_toolace_official as adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--data-dir", default="lv1-lv2-samples/level-1-given-desc")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--max-error-details", type=int, default=80)
    args = parser.parse_args()

    adapter.args = args
    api_bank_root = Path(args.api_bank_root).resolve()
    os.chdir(api_bank_root)
    sys.path.insert(0, str(api_bank_root))
    adapter.install_optional_dependency_stubs()
    from evaluator_by_json import Evaluator, Sample

    rows = adapter.iter_samples(api_bank_root / args.data_dir, Sample, Evaluator)
    pred_map = {}
    with Path(args.predictions).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            pred = json.loads(line)
            pred_map[(pred["file"], pred["id"])] = pred
    tool_search_enabled = not Path(args.data_dir).name.endswith("given-desc")
    paper_ability = "Retrieve+Call (ToolSearcher; API-Bank lv1-lv2-samples/level-2-toolsearcher)" if tool_search_enabled else "Call (known API descriptions; API-Bank lv1-lv2-samples/level-1-given-desc)"
    score = adapter.score_predictions(rows, pred_map, paper_ability=paper_ability)
    Path(args.score_output).write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: score[key] for key in ("paper_metric", "paper_ability", "response_rouge_l", "total_api_calls", "correct_api_calls", "accuracy", "by_filename_level", "errors")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
