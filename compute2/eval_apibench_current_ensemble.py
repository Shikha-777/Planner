#!/usr/bin/env python3
"""Evaluate the current GPT-OSS planner/APIGen/recovery stack on Gorilla APIBench.

This runner avoids using per-row ``api_data`` as prompt context.  It retrieves
candidate API docs from the public provider catalog files and scores generated
API calls against the held-out ``api_call`` field.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ID = "gorilla-llm/APIBench"
HF_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
DEFAULT_PROJ = "/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss"
DEFAULT_APIGEN = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-xlam60k-lora-2xa40-noGC1024"
DEFAULT_PLANNER = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-taskdecomp-lora"
DEFAULT_RECOVERY = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora"

PROVIDER_FILES = {
    "huggingface": ("huggingface_eval.json", "huggingface_api.jsonl"),
    "torchhub": ("torchhub_eval.json", "torchhub_api.jsonl"),
    "tensorflow": ("tensorflow_eval.json", "tensorflowhub_api.jsonl"),
}

JSON_CALL_RE = re.compile(r"\{[\s\S]*?\}")
API_CALL_FIELD_RE = re.compile(r'"api_call"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
INSTRUCTION_RE = re.compile(r"###Instruction:\s*(.*?)(?:\n###Output:|$)", re.DOTALL)
STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "are",
    "as",
    "be",
    "by",
    "can",
    "for",
    "from",
    "get",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "model",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "tool",
    "use",
    "used",
    "using",
    "what",
    "where",
    "with",
}


def download_if_missing(catalog_dir: Path, filename: str) -> Path:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f"{HF_BASE}/{filename}"
    print(f"[apibench] downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_jsonlish(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_instruction(row: dict[str, Any]) -> str:
    code = str(row.get("code") or "")
    match = INSTRUCTION_RE.search(code)
    if match:
        return " ".join(match.group(1).split())
    return code.strip()


def tokens(text: Any) -> list[str]:
    return [
        tok
        for tok in re.findall(r"[a-zA-Z0-9_./:-]+", str(text).lower())
        if len(tok) > 1 and tok not in STOPWORDS
    ]


def compact_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": doc.get("domain"),
        "framework": doc.get("framework"),
        "functionality": doc.get("functionality"),
        "api_name": doc.get("api_name"),
        "api_call": doc.get("api_call"),
        "api_arguments": doc.get("api_arguments"),
        "description": doc.get("description"),
    }


def retrieve_docs(query: str, docs: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    query_terms = tokens(query)
    query_counts = Counter(query_terms)
    if not query_terms:
        return [compact_doc(doc) for doc in docs[:k]]

    doc_tokens: list[list[str]] = []
    doc_freq: Counter[str] = Counter()
    for doc in docs:
        hay = " ".join(
            str(doc.get(key) or "")
            for key in ("domain", "framework", "functionality", "api_name", "api_call", "description")
        )
        toks = tokens(hay)
        doc_tokens.append(toks)
        doc_freq.update(set(toks))

    total_docs = max(1, len(docs))
    avg_len = sum(len(toks) for toks in doc_tokens) / total_docs if doc_tokens else 1.0
    k1 = 1.5
    b = 0.75
    scored = []
    for idx, (doc, toks) in enumerate(zip(docs, doc_tokens)):
        doc_counts = Counter(toks)
        doc_len = max(1, len(toks))
        score = 0.0
        for tok, qtf in query_counts.items():
            tf = doc_counts.get(tok, 0)
            if tf <= 0:
                continue
            idf = math.log(1.0 + (total_docs - doc_freq[tok] + 0.5) / (doc_freq[tok] + 0.5))
            denom = tf + k1 * (1.0 - b + b * doc_len / max(avg_len, 1e-6))
            score += idf * ((tf * (k1 + 1.0)) / denom) * min(qtf, 3)
        scored.append((score, -idx, doc))
    scored.sort(reverse=True)
    return [compact_doc(doc) for score, _, doc in scored[:k] if score > 0] or [compact_doc(doc) for doc in docs[:k]]


def short_json(value: Any, limit: int = 20000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def extract_api_call(text: str) -> str:
    raw = text.strip()
    for obj_text in JSON_CALL_RE.findall(raw):
        obj = maybe_json(obj_text)
        if isinstance(obj, dict):
            for key in ("api_call", "call", "code"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    match = API_CALL_FIELD_RE.search(raw)
    if match:
        try:
            return json.loads('"' + match.group(1) + '"').strip()
        except Exception:
            return match.group(1).strip()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("```", "###", "<<<")):
            continue
        if "(" in line and ")" in line:
            return re.sub(r"^(api_call|call)\s*[:=]\s*", "", line, flags=re.IGNORECASE).strip()
    return raw


def function_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = function_path(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return function_path(node)
        return ast.unparse(node) if hasattr(ast, "unparse") else ""


def parse_first_call(text: str) -> dict[str, Any] | None:
    call_text = extract_api_call(text)
    candidates = [call_text]
    if "=" in call_text and not call_text.lstrip().startswith(("dict(", "{")):
        candidates.append(call_text.split("=", 1)[1].strip())
    for candidate in candidates:
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        if not calls:
            continue
        call = calls[0]
        return {
            "function": function_path(call.func),
            "args": [literal_value(arg) for arg in call.args],
            "kwargs": {kw.arg: literal_value(kw.value) for kw in call.keywords if kw.arg},
            "text": call_text,
        }
    return None


def norm_scalar(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).strip().strip("\"'").lower())


def call_match(pred: str, gold: str) -> tuple[bool, dict[str, Any]]:
    pred_call = parse_first_call(pred)
    gold_call = parse_first_call(gold)
    details = {"pred": pred_call, "gold": gold_call}
    if not pred_call or not gold_call:
        return norm_scalar(pred) == norm_scalar(gold), details

    pred_fn = norm_scalar(pred_call["function"])
    gold_fn = norm_scalar(gold_call["function"])
    if pred_fn != gold_fn:
        return False, details

    gold_args = [norm_scalar(arg) for arg in gold_call["args"]]
    pred_args = [norm_scalar(arg) for arg in pred_call["args"]]
    gold_kwargs = {key: norm_scalar(value) for key, value in gold_call["kwargs"].items()}
    pred_kwargs = {key: norm_scalar(value) for key, value in pred_call["kwargs"].items()}

    for idx, gold_value in enumerate(gold_args):
        if idx < len(pred_args) and pred_args[idx] == gold_value:
            continue
        if gold_value and gold_value in pred_kwargs.values():
            continue
        return False, details
    for key, gold_value in gold_kwargs.items():
        if pred_kwargs.get(key) == gold_value:
            continue
        if gold_value and gold_value in pred_args:
            continue
        return False, details
    return True, details


class GPTOSSAdapterBank:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        try:
            from transformers import Mxfp4Config
        except ImportError:
            Mxfp4Config = None

        self.torch = torch
        self.executor_adapter = "executor"
        self.planner_adapter = "planner"
        self.recovery_adapter = "recovery"
        model_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": args.device_map}
        if not args.no_mxfp4_dequant and Mxfp4Config is not None:
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        print(f"[apibench] loading executor adapter {args.executor_adapter}", flush=True)
        self.model = AutoPeftModelForCausalLM.from_pretrained(args.executor_adapter, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(args.executor_adapter)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        existing = list(getattr(self.model, "peft_config", {}).keys())
        if existing:
            self.executor_adapter = existing[0]
        print(f"[apibench] executor adapter active name: {self.executor_adapter}", flush=True)
        print(f"[apibench] loading planner adapter {args.planner_adapter}", flush=True)
        self.model.load_adapter(args.planner_adapter, adapter_name=self.planner_adapter, is_trainable=False)
        print(f"[apibench] loading recovery adapter {args.recovery_adapter}", flush=True)
        self.model.load_adapter(args.recovery_adapter, adapter_name=self.recovery_adapter, is_trainable=False)
        self.model.eval()

    def generate(
        self,
        adapter: str,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> dict[str, Any]:
        self.model.set_adapter(adapter)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(next(self.model.parameters()).device)
        input_len = inputs.shape[-1]
        eos_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "eos_token_id": eos_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        start = time.time()
        with self.torch.inference_mode():
            output = self.model.generate(inputs, **kwargs)
        new_tokens = output[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return {
            "raw_text": text,
            "latency_ms": round((time.time() - start) * 1000, 3),
            "generated_tokens": int(new_tokens.numel()),
            "adapter": adapter,
        }

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()


def build_plan(bank: GPTOSSAdapterBank, instruction: str, docs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    system = "You are a private planner for APIBench. Return compact JSON only."
    user = {
        "instruction": instruction,
        "retrieved_api_docs": docs,
        "schema": {
            "need": "string",
            "best_api_doc_index": "integer",
            "rationale": "short private string",
            "api_call": "exact Python API call string",
        },
    }
    result = bank.generate(
        bank.planner_adapter,
        [{"role": "system", "content": system}, {"role": "user", "content": short_json(user, 24000)}],
        max_new_tokens=args.max_plan_tokens,
        temperature=0.0,
        top_p=args.top_p,
    )
    result["parsed"] = maybe_json(extract_api_call(result["raw_text"]))
    return result


def generate_call(
    bank: GPTOSSAdapterBank,
    instruction: str,
    docs: list[dict[str, Any]],
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    system = (
        "You are an APIBench API-call generator. Choose the best API from the retrieved docs. "
        "Return exactly one JSON object: {\"api_call\":\"...\"}. Do not explain. "
        "Use API names, URLs, repositories, model ids, and arguments only from the retrieved docs."
    )
    user = {
        "instruction": instruction,
        "retrieved_api_docs": docs,
        "private_plan": plan.get("parsed") or plan.get("raw_text", ""),
    }
    result = bank.generate(
        bank.executor_adapter,
        [{"role": "system", "content": system}, {"role": "user", "content": short_json(user, 26000)}],
        max_new_tokens=args.max_action_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result["api_call"] = extract_api_call(result["raw_text"])
    result["source"] = "executor"
    return result


def recover_call(
    bank: GPTOSSAdapterBank,
    instruction: str,
    docs: list[dict[str, Any]],
    plan: dict[str, Any],
    executor: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    system = (
        "Repair an APIBench candidate. Return exactly one JSON object: {\"api_call\":\"...\"}. "
        "The API call must be a syntactically valid Python call grounded in the retrieved docs."
    )
    user = {
        "instruction": instruction,
        "retrieved_api_docs": docs,
        "private_plan": plan.get("parsed") or plan.get("raw_text", ""),
        "candidate_raw": executor.get("raw_text", ""),
        "candidate_api_call": executor.get("api_call", ""),
    }
    result = bank.generate(
        bank.recovery_adapter,
        [{"role": "system", "content": system}, {"role": "user", "content": short_json(user, 26000)}],
        max_new_tokens=args.max_recovery_tokens,
        temperature=0.0,
        top_p=args.top_p,
    )
    result["api_call"] = extract_api_call(result["raw_text"])
    result["source"] = "recovery"
    return result


def choose_candidate(executor: dict[str, Any], recovery: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any]:
    valid_doc_calls = {norm_scalar(doc.get("api_call")) for doc in docs}
    for candidate in (executor, recovery):
        parsed = parse_first_call(candidate.get("api_call", ""))
        if parsed and norm_scalar(candidate.get("api_call")) in valid_doc_calls:
            return candidate
    for candidate in (executor, recovery):
        if parse_first_call(candidate.get("api_call", "")):
            return candidate
    return executor


def run(args: argparse.Namespace) -> None:
    catalog_dir = Path(args.catalog_dir)
    output_path = Path(args.output)
    score_path = Path(args.score_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)

    provider_groups: list[list[tuple[str, dict[str, Any], list[dict[str, Any]]]]] = []
    for provider in args.providers.split(","):
        provider = provider.strip()
        if provider not in PROVIDER_FILES:
            raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDER_FILES)}")
        eval_file, api_file = PROVIDER_FILES[provider]
        rows = load_jsonlish(download_if_missing(catalog_dir, eval_file))
        docs = load_jsonlish(download_if_missing(catalog_dir, api_file))
        group = []
        for row in rows:
            group.append((provider, row, docs))
        provider_groups.append(group)

    provider_records: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    if args.no_balanced_sample:
        for group in provider_groups:
            provider_records.extend(group)
    else:
        max_len = max((len(group) for group in provider_groups), default=0)
        for idx in range(max_len):
            for group in provider_groups:
                if idx < len(group):
                    provider_records.append(group[idx])

    records = provider_records[args.offset:]
    if args.limit:
        records = records[: args.limit]
    print(f"[apibench] evaluating {len(records)} records providers={args.providers}", flush=True)

    bank = GPTOSSAdapterBank(args)
    correct = 0
    retrieval_hits = 0
    rows_out = []
    with output_path.open("w", encoding="utf-8") as out:
        for idx, (provider, row, docs_all) in enumerate(records):
            instruction = extract_instruction(row)
            docs = retrieve_docs(instruction, docs_all, args.retrieval_k)
            gold = row.get("api_call", "")
            gold_retrieved = gold in {doc.get("api_call") for doc in docs}
            retrieval_hits += int(gold_retrieved)
            plan = build_plan(bank, instruction, docs, args)
            executor = generate_call(bank, instruction, docs, plan, args)
            recovery = recover_call(bank, instruction, docs, plan, executor, args)
            final = choose_candidate(executor, recovery, docs)
            ok, details = call_match(final.get("api_call", ""), gold)
            correct += int(ok)
            record = {
                "index": args.offset + idx,
                "provider": provider,
                "instruction": instruction,
                "gold_api_call": gold,
                "prediction": final.get("api_call", ""),
                "correct": ok,
                "gold_retrieved": gold_retrieved,
                "source": final.get("source"),
                "retrieved_api_calls": [doc.get("api_call") for doc in docs],
                "executor_raw": executor.get("raw_text", "")[:2000],
                "recovery_raw": recovery.get("raw_text", "")[:2000],
                "match_details": details,
            }
            rows_out.append(record)
            out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            out.flush()
            print(
                f"[apibench] {idx + 1}/{len(records)} provider={provider} correct={ok} "
                f"pred={final.get('api_call', '')[:120]}",
                flush=True,
            )

    bank.unload()
    summary = {
        "benchmark": "gorilla-llm/APIBench",
        "setting": "retrieved_docs_no_gold_api_data",
        "providers": args.providers.split(","),
        "limit": args.limit,
        "offset": args.offset,
        "retrieval_k": args.retrieval_k,
        "correct": correct,
        "total": len(rows_out),
        "accuracy": correct / len(rows_out) if rows_out else 0.0,
        "retrieval_hits": retrieval_hits,
        "retrieval_hit_rate": retrieval_hits / len(rows_out) if rows_out else 0.0,
        "output": str(output_path),
        "errors": [row for row in rows_out if not row["correct"]][: args.max_error_details],
    }
    score_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--providers", default="huggingface,torchhub,tensorflow")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--retrieval-k", type=int, default=12)
    parser.add_argument("--no-balanced-sample", action="store_true")
    parser.add_argument("--executor-adapter", default=DEFAULT_APIGEN)
    parser.add_argument("--planner-adapter", default=DEFAULT_PLANNER)
    parser.add_argument("--recovery-adapter", default=DEFAULT_RECOVERY)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-mxfp4-dequant", action="store_true")
    parser.add_argument("--max-plan-tokens", type=int, default=220)
    parser.add_argument("--max-action-tokens", type=int, default=160)
    parser.add_argument("--max-recovery-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-error-details", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
