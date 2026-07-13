#!/usr/bin/env python3
"""Run API-Bank level-1/2 API-call evaluation with Team-ACE/ToolACE-8B."""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import os
import random
import re
import sys
import time
import types
from pathlib import Path
from typing import Any


TOOLACE_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.
{functions}
"""


API_BANK_USER_PROMPT = """Based on the API descriptions and conversation history, generate the single next API request that the AI should call.
Return only one call in this format:
[ApiName(key1='value1', key2='value2')]

Use exact API names and parameter names from the descriptions. Use values grounded in the conversation and previous API responses.
Call only the next chronological API request. Do not batch later API calls.
If a useful API requires a token and no previous API response in the conversation contains that token, call GetUserToken first using the user's username/password. Never invent placeholder tokens such as user_token.
If a previous API response already contains a token, copy that exact token into later token-gated API calls.
This year is 2023.

Conversation history:
{history}
"""


def install_optional_dependency_stubs() -> None:
    """Let API-Bank's API-call evaluator import without optional dialog/search deps."""
    if "utils" not in sys.modules:
        utils_mod = types.ModuleType("utils")

        class _UnusedWrapper:  # pragma: no cover - generation uses local HF model, not API wrappers.
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("API-Bank wrapper utilities are unavailable in this eval environment")

        utils_mod.ChatGPTWrapper = _UnusedWrapper
        utils_mod.DavinciWrapper = _UnusedWrapper
        utils_mod.GPT4Wrapper = _UnusedWrapper
        sys.modules["utils"] = utils_mod

    if "rouge" not in sys.modules:
        rouge_mod = types.ModuleType("rouge")

        class Rouge:  # pragma: no cover - only used for dialog scoring, not API-call scoring.
            def get_scores(self, *_args: Any, **_kwargs: Any) -> list[dict[str, dict[str, float]]]:
                return [{"rouge-l": {"f": 0.0}}]

        rouge_mod.Rouge = Rouge
        sys.modules["rouge"] = rouge_mod

    if "googletrans" not in sys.modules:
        try:
            import googletrans  # noqa: F401
        except Exception:
            googletrans_mod = types.ModuleType("googletrans")

            class Translator:  # pragma: no cover - only present so unrelated Translate API imports.
                def translate(self, text: str, dest: str = "en", **_kwargs: Any) -> Any:
                    return types.SimpleNamespace(text=text, dest=dest)

            googletrans_mod.Translator = Translator
            sys.modules["googletrans"] = googletrans_mod

    if "sentence_transformers" not in sys.modules and importlib.util.find_spec("sentence_transformers") is None:
        st_mod = types.ModuleType("sentence_transformers")

        class SentenceTransformer:  # pragma: no cover - level-1 given-desc does not instantiate this.
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("sentence_transformers is unavailable in this eval environment")

        class _Util:
            @staticmethod
            def cos_sim(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("sentence_transformers is unavailable in this eval environment")

        st_mod.SentenceTransformer = SentenceTransformer
        st_mod.util = _Util()
        sys.modules["sentence_transformers"] = st_mod


def py_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [py_value(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [py_value(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {py_value(k): py_value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = py_value(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.Name):
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        return node.id
    return ast.unparse(node) if hasattr(ast, "unparse") else ""


def func_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = func_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def call_from_ast(node: ast.AST) -> dict[str, Any] | None:
    if not isinstance(node, ast.Call):
        return None
    name = func_name(node.func)
    if not name:
        return None
    return {"name": name, "arguments": {kw.arg: py_value(kw.value) for kw in node.keywords if kw.arg}}


def parse_first_call(text: str) -> dict[str, Any] | None:
    match = re.search(r"\[[\s\S]*\]", text.strip())
    if not match:
        return None
    try:
        tree = ast.parse(match.group(0), mode="eval")
    except SyntaxError:
        return None
    root = tree.body
    nodes = root.elts if isinstance(root, ast.List) else [root]
    for node in nodes:
        call = call_from_ast(node)
        if call:
            return call
    return None


def quote_arg(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def call_to_apibank_text(call: dict[str, Any] | None, raw_text: str) -> str:
    if not call:
        return raw_text.strip()
    args = ", ".join(f"{key}={quote_arg(value)}" for key, value in call["arguments"].items())
    return f"[{call['name']}({args})]"


def role_line(item: dict[str, Any]) -> str:
    role = item["role"]
    if role == "User":
        return f"User: {item['text']}"
    if role == "AI":
        return f"AI: {item['text']}"
    if role == "API":
        args = ", ".join(f"{key}={quote_arg(value)}" for key, value in item.get("param_dict", {}).items())
        output = item.get("result", {}).get("output")
        exception = item.get("result", {}).get("exception")
        response = {"output": output, "exception": exception}
        return f"API: [{item['api_name']}({args})] Response: {json.dumps(response, ensure_ascii=False)}"
    raise ValueError(f"Invalid API-Bank role: {role}")


def toolace_tools(api_description_blob: str) -> list[dict[str, Any]]:
    tools = []
    for line in api_description_blob.splitlines():
        line = line.strip()
        if not line:
            continue
        info = json.loads(line)
        tools.append(
            {
                "name": info.get("name") or info.get("apiCode"),
                "description": info.get("description", ""),
                "arguments": info.get("input_parameters") or info.get("parameters", {}),
            }
        )
    return tools


def api_description_from_info(info: dict[str, Any]) -> str:
    return json.dumps(
        {
            "name": info.get("name") or info.get("apiCode"),
            "description": info.get("description", ""),
            "input_parameters": info.get("input_parameters") or info.get("parameters", {}),
            "output_parameters": info.get("output_parameters") or info.get("response", {}),
        },
        ensure_ascii=False,
    )


def api_descriptions_for_history(evaluator: Any, chat_history: list[dict[str, Any]], tool_search_enabled: bool) -> str:
    if not tool_search_enabled:
        return "\n".join(evaluator.get_api_description(api_name) for api_name in evaluator.dataset[0].apis)
    descriptions = [evaluator.get_api_description("ToolSearcher")]
    seen = {"ToolSearcher"}
    for item in chat_history:
        if item.get("role") != "API" or item.get("api_name") != "ToolSearcher":
            continue
        output = item.get("result", {}).get("output")
        results = output if isinstance(output, list) else [output]
        for api_info in results:
            if not isinstance(api_info, dict):
                continue
            name = api_info.get("name") or api_info.get("apiCode")
            if name and name not in seen:
                descriptions.append(api_description_from_info(api_info))
                seen.add(name)
    return "\n".join(descriptions)


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    model.eval()
    return model, tokenizer


def generate_one(model: Any, tokenizer: Any, api_descriptions: str, chat_history: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    history = "\n".join(role_line(item) for item in chat_history)
    user_prompt = API_BANK_USER_PROMPT.format(history=history)
    messages = [
        {"role": "system", "content": TOOLACE_SYSTEM_PROMPT.format(functions=json.dumps(toolace_tools(api_descriptions), ensure_ascii=False))},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    input_len = inputs.shape[-1]
    start = time.time()
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        generated = model.generate(inputs, **generate_kwargs)
    new_tokens = generated[0][input_len:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    call = parse_first_call(raw_text)
    return {
        "raw_text": raw_text,
        "pred": call_to_apibank_text(call, raw_text),
        "latency_ms": round((time.time() - start) * 1000, 3),
        "generated_tokens": int(new_tokens.numel()),
    }


def iter_samples(data_dir: Path, sample_cls: Any, evaluator_cls: Any, limit: int = 0, sample: int = 0, seed: int = 0) -> list[tuple[str, int, Any, Any]]:
    rows = []
    for file_name in sorted(path.name for path in data_dir.glob("*.jsonl")):
        history = [json.loads(line) for line in (data_dir / file_name).read_text(encoding="utf-8").splitlines() if line.strip()]
        evaluator = evaluator_cls(sample_cls.from_chat_history(history))
        for sample_id in evaluator.get_all_sample_ids():
            sample_obj = evaluator.dataset[sample_id]
            if sample_obj.ground_truth["role"] == "API":
                rows.append((file_name, sample_id, evaluator, sample_obj))
    if sample:
        rows = random.Random(seed).sample(rows, min(sample, len(rows)))
    if limit:
        rows = rows[:limit]
    return rows


def score_predictions(rows: list[tuple[str, int, Any, Any]], pred_map: dict[tuple[str, int], dict[str, Any]], paper_ability: str = "Call") -> dict[str, Any]:
    errors: dict[str, int] = {
        "missing_prediction": 0,
        "no_api_call": 0,
        "api_name_mismatch": 0,
        "exception": 0,
        "key_error": 0,
        "incorrect_result": 0,
        "parse_or_eval_error": 0,
    }
    correct = 0
    by_level = {
        "level_1": {"total": 0, "correct": 0},
        "level_2": {"total": 0, "correct": 0},
        "other": {"total": 0, "correct": 0},
    }
    details = []
    get_api_call = sys.modules["evaluator_by_json"].get_api_call
    for file_name, sample_id, evaluator, sample_obj in rows:
        if "-level-1-" in file_name:
            level_key = "level_1"
        elif "-level-2-" in file_name:
            level_key = "level_2"
        else:
            level_key = "other"
        by_level[level_key]["total"] += 1
        pred = pred_map.get((file_name, sample_id))
        if not pred:
            errors["missing_prediction"] += 1
            continue
        api_call = get_api_call(pred["pred"])
        if not api_call:
            errors["no_api_call"] += 1
            details.append({"file": file_name, "id": sample_id, "error": "no_api_call", "pred": pred["pred"], "expected": sample_obj.ground_truth})
            continue
        original_ground_truth = copy.deepcopy(sample_obj.ground_truth)
        try:
            is_correct, result = evaluator.evaluate(sample_id, api_call)
        except Exception as exc:  # Keep the sweep going while preserving actionable diagnostics.
            sample_obj.ground_truth = original_ground_truth
            errors["parse_or_eval_error"] += 1
            details.append({"file": file_name, "id": sample_id, "error": f"{type(exc).__name__}: {exc}", "pred": pred["pred"], "expected": sample_obj.ground_truth})
            continue
        finally:
            sample_obj.ground_truth = original_ground_truth
        if is_correct:
            correct += 1
            by_level[level_key]["correct"] += 1
            continue
        if isinstance(result, str) and result.startswith("API Name Mismatch"):
            key = "api_name_mismatch"
        elif isinstance(result, str) and result.startswith("KeyError"):
            key = "key_error"
        elif isinstance(result, dict) and result.get("exception"):
            key = "exception"
        else:
            key = "incorrect_result"
        errors[key] += 1
        details.append({"file": file_name, "id": sample_id, "error": key, "pred": pred["pred"], "raw_text": pred.get("raw_text", ""), "result": result, "expected": sample_obj.ground_truth})
    total = len(rows)
    for group in by_level.values():
        group["accuracy"] = round(group["correct"] / group["total"], 6) if group["total"] else 0.0
    return {
        "paper_metric": "API call correctness: correct executable API predictions divided by total API prediction samples.",
        "paper_ability": paper_ability,
        "response_rouge_l": None,
        "response_rouge_l_note": "Not computed in this API-call-only ToolACE run. The API-Bank paper reports response quality separately with ROUGE-L.",
        "total_api_calls": total,
        "correct_api_calls": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "by_filename_level": by_level,
        "errors": errors,
        "details": details[: args.max_error_details],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--data-dir", default="lv1-lv2-samples/level-1-given-desc")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--base", default="Team-ACE/ToolACE-8B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-error-details", type=int, default=80)
    parsed = parser.parse_args()

    global args
    args = parsed

    api_bank_root = Path(args.api_bank_root).resolve()
    os.chdir(api_bank_root)
    sys.path.insert(0, str(api_bank_root))
    install_optional_dependency_stubs()
    from evaluator_by_json import Evaluator, Sample

    data_dir = api_bank_root / args.data_dir
    tool_search_enabled = not data_dir.name.endswith("given-desc")
    rows = iter_samples(data_dir, Sample, Evaluator, limit=args.limit, sample=args.sample, seed=args.seed)
    model, tokenizer = load_model(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_map: dict[tuple[str, int], dict[str, Any]] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for n, (file_name, sample_id, evaluator, sample_obj) in enumerate(rows, start=1):
            _, chat_history = evaluator.get_model_input(sample_id)
            api_descriptions = api_descriptions_for_history(evaluator, chat_history, tool_search_enabled)
            try:
                result = generate_one(model, tokenizer, api_descriptions, chat_history, args)
                error = ""
            except Exception as exc:
                result = {"raw_text": "", "pred": "", "latency_ms": 0.0, "generated_tokens": 0}
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "file": file_name,
                "id": sample_id,
                "pred": result["pred"],
                "raw_text": result["raw_text"],
                "expected_output": f"[{sample_obj.ground_truth['api_name']}(...)]",
                "expected": sample_obj.ground_truth,
                "latency_ms": result["latency_ms"],
                "generated_tokens": result["generated_tokens"],
                "error": error,
            }
            pred_map[(file_name, sample_id)] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"n": n, "total": len(rows), "file": file_name, "id": sample_id, "pred": row["pred"], "error": error}, ensure_ascii=False), flush=True)

    paper_ability = "Retrieve+Call (ToolSearcher; API-Bank lv1-lv2-samples/level-2-toolsearcher)" if tool_search_enabled else "Call (known API descriptions; API-Bank lv1-lv2-samples/level-1-given-desc)"
    score = score_predictions(rows, pred_map, paper_ability=paper_ability)
    score_path = Path(args.score_output)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: score[key] for key in ("paper_metric", "paper_ability", "response_rouge_l", "total_api_calls", "correct_api_calls", "accuracy", "by_filename_level", "errors")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
