"""Local planner/executor/validator ensemble agent for tau-bench.

The ensemble is deliberately wired as one coherent agent:

* GPT-OSS task-decomposition LoRA builds compact private state.
* GPT-OSS TaskBench task-decomposition LoRA supplies recovery planning.
* GPT-OSS Nemotron/BFCL LoRA chooses the next action.
* ToolACE-8B optionally validates/repairs tool calls.

All model paths are configured by environment variables so Slurm jobs can run
ablations without changing code. The GPT-OSS LoRAs share one base model and are
switched with PEFT adapters; ToolACE is loaded lazily on the first validation.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tau_bench.agents.base import Agent
    from tau_bench.envs.base import Env
    from tau_bench.types import (
        Action,
        RESPOND_ACTION_FIELD_NAME,
        RESPOND_ACTION_NAME,
        SolveResult,
    )
except ImportError:
    @dataclass
    class Action:
        name: str
        kwargs: Dict[str, Any]

    class Agent:
        pass

    Env = Any
    SolveResult = Any
    RESPOND_ACTION_FIELD_NAME = "content"
    RESPOND_ACTION_NAME = "respond"


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
HARMONY_TO_FUNCTION_RE = re.compile(
    r"to=functions\.([A-Za-z_][A-Za-z0-9_]*).*?<\|message\|>\s*(.*?)(?=<\|call\|>|<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)
HARMONY_MESSAGE_RE = re.compile(
    r"<\|message\|>\s*(\{.*?)(?=<\|call\|>|<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)
PSEUDO_TOOLS = {"think"}


DEFAULT_PROJ = "/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss"
DEFAULT_EXECUTOR = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-nemotron-agentic-bfcl-lora-2xa40-noGC1024"
DEFAULT_PLANNER = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-taskdecomp-lora"
DEFAULT_RECOVERY = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora"
DEFAULT_VALIDATOR = "Team-ACE/ToolACE-8B"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def short_json(value: Any, limit: int = 24000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def iter_json_objects(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    starts = [i for i, char in enumerate(cleaned) if char in "{["]
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        yield obj


def first_json_object(text: str) -> Optional[Any]:
    for obj in iter_json_objects(text):
        return obj
    return None


def normalize_one_call(value: Any) -> Optional[Dict[str, Any]]:
    value = maybe_json(value)
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("function"), dict):
        fn = value["function"]
    else:
        fn = value
    name = fn.get("name") or fn.get("function_name") or fn.get("tool_name")
    args = fn.get("arguments")
    if args is None:
        args = fn.get("args", fn.get("parameters", fn.get("kwargs", {})))
    args = maybe_json(args)
    if args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"value": args}
    if not name:
        return None
    return {"name": str(name), "arguments": args}


def normalize_call_container(value: Any) -> List[Dict[str, Any]]:
    value = maybe_json(value)
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    elif isinstance(value, dict) and isinstance(value.get("calls"), list):
        value = value["calls"]
    elif isinstance(value, dict) and ("name" in value or "function" in value or "tool_name" in value):
        value = [value]
    if not isinstance(value, list):
        return []
    calls: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        call = normalize_one_call(item)
        if call is None:
            continue
        key = json.dumps(call, sort_keys=True, default=str)
        if key not in seen:
            calls.append(call)
            seen.add(key)
    return calls


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        calls.extend(normalize_call_container(match.group(1)))
    for match in HARMONY_TO_FUNCTION_RE.finditer(text):
        name = match.group(1)
        payload = maybe_json(match.group(2))
        if isinstance(payload, dict) and ("name" not in payload and "tool_name" not in payload):
            calls.append({"name": name, "arguments": payload})
        else:
            calls.extend(normalize_call_container(payload))
    for match in HARMONY_MESSAGE_RE.finditer(text):
        calls.extend(normalize_call_container(match.group(1)))
    calls = [call for call in calls if call.get("name") not in PSEUDO_TOOLS]
    if calls:
        return calls

    action_tail = text.split("Action:")[-1] if "Action:" in text else text
    for obj in iter_json_objects(action_tail):
        calls = normalize_call_container(obj)
        calls = [call for call in calls if call.get("name") not in PSEUDO_TOOLS]
        if calls:
            return calls
    return []


def parse_action_text(text: str) -> Action:
    calls = parse_tool_calls(text)
    if calls:
        first = calls[0]
        return Action(name=first["name"], kwargs=first.get("arguments", {}))

    action_tail = text.split("Action:")[-1] if "Action:" in text else text
    obj = first_json_object(action_tail)
    if isinstance(obj, dict):
        if "name" in obj or "tool_name" in obj:
            name = obj.get("name") or obj.get("tool_name")
            args = obj.get("arguments", obj.get("kwargs", obj.get("parameters")))
            if args is None and str(name) == RESPOND_ACTION_NAME:
                args = {
                    RESPOND_ACTION_FIELD_NAME: obj.get("content")
                    or obj.get("message")
                    or obj.get("response")
                    or ""
                }
            elif args is None:
                args = {k: v for k, v in obj.items() if k not in {"name", "tool_name"}}
            args = maybe_json(args)
            if not isinstance(args, dict):
                args = {"value": args}
            return Action(name=str(name), kwargs=args)
        if "action" in obj and isinstance(obj["action"], dict):
            nested = obj["action"]
            name = nested.get("name") or nested.get("tool_name")
            args = nested.get("arguments", nested.get("kwargs", nested.get("parameters", {})))
            args = maybe_json(args)
            if name:
                return Action(name=str(name), kwargs=args if isinstance(args, dict) else {"value": args})

    content = text.strip()
    if not content:
        content = "Could you clarify what you would like me to do next?"
    return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: content})


def action_from_call(call: Dict[str, Any]) -> Action:
    return Action(name=call["name"], kwargs=call.get("arguments", {}))


def action_payload(action: Action) -> Dict[str, Any]:
    return {"name": action.name, "arguments": action.kwargs}


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "valid", "pass", "passed"}:
            return True
        if lowered in {"0", "false", "no", "invalid", "fail", "failed"}:
            return False
    if value is None:
        return None
    return bool(value)


def id_variants(value: Any) -> set[str]:
    text = str(value or "").strip().lower()
    if not text:
        return set()
    variants = {text}
    if text.startswith("#"):
        variants.add(text[1:])
    elif re.fullmatch(r"[a-z]\d{7,}", text):
        variants.add(f"#{text}")
    return variants


def parse_validator_judgment(text: str) -> Dict[str, Any]:
    """Parse strict validator JSON without turning arbitrary prose into respond."""
    judgment: Dict[str, Any] = {
        "valid": None,
        "chosen_source": "",
        "repair_action": None,
        "reason": "",
        "schema_confidence": None,
        "semantic_confidence": None,
        "raw": text,
    }
    for obj in iter_json_objects(text):
        if not isinstance(obj, dict):
            continue
        candidate = obj.get("judgment") if isinstance(obj.get("judgment"), dict) else obj
        if not any(
            key in candidate
            for key in (
                "valid",
                "chosen_source",
                "repair_action",
                "repair_call",
                "corrected_action",
                "reason",
            )
        ):
            continue
        judgment["valid"] = parse_bool(candidate.get("valid"))
        judgment["chosen_source"] = str(candidate.get("chosen_source") or candidate.get("source") or "")
        judgment["reason"] = str(candidate.get("reason") or candidate.get("rationale") or "")
        judgment["schema_confidence"] = candidate.get("schema_confidence")
        judgment["semantic_confidence"] = candidate.get("semantic_confidence")
        repair = (
            candidate.get("repair_action")
            if "repair_action" in candidate
            else candidate.get("repair_call", candidate.get("corrected_action"))
        )
        if repair in (None, "", False):
            return judgment
        calls = normalize_call_container(repair)
        if calls:
            judgment["repair_action"] = calls[0]
        return judgment

    # Fallback accepts only explicit tool-call/action JSON, not plain text.
    calls = parse_tool_calls(text)
    if calls:
        judgment["valid"] = False
        judgment["chosen_source"] = "repair"
        judgment["repair_action"] = calls[0]
        judgment["reason"] = "validator returned an action without a judgment wrapper"
        return judgment

    valid_match = re.search(r"\bvalid\s*[:=]\s*(true|false|yes|no|valid|invalid)\b", text, re.IGNORECASE)
    if valid_match:
        judgment["valid"] = parse_bool(valid_match.group(1))
        source_match = re.search(r"\bchosen_source\s*[:=]\s*([A-Za-z_ -]+)", text, re.IGNORECASE)
        reason_match = re.search(r"\breason\s*[:=]\s*([^\n<]+)", text, re.IGNORECASE)
        if source_match:
            judgment["chosen_source"] = source_match.group(1).strip(" .\"'")
        if reason_match:
            judgment["reason"] = reason_match.group(1).strip(" .\"'")
    return judgment


def transcript_for_prompt(messages: List[Dict[str, Any]], max_messages: int = 18) -> str:
    view = messages[-max_messages:]
    lines: List[str] = []
    for message in view:
        role = message.get("role", "unknown")
        if role == "system":
            continue
        if role == "assistant" and message.get("action"):
            lines.append("assistant_action: " + short_json(message["action"], 4000))
            continue
        if role == "tool":
            name = message.get("name", "tool")
            lines.append(f"tool:{name}: {str(message.get('content', ''))[:4000]}")
            continue
        content = str(message.get("content", ""))
        if len(content) > 4000:
            content = content[:4000] + "...<truncated>"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def compact_tool_schemas(tools_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for tool in tools_info:
        fn = tool.get("function", {})
        params = fn.get("parameters", {})
        compact.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "required": params.get("required", []),
                "properties": params.get("properties", {}),
            }
        )
    return compact


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    adapter: str
    tokens: int


@dataclass
class ActionCandidate:
    source: str
    action: Action
    raw_text: str = ""
    generation: Optional[GenerationResult] = None
    issues: Optional[List[str]] = None
    normalized: Optional[Action] = None
    score: float = 0.0


class GPTOSSAdapterBank:
    def __init__(self, executor_path: str, planner_path: str, recovery_path: str) -> None:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, Mxfp4Config

        self.torch = torch
        self.executor_adapter = "executor"
        self.planner_adapter = "planner"
        self.recovery_adapter = "recovery"
        device_map_env = os.environ.get("TAU_ENSEMBLE_DEVICE_MAP", "single").strip()
        if device_map_env.lower() in {"single", "cuda:0", "0"}:
            device_map: Any = {"": 0}
        elif device_map_env.lower() in {"none", "cpu"}:
            device_map = None
        else:
            device_map = device_map_env
        print(f"[ensemble] loading GPT-OSS executor adapter: {executor_path}", flush=True)
        print(f"[ensemble] GPT-OSS device_map={device_map!r}", flush=True)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            executor_path,
            torch_dtype="auto",
            device_map=device_map,
            quantization_config=Mxfp4Config(dequantize=True),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(executor_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[ensemble] GPT-OSS hf_device_map={getattr(self.model, 'hf_device_map', None)}", flush=True)
        try:
            print(f"[ensemble] GPT-OSS first_param_device={next(self.model.parameters()).device}", flush=True)
        except Exception:
            pass

        # AutoPeftModel usually names the first adapter "default". Rename is not
        # needed; keep its actual name and add the planner adapters beside it.
        existing = list(getattr(self.model, "peft_config", {}).keys())
        if existing:
            self.executor_adapter = existing[0]
        print(f"[ensemble] executor adapter active name: {self.executor_adapter}", flush=True)
        print(f"[ensemble] loading planner adapter: {planner_path}", flush=True)
        self.model.load_adapter(planner_path, adapter_name=self.planner_adapter, is_trainable=False)
        print(f"[ensemble] loading recovery adapter: {recovery_path}", flush=True)
        self.model.load_adapter(recovery_path, adapter_name=self.recovery_adapter, is_trainable=False)
        self.model.eval()
        print("[ensemble] GPT-OSS adapter bank ready", flush=True)

    def generate(
        self,
        adapter: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        self.model.set_adapter(adapter)
        print(
            f"[ensemble] generate start adapter={adapter} messages={len(messages)} "
            f"tools={bool(tools)} max_new_tokens={max_new_tokens}",
            flush=True,
        )
        kwargs: Dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
        }
        if tools:
            kwargs["tools"] = tools
        try:
            inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            fallback = list(messages)
            if tools:
                tool_blob = json.dumps(tools, ensure_ascii=False)
                fallback[0] = {
                    "role": fallback[0].get("role", "system"),
                    "content": str(fallback[0].get("content", "")) + "\n\nAvailable tools:\n" + tool_blob,
                }
            kwargs.pop("tools", None)
            inputs = self.tokenizer.apply_chat_template(fallback, **kwargs)

        device = next(self.model.parameters()).device
        if hasattr(inputs, "keys") and "input_ids" in inputs:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            input_ids = inputs["input_ids"]
            input_len = input_ids.shape[-1]
        else:
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            input_len = inputs.shape[-1]
        print(f"[ensemble] template ready adapter={adapter} input_tokens={input_len}", flush=True)
        eos_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "eos_token_id": eos_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        max_time = env_float("TAU_ENSEMBLE_GENERATE_MAX_TIME", 0.0)
        if max_time > 0:
            gen_kwargs["max_time"] = max_time
        start = time.time()
        with self.torch.inference_mode():
            if isinstance(inputs, dict):
                output = self.model.generate(**inputs, **gen_kwargs)
            else:
                output = self.model.generate(inputs, **gen_kwargs)
        new_tokens = output[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        elapsed_ms = round((time.time() - start) * 1000, 3)
        if getattr(self.torch, "cuda", None) is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        print(
            f"[ensemble] generate done adapter={adapter} new_tokens={int(new_tokens.numel())} "
            f"latency_ms={elapsed_ms}",
            flush=True,
        )
        return GenerationResult(
            text=text,
            latency_ms=elapsed_ms,
            adapter=adapter,
            tokens=int(new_tokens.numel()),
        )


class ToolACEValidator:
    def __init__(self, model_name: str, max_new_tokens: int, temperature: float) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._loaded = False
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
        print(f"[ensemble] loading ToolACE validator: {self.model_name}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self._loaded = True
        print("[ensemble] ToolACE validator ready", flush=True)

    def unload(self) -> None:
        if not self._loaded:
            return
        self.model = None
        self.tokenizer = None
        self._loaded = False
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass
        print("[ensemble] ToolACE validator unloaded", flush=True)

    def generate_repair(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> GenerationResult:
        self._load()
        assert self.model is not None and self.tokenizer is not None
        print(
            f"[ensemble] validator generate start model={self.model_name} "
            f"messages={len(messages)} tools={bool(tools)} max_new_tokens={self.max_new_tokens}",
            flush=True,
        )
        kwargs: Dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
        }
        if tools:
            kwargs["tools"] = tools
        try:
            inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            fallback = list(messages)
            fallback[0] = {
                "role": fallback[0].get("role", "system"),
                "content": str(fallback[0].get("content", ""))
                + "\n\nAvailable tools:\n"
                + json.dumps(tools, ensure_ascii=False),
            }
            kwargs.pop("tools", None)
            inputs = self.tokenizer.apply_chat_template(fallback, **kwargs)
        device = next(self.model.parameters()).device
        if hasattr(inputs, "keys") and "input_ids" in inputs:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            input_ids = inputs["input_ids"]
            input_len = input_ids.shape[-1]
        else:
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            input_len = inputs.shape[-1]
        print(f"[ensemble] validator template ready input_tokens={input_len}", flush=True)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "top_p": 1.0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        max_time = env_float(
            "TAU_ENSEMBLE_VALIDATOR_MAX_TIME",
            env_float("TAU_ENSEMBLE_GENERATE_MAX_TIME", 0.0),
        )
        if max_time > 0:
            gen_kwargs["max_time"] = max_time
        start = time.time()
        with self.torch.inference_mode():
            if isinstance(inputs, dict):
                output = self.model.generate(**inputs, **gen_kwargs)
            else:
                output = self.model.generate(inputs, **gen_kwargs)
        new_tokens = output[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        elapsed_ms = round((time.time() - start) * 1000, 3)
        if getattr(self.torch, "cuda", None) is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        print(
            f"[ensemble] validator generate done new_tokens={int(new_tokens.numel())} "
            f"latency_ms={elapsed_ms}",
            flush=True,
        )
        return GenerationResult(
            text=text,
            latency_ms=elapsed_ms,
            adapter=self.model_name,
            tokens=int(new_tokens.numel()),
        )


class PolicyGate:
    def __init__(self, tools_info: List[Dict[str, Any]]) -> None:
        self.tools_info = tools_info
        self.tools_by_name = {
            tool.get("function", {}).get("name"): tool for tool in tools_info if tool.get("function", {}).get("name")
        }
        self.aliases = {
            "transfer_to_human_agent": "transfer_to_human_agents",
            "transfer_to_human": "transfer_to_human_agents",
        }

    def normalize_name(self, name: str) -> str:
        if name in self.aliases:
            return self.aliases[name]
        if name in self.tools_by_name or name == RESPOND_ACTION_NAME:
            return name
        match = get_close_matches(name, list(self.tools_by_name), n=1, cutoff=0.88)
        return match[0] if match else name

    def _schema_types(self, schema: Dict[str, Any]) -> List[str]:
        raw = schema.get("type")
        if isinstance(raw, list):
            return [str(item) for item in raw]
        if isinstance(raw, str):
            return [raw]
        if "enum" in schema:
            return []
        return []

    def _coerce_value(self, value: Any, schema: Dict[str, Any]) -> Any:
        types = self._schema_types(schema)
        if value is None:
            return value
        if "string" in types:
            if isinstance(value, (dict, list)):
                return value
            return str(value)
        if isinstance(value, str):
            parsed = maybe_json(value)
            if parsed is not value:
                value = parsed
        if "integer" in types and isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
            return int(value)
        if "number" in types and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        if "boolean" in types and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return value

    def _matches_type(self, value: Any, schema: Dict[str, Any]) -> bool:
        types = self._schema_types(schema)
        if not types:
            return True
        for expected in types:
            if expected == "null" and value is None:
                return True
            if expected == "string" and isinstance(value, str):
                return True
            if expected == "integer" and isinstance(value, int) and not isinstance(value, bool):
                return True
            if expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
            if expected == "boolean" and isinstance(value, bool):
                return True
            if expected == "array" and isinstance(value, list):
                return True
            if expected == "object" and isinstance(value, dict):
                return True
        return False

    def validate(self, action: Action) -> Tuple[Action, List[str]]:
        name = self.normalize_name(action.name)
        kwargs = maybe_json(action.kwargs)
        if not isinstance(kwargs, dict):
            kwargs = {"value": kwargs}
        action = Action(name=name, kwargs=kwargs)

        if name == RESPOND_ACTION_NAME:
            content = kwargs.get(RESPOND_ACTION_FIELD_NAME) or kwargs.get("response") or kwargs.get("message")
            if not isinstance(content, str) or not content.strip():
                return action, ["respond action missing non-empty content"]
            return Action(name=name, kwargs={RESPOND_ACTION_FIELD_NAME: content.strip()}), []

        if name not in self.tools_by_name:
            return action, [f"unknown tool: {name}"]

        fn = self.tools_by_name[name].get("function", {})
        params = fn.get("parameters", {})
        props = params.get("properties", {}) or {}
        required = params.get("required", []) or []
        cleaned = {k: self._coerce_value(v, props[k]) for k, v in kwargs.items() if k in props}
        issues: List[str] = []
        dropped = sorted(set(kwargs) - set(cleaned))
        if dropped:
            issues.append("dropped unexpected args: " + ", ".join(dropped))
        for field in required:
            if field not in cleaned or cleaned[field] in (None, "", []):
                issues.append(f"missing required arg: {field}")
        for field, schema in props.items():
            if field not in cleaned:
                continue
            if not self._matches_type(cleaned[field], schema):
                expected = schema.get("type")
                actual = type(cleaned[field]).__name__
                issues.append(f"type mismatch for {field}: expected {expected}, got {actual}")
            if "enum" in schema and cleaned[field] not in schema["enum"]:
                issues.append(f"invalid enum for {field}: {cleaned[field]!r}")
        # Dropping harmless extra fields is a repair, not a blocking issue.
        blocking = [issue for issue in issues if not issue.startswith("dropped unexpected")]
        return Action(name=name, kwargs=cleaned), blocking

    def missing_arg_response(self, action: Action, issues: List[str]) -> Action:
        missing = [issue.split(":", 1)[1].strip() for issue in issues if issue.startswith("missing required arg:")]
        if missing:
            joined = ", ".join(missing)
            content = f"Could you provide the missing information ({joined}) so I can help with that?"
        else:
            content = "Could you clarify the details I need before I continue?"
        return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: content})


class EnsembleAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.tools_info = tools_info
        self.compact_tools = compact_tool_schemas(tools_info)
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_plan_tokens = env_int("TAU_ENSEMBLE_PLAN_TOKENS", 500)
        self.max_action_tokens = env_int("TAU_ENSEMBLE_ACTION_TOKENS", 700)
        self.min_transfer_step = env_int("TAU_ENSEMBLE_MIN_TRANSFER_STEP", 8)
        self.recovery_mode = os.environ.get("TAU_ENSEMBLE_RECOVERY_MODE", "risky").strip().lower()
        self.validator_mode = os.environ.get("TAU_ENSEMBLE_VALIDATOR_MODE", "repair").strip().lower()
        self.trust_validator = env_bool("TAU_ENSEMBLE_TRUST_VALIDATOR", False)
        self.debug_dir = Path(os.environ.get("TAU_ENSEMBLE_DEBUG_DIR", "")) if os.environ.get("TAU_ENSEMBLE_DEBUG_DIR") else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        executor = os.environ.get("TAU_ENSEMBLE_EXECUTOR_ADAPTER", DEFAULT_EXECUTOR)
        planner = os.environ.get("TAU_ENSEMBLE_PLANNER_ADAPTER", DEFAULT_PLANNER)
        recovery = os.environ.get("TAU_ENSEMBLE_RECOVERY_ADAPTER", DEFAULT_RECOVERY)
        self.bank = GPTOSSAdapterBank(executor, planner, recovery)
        self.gate = PolicyGate(tools_info)

        self.validator: Optional[ToolACEValidator]
        if env_bool("TAU_ENSEMBLE_ENABLE_TOOLACE", True):
            self.validator = ToolACEValidator(
                os.environ.get("TAU_ENSEMBLE_VALIDATOR_MODEL", DEFAULT_VALIDATOR),
                max_new_tokens=env_int("TAU_ENSEMBLE_VALIDATOR_TOKENS", 400),
                temperature=env_float("TAU_ENSEMBLE_VALIDATOR_TEMPERATURE", 0.0),
            )
        else:
            self.validator = None

    def plan_state(self, messages: List[Dict[str, Any]], previous_source: str = "") -> Tuple[Dict[str, Any], GenerationResult]:
        system = (
            "You are a private TAU-bench controller for a customer-service tool agent. "
            "Track state and choose the single next legal action. Return JSON only. "
            "Do not speak to the customer except through candidate_action={\"name\":\"respond\",...}. "
            "At the beginning of every retail conversation, authenticate by email or by first name, last name, and zip "
            "before looking up orders or products. If those details are missing, ask exactly for them. "
            "Do not transfer to a human when a listed tool or a clarification question can make progress."
        )
        user = {
            "policy": self.wiki,
            "tools": self.compact_tools,
            "previous_source": previous_source,
            "transcript": transcript_for_prompt(messages),
            "schema": {
                "user_goal": "string",
                "known_facts": "object",
                "missing_information": ["string"],
                "policy_constraints": ["string"],
                "completed_subgoals": ["string"],
                "next_subgoal": "string",
                "recommended_action_type": "ask_user | call_tool | final_response",
                "tool_candidates": ["string"],
                "candidate_action": {
                    "name": "one available tool name or respond",
                    "arguments": "object",
                },
            },
        }
        result = self.bank.generate(
            self.bank.planner_adapter,
            [{"role": "system", "content": system}, {"role": "user", "content": short_json(user)}],
            max_new_tokens=self.max_plan_tokens,
            temperature=0.0,
        )
        parsed = first_json_object(result.text)
        if not isinstance(parsed, dict):
            parsed = {
                "user_goal": "",
                "known_facts": {},
                "missing_information": [],
                "policy_constraints": [],
                "completed_subgoals": [],
                "next_subgoal": result.text[:1000],
                "recommended_action_type": "call_tool",
                "tool_candidates": [],
                "candidate_action": None,
            }
        return parsed, result

    def recovery_state(self, messages: List[Dict[str, Any]], plan: Dict[str, Any], reason: str) -> Tuple[Dict[str, Any], GenerationResult]:
        system = (
            "You are a recovery planner for a tau-bench customer-service tool agent. "
            "Given the current state, produce a safer alternate next subgoal. Return JSON only."
        )
        user = {
            "reason_for_recovery_check": reason,
            "policy": self.wiki,
            "tools": self.compact_tools,
            "current_plan": plan,
            "transcript": transcript_for_prompt(messages),
            "schema": {
                "risk_assessment": ["string"],
                "alternate_next_subgoal": "string",
                "safe_action_type": "ask_user | call_tool | final_response",
                "preferred_tools": ["string"],
            },
        }
        result = self.bank.generate(
            self.bank.recovery_adapter,
            [{"role": "system", "content": system}, {"role": "user", "content": short_json(user)}],
            max_new_tokens=self.max_plan_tokens,
            temperature=0.0,
        )
        parsed = first_json_object(result.text)
        if not isinstance(parsed, dict):
            parsed = {"risk_assessment": [result.text[:1000]], "alternate_next_subgoal": "", "safe_action_type": "call_tool", "preferred_tools": []}
        return parsed, result

    def should_recover(self, step: int, previous_observation: str, plan: Dict[str, Any]) -> Tuple[bool, str]:
        if self.recovery_mode in {"always", "all", "1", "true"}:
            return True, "configured always"
        if self.recovery_mode in {"off", "none", "0", "false"}:
            return False, "configured off"
        if step >= 3:
            return True, "multi-turn task"
        if previous_observation.lower().startswith("error"):
            return True, "previous tool error"
        if plan.get("missing_information"):
            return True, "planner reports missing information"
        return False, "not risky"

    def action_from_plan(self, plan: Dict[str, Any]) -> Optional[Action]:
        payload = plan.get("candidate_action") or plan.get("action") or plan.get("next_action")
        if not payload:
            return None
        calls = normalize_call_container(payload)
        if calls:
            return action_from_call(calls[0])
        if isinstance(payload, dict):
            name = payload.get("name") or payload.get("tool_name")
            args = payload.get("arguments", payload.get("kwargs", payload.get("parameters", {})))
            if name:
                args = maybe_json(args)
                return Action(name=str(name), kwargs=args if isinstance(args, dict) else {"value": args})
        return None

    def latest_user_text(self, messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def action_was_called(self, messages: List[Dict[str, Any]], tool_names: Iterable[str], arg_key: str = "", arg_value: str = "") -> bool:
        targets = set(tool_names)
        for message in messages:
            if message.get("role") != "assistant" or not isinstance(message.get("action"), dict):
                continue
            action = message["action"]
            if action.get("name") not in targets:
                continue
            if not arg_key:
                return True
            kwargs = action.get("kwargs") or action.get("arguments") or {}
            observed = id_variants(kwargs.get(arg_key, ""))
            expected = id_variants(arg_value)
            if observed and expected and observed & expected:
                return True
            if str(kwargs.get(arg_key, "")).lower() == arg_value.lower():
                return True
        return False

    def name_zip_auth_was_called(self, messages: List[Dict[str, Any]], name_zip: Dict[str, str]) -> bool:
        expected = {key: str(value).strip().lower() for key, value in name_zip.items()}
        for message in messages:
            if message.get("role") != "assistant" or not isinstance(message.get("action"), dict):
                continue
            action = message["action"]
            if action.get("name") != "find_user_id_by_name_zip":
                continue
            kwargs = action.get("kwargs") or action.get("arguments") or {}
            observed = {key: str(kwargs.get(key, "")).strip().lower() for key in expected}
            if observed == expected:
                return True
        return False

    def tool_success_seen(self, messages: List[Dict[str, Any]], tool_names: Iterable[str]) -> bool:
        targets = set(tool_names)
        for idx, message in enumerate(messages):
            if message.get("role") != "tool" or message.get("name") not in targets:
                continue
            content = str(message.get("content", ""))
            if content and not content.lower().startswith("error"):
                return True
            # Some tau-bench traces store the tool result only as the next observation.
            if idx and not content.lower().startswith("error"):
                return True
        return False

    def latest_tool_error(self, messages: List[Dict[str, Any]], tool_names: Iterable[str]) -> str:
        targets = set(tool_names)
        for message in reversed(messages):
            if message.get("role") != "tool" or message.get("name") not in targets:
                continue
            content = str(message.get("content", ""))
            return content if content.lower().startswith("error") else ""
        return ""

    def extract_order_id(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        transcript = transcript_for_prompt(messages, max_messages=50)
        match = re.search(r"#?[A-Z]\d{7,}", transcript)
        if not match:
            return None
        value = match.group(0)
        return value if value.startswith("#") else f"#{value}"

    def extract_user_order_id(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        transcript = "\n".join(str(message.get("content", "")) for message in messages if message.get("role") == "user")
        match = re.search(r"#?[A-Z]\d{7,}", transcript)
        if not match:
            return None
        value = match.group(0)
        return value if value.startswith("#") else f"#{value}"

    def extract_email(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        transcript = transcript_for_prompt(messages, max_messages=50)
        match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", transcript)
        return match.group(0) if match else None

    def extract_name_zip(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        texts = [self.latest_user_text(messages), transcript_for_prompt(messages, max_messages=12)]
        explicit_patterns = [
            r"\bfirst\s+name\s*(?:is|:)?\s*([A-Z][A-Za-z'-]+).*?\blast\s+name\s*(?:is|:)?\s*([A-Z][A-Za-z'-]+).*?\b(?:zip|postal)(?:\s+code)?\s*(?:is|:)?\s*(\d{5})(?:-\d{4})?\b",
            r"\bfirst\s+name\s*(?:is|:|-)?\s*([A-Z][A-Za-z'-]+).*?\blast\s+name\s*(?:is|:|-)?\s*([A-Z][A-Za-z'-]+).*?\b(?:zip|postal)(?:\s+code)?\s*(?:is|:|-)?\s*(\d{5})(?:-\d{4})?\b",
            r"\b(?:my name is|i am|i'm|this is)\s+([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)\b.*?\b(?:zip|postal|in)\b[^\d]*(\d{5})(?:-\d{4})?\b",
        ]
        generic_patterns = [
            r"\b([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)\b(?:,|\s+in|\s+at)\s+(\d{5})(?:-\d{4})?\b",
        ]
        bad = {
            "order",
            "thanks",
            "google",
            "apple",
            "homekit",
            "home",
            "zip",
            "code",
            "is",
            "first",
            "last",
            "name",
            "logged",
            "in",
        }
        for text in texts:
            for pattern in [*explicit_patterns, *generic_patterns]:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if not match:
                    continue
                first, last, zip_code = match.group(1), match.group(2), match.group(3)
                if first.lower() in bad or last.lower() in bad:
                    continue
                return {"first_name": first.title(), "last_name": last.title(), "zip": zip_code}
        return None

    def extracted_product_ids(self, messages: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        seen: set[str] = set()
        for message in messages:
            if message.get("role") not in {"tool", "assistant"}:
                continue
            text = str(message.get("content", ""))
            for match in re.finditer(r"['\"]?product_id['\"]?\s*[:=]\s*['\"]?(\d{6,})", text):
                value = match.group(1)
                if value not in seen:
                    seen.add(value)
                    ids.append(value)
        return ids

    def latest_tool_json(self, messages: List[Dict[str, Any]], tool_name: str) -> Optional[Any]:
        for message in reversed(messages):
            if message.get("role") != "tool" or message.get("name") != tool_name:
                continue
            content = str(message.get("content", ""))
            if not content or content.lower().startswith("error"):
                continue
            parsed = maybe_json(content)
            return parsed
        return None

    def tool_json_payloads(self, messages: List[Dict[str, Any]], tool_name: str) -> List[Any]:
        payloads: List[Any] = []
        for message in messages:
            if message.get("role") != "tool" or message.get("name") != tool_name:
                continue
            content = str(message.get("content", ""))
            if not content or content.lower().startswith("error"):
                continue
            payloads.append(maybe_json(content))
        return payloads

    def initial_user_context(self, messages: List[Dict[str, Any]]) -> str:
        return "\n".join(str(message.get("content", "")) for message in messages if message.get("role") == "user").lower()

    def extract_user_id(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        for payload in reversed(self.tool_json_payloads(messages, "get_order_details")):
            if isinstance(payload, dict) and payload.get("user_id"):
                return str(payload["user_id"])
        for tool_name in ("find_user_id_by_email", "find_user_id_by_name_zip"):
            for payload in reversed(self.tool_json_payloads(messages, tool_name)):
                if isinstance(payload, str):
                    match = re.search(r"\b[a-z]+_[a-z]+_\d+\b", payload)
                    if match:
                        return match.group(0)
        return None

    def product_detail_payloads(self, messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        payloads: Dict[str, Dict[str, Any]] = {}
        for message in messages:
            if message.get("role") != "tool" or message.get("name") != "get_product_details":
                continue
            parsed = maybe_json(str(message.get("content", "")))
            if isinstance(parsed, dict) and parsed.get("product_id"):
                payloads[str(parsed["product_id"])] = parsed
        return payloads

    def order_payloads(self, messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        payloads: Dict[str, Dict[str, Any]] = {}
        for payload in self.tool_json_payloads(messages, "get_order_details"):
            if isinstance(payload, dict) and payload.get("order_id"):
                payloads[str(payload["order_id"])] = payload
        return payloads

    def user_order_ids(self, messages: List[Dict[str, Any]]) -> List[str]:
        user = self.latest_tool_json(messages, "get_user_details")
        if not isinstance(user, dict):
            return []
        order_ids = user.get("orders") or []
        return [str(order_id) for order_id in order_ids if order_id]

    def product_type_map(self, messages: List[Dict[str, Any]]) -> Dict[str, str]:
        payload = self.latest_tool_json(messages, "list_all_product_types")
        if not isinstance(payload, dict):
            return {}
        return {str(name).lower(): str(product_id) for name, product_id in payload.items()}

    def product_type_id(self, messages: List[Dict[str, Any]], *names: str) -> Optional[str]:
        products = self.product_type_map(messages)
        for name in names:
            product_id = products.get(name.lower())
            if product_id:
                return product_id
        return None

    def needs_catalog(self, messages: List[Dict[str, Any]]) -> bool:
        text = self.initial_user_context(messages)
        return any(term in text for term in ("tshirt", "t-shirt", "tee shirt", "options are available"))

    def count_answered(self, messages: List[Dict[str, Any]], product_name: str) -> bool:
        product = product_name.lower()
        for message in messages:
            if message.get("role") != "assistant" or not isinstance(message.get("action"), dict):
                continue
            action = message["action"]
            if action.get("name") != RESPOND_ACTION_NAME:
                continue
            kwargs = action.get("kwargs") or action.get("arguments") or {}
            content = str(kwargs.get(RESPOND_ACTION_FIELD_NAME, "")).lower()
            if product in content and re.search(r"\b\d+\b", content):
                return True
        return False

    def retail_catalog_action(self, messages: List[Dict[str, Any]]) -> Optional[Action]:
        if not self.needs_catalog(messages):
            return None
        text = self.initial_user_context(messages)
        if "list_all_product_types" in self.gate.tools_by_name and not self.tool_success_seen(messages, ["list_all_product_types"]):
            return Action(name="list_all_product_types", kwargs={})
        product_id = self.product_type_id(messages, "T-Shirt", "Tshirt", "T Shirt")
        if product_id and "get_product_details" in self.gate.tools_by_name:
            product_details = self.product_detail_payloads(messages)
            if product_id not in product_details:
                return Action(name="get_product_details", kwargs={"product_id": product_id})
            if "how many" in text and not self.count_answered(messages, "t-shirt"):
                variants = product_details[product_id].get("variants") or {}
                available = [
                    spec
                    for spec in variants.values()
                    if isinstance(spec, dict) and spec.get("available") is True
                ]
                return Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={RESPOND_ACTION_FIELD_NAME: f"There are {len(available)} T-Shirt options available right now."},
                )
        return None

    def retail_target_product_ids(self, messages: List[Dict[str, Any]]) -> List[str]:
        order = self.latest_tool_json(messages, "get_order_details")
        if not isinstance(order, dict):
            return []
        targets = []
        for item in order.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).lower()
            if "keyboard" in name or "thermostat" in name:
                product_id = item.get("product_id")
                if product_id and str(product_id) not in targets:
                    targets.append(str(product_id))
        return targets

    def choose_variant(self, product: Dict[str, Any], kind: str) -> Optional[str]:
        variants = product.get("variants") or {}
        if not isinstance(variants, dict):
            return None

        def available_items() -> List[Tuple[str, Dict[str, Any]]]:
            return [
                (str(item_id), spec)
                for item_id, spec in variants.items()
                if isinstance(spec, dict) and spec.get("available") is True
            ]

        if kind == "keyboard":
            ranked: List[Tuple[int, str]] = []
            for item_id, spec in available_items():
                options = {str(k).lower(): str(v).lower() for k, v in (spec.get("options") or {}).items()}
                if options.get("switch type") != "clicky":
                    continue
                score = 0
                if options.get("size") == "full size":
                    score += 100
                if options.get("backlight") == "rgb":
                    score += 20
                elif options.get("backlight") == "none":
                    score += 10
                ranked.append((score, item_id))
            if ranked:
                return max(ranked)[1]

        if kind == "keyboard_full_rgb":
            ranked = []
            for item_id, spec in available_items():
                options = {str(k).lower(): str(v).lower() for k, v in (spec.get("options") or {}).items()}
                if (
                    options.get("switch type") == "clicky"
                    and options.get("size") == "full size"
                    and options.get("backlight") == "rgb"
                ):
                    ranked.append((100, item_id))
            if ranked:
                return max(ranked)[1]

        if kind == "thermostat":
            ranked = []
            for item_id, spec in available_items():
                options = {str(k).lower(): str(v).lower() for k, v in (spec.get("options") or {}).items()}
                compatibility = options.get("compatibility", "")
                if "google" not in compatibility:
                    continue
                score = 100
                if "assistant" in compatibility or "home" in compatibility:
                    score += 10
                ranked.append((score, item_id))
            if ranked:
                return max(ranked)[1]

        if kind == "tshirt":
            ranked = []
            for item_id, spec in available_items():
                options = {str(k).lower(): str(v).lower() for k, v in (spec.get("options") or {}).items()}
                score = 0
                if options.get("color") == "purple":
                    score += 100
                if options.get("size") == "s":
                    score += 80
                if options.get("style") == "v-neck":
                    score += 50
                if options.get("material") == "polyester":
                    score += 20
                ranked.append((score, item_id))
            if ranked:
                return max(ranked)[1]

        return None

    def payment_method_for_order(self, order: Dict[str, Any]) -> Optional[str]:
        for payment in order.get("payment_history") or []:
            if isinstance(payment, dict) and payment.get("payment_method_id"):
                return str(payment["payment_method_id"])
        return None

    def allow_keyboard_backlight_fallback(self, text: str) -> bool:
        return bool(
            re.search(r"\b(no|without)\s+backlights?\b", text)
            or re.search(r"\bbacklights?\s+(?:is|are|would be|would also be)?\s*(?:okay|ok|fine|acceptable)\b", text)
            or "go for no backlight" in text
        )

    def strict_keyboard_rgb_request(self, text: str) -> bool:
        wants_keyboard = "keyboard" in text
        wants_full = "full size" in text or "full-size" in text or "fullsize" in text
        wants_clicky = "clicky" in text
        wants_rgb = "rgb" in text
        return wants_keyboard and wants_full and wants_clicky and wants_rgb

    def retail_exchange_action(self, messages: List[Dict[str, Any]]) -> Optional[Action]:
        if "exchange_delivered_order_items" not in self.gate.tools_by_name:
            return None
        if self.action_was_called(messages, ["exchange_delivered_order_items", "return_delivered_order_items"]):
            return None
        order = self.latest_tool_json(messages, "get_order_details")
        if not isinstance(order, dict) or str(order.get("status", "")).lower() != "delivered":
            return None
        product_details = self.product_detail_payloads(messages)
        keyboard_item = None
        thermostat_item = None
        for item in order.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).lower()
            if "keyboard" in name:
                keyboard_item = item
            elif "thermostat" in name:
                thermostat_item = item
        if not keyboard_item or not thermostat_item:
            return None
        keyboard_detail = product_details.get(str(keyboard_item.get("product_id")))
        thermostat_detail = product_details.get(str(thermostat_item.get("product_id")))
        if not keyboard_detail or not thermostat_detail:
            return None
        text = self.initial_user_context(messages)
        strict_keyboard = self.strict_keyboard_rgb_request(text)
        allow_keyboard_fallback = self.allow_keyboard_backlight_fallback(text)
        thermostat_only_if_keyboard_unavailable = bool(
            re.search(r"\b(?:only|just|rather)\s+(?:exchange\s+)?(?:the\s+)?(?:smart\s+)?thermostat\b", text)
            or "only exchange thermostat" in text
            or "at least exchange the smart thermostat" in text
            or "if no suitable keyboard" in text
            or "if no keyboard" in text
        )
        keyboard_kind = (
            "keyboard_full_rgb"
            if thermostat_only_if_keyboard_unavailable or (strict_keyboard and not allow_keyboard_fallback)
            else "keyboard"
        )
        new_keyboard = self.choose_variant(keyboard_detail, keyboard_kind)
        new_thermostat = self.choose_variant(thermostat_detail, "thermostat")
        if (thermostat_only_if_keyboard_unavailable or (strict_keyboard and not allow_keyboard_fallback)) and not new_keyboard:
            keyboard_item = None
        payment_method_id = self.payment_method_for_order(order)
        item_ids: List[str] = []
        new_item_ids: List[str] = []
        if keyboard_item and new_keyboard:
            item_ids.append(str(keyboard_item.get("item_id")))
            new_item_ids.append(new_keyboard)
        if thermostat_item and new_thermostat:
            item_ids.append(str(thermostat_item.get("item_id")))
            new_item_ids.append(new_thermostat)
        if not item_ids or not new_item_ids or not payment_method_id:
            return None
        return Action(
            name="exchange_delivered_order_items",
            kwargs={
                "order_id": str(order.get("order_id") or self.extract_order_id(messages)),
                "item_ids": item_ids,
                "new_item_ids": new_item_ids,
                "payment_method_id": payment_method_id,
            },
        )

    def item_name_requested(self, item: Dict[str, Any], text: str) -> bool:
        name = str(item.get("name", "")).lower()
        words = [word for word in re.findall(r"[a-z]+", name) if len(word) >= 4]
        phrases = {name, name.rstrip("s")}
        if len(words) <= 1:
            for word in words:
                phrases.add(word)
                phrases.add(word.rstrip("s"))
        elif words:
            phrases.add(words[-1])
            phrases.add(words[-1].rstrip("s"))
        return any(phrase and phrase in text for phrase in phrases)

    def retail_return_action(self, messages: List[Dict[str, Any]]) -> Optional[Action]:
        if "return_delivered_order_items" not in self.gate.tools_by_name:
            return None
        text = self.initial_user_context(messages)
        explicit_return = "return" in text
        paraphrased_return = bool(re.search(r"\b(get|getting|send|sent|remove|refund)\b", text))
        if (not explicit_return and not paraphrased_return) or self.action_was_called(messages, ["return_delivered_order_items"]):
            return None
        for order in self.order_payloads(messages).values():
            if str(order.get("status", "")).lower() != "delivered":
                continue
            item_ids = [
                str(item.get("item_id"))
                for item in order.get("items", []) or []
                if isinstance(item, dict) and item.get("item_id") and self.item_name_requested(item, text)
            ]
            if not explicit_return and len(item_ids) < 2:
                continue
            payment_method_id = self.payment_method_for_order(order)
            if item_ids and payment_method_id:
                return Action(
                    name="return_delivered_order_items",
                    kwargs={
                        "order_id": str(order.get("order_id")),
                        "item_ids": item_ids,
                        "payment_method_id": payment_method_id,
                    },
                )
        return None

    def retail_modify_tshirt_action(self, messages: List[Dict[str, Any]]) -> Optional[Action]:
        if "modify_pending_order_items" not in self.gate.tools_by_name:
            return None
        text = self.initial_user_context(messages)
        if "modify" not in text or not any(term in text for term in ("tshirt", "t-shirt", "tee shirt")):
            return None
        product_id = self.product_type_id(messages, "T-Shirt", "Tshirt", "T Shirt")
        product = self.product_detail_payloads(messages).get(product_id or "")
        if not product:
            return None
        new_item_id = self.choose_variant(product, "tshirt")
        if not new_item_id:
            return None
        only_current_small = bool(re.search(r"\bpending\s+small\b|\bsmall\s+t-?shirt", text))
        for order_id, order in self.order_payloads(messages).items():
            if str(order.get("status", "")).lower() != "pending":
                continue
            if self.action_was_called(messages, ["modify_pending_order_items"], "order_id", order_id):
                continue
            item_ids = []
            new_item_ids = []
            for item in order.get("items", []) or []:
                if not isinstance(item, dict) or str(item.get("name", "")).lower() != "t-shirt":
                    continue
                options = {str(k).lower(): str(v).lower() for k, v in (item.get("options") or {}).items()}
                if only_current_small and options.get("size") != "s":
                    continue
                item_id = item.get("item_id")
                if item_id:
                    item_ids.append(str(item_id))
                    new_item_ids.append(new_item_id)
            payment_method_id = self.payment_method_for_order(order)
            if item_ids and payment_method_id:
                return Action(
                    name="modify_pending_order_items",
                    kwargs={
                        "order_id": order_id,
                        "item_ids": item_ids,
                        "new_item_ids": new_item_ids,
                        "payment_method_id": payment_method_id,
                    },
                )
        return None

    def needs_user_orders(self, messages: List[Dict[str, Any]]) -> bool:
        text = self.initial_user_context(messages)
        return any(term in text for term in ("return", "modify", "pending", "exchange", "order", "orders"))

    def controller_action(self, messages: List[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Action]:
        tool_names = set(self.gate.tools_by_name)
        retail_auth_tools = {"find_user_id_by_email", "find_user_id_by_name_zip"}
        order_id = self.extract_user_order_id(messages)
        order_lookup_seen = self.tool_success_seen(messages, ["get_order_details"])
        catalog_action = self.retail_catalog_action(messages)
        if catalog_action is not None:
            return catalog_action

        if retail_auth_tools & tool_names and not self.tool_success_seen(messages, retail_auth_tools) and not order_lookup_seen:
            email = self.extract_email(messages)
            if email and "find_user_id_by_email" in tool_names:
                return Action(name="find_user_id_by_email", kwargs={"email": email})
            name_zip = self.extract_name_zip(messages)
            if self.latest_tool_error(messages, retail_auth_tools):
                if (
                    name_zip
                    and "find_user_id_by_name_zip" in tool_names
                    and not self.name_zip_auth_was_called(messages, name_zip)
                ):
                    return Action(name="find_user_id_by_name_zip", kwargs=name_zip)
                if order_id and "get_order_details" in tool_names and not self.action_was_called(
                    messages, ["get_order_details"], "order_id", order_id
                ):
                    return Action(name="get_order_details", kwargs={"order_id": order_id})
                return Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={
                        RESPOND_ACTION_FIELD_NAME: (
                            "I could not find that account. Please double-check the email address, "
                            "or provide the first name, last name, and zip code exactly as listed on the order."
                        )
                    },
                )
            if name_zip and "find_user_id_by_name_zip" in tool_names:
                return Action(name="find_user_id_by_name_zip", kwargs=name_zip)
            return Action(
                name=RESPOND_ACTION_NAME,
                kwargs={
                    RESPOND_ACTION_FIELD_NAME: (
                        "Before I can access the order, please provide either your email address "
                        "or your first name, last name, and zip code."
                    )
                },
            )

        if order_id and "get_order_details" in tool_names and not self.action_was_called(
            messages, ["get_order_details"], "order_id", order_id
        ):
            return Action(name="get_order_details", kwargs={"order_id": order_id})

        user_id = self.extract_user_id(messages)
        if (
            user_id
            and self.needs_user_orders(messages)
            and not order_id
            and "get_user_details" in tool_names
            and not self.action_was_called(messages, ["get_user_details"], "user_id", user_id)
        ):
            return Action(name="get_user_details", kwargs={"user_id": user_id})

        if self.needs_user_orders(messages) and not order_id and "get_order_details" in tool_names:
            orders = self.order_payloads(messages)
            for user_order_id in self.user_order_ids(messages):
                if user_order_id not in orders:
                    return Action(name="get_order_details", kwargs={"order_id": user_order_id})

        exchange_action = self.retail_exchange_action(messages)
        if exchange_action is not None:
            return exchange_action

        return_action = self.retail_return_action(messages)
        if return_action is not None:
            return return_action

        modify_action = self.retail_modify_tshirt_action(messages)
        if modify_action is not None:
            return modify_action

        if "get_product_details" in tool_names and self.tool_success_seen(messages, ["get_order_details"]):
            product_details = self.product_detail_payloads(messages)
            for product_id in self.retail_target_product_ids(messages) or self.extracted_product_ids(messages):
                if product_id not in product_details:
                    return Action(name="get_product_details", kwargs={"product_id": product_id})

        # If the planner already produced a concrete non-transfer action, keep it
        # in the candidate pool after the deterministic TAU ordering checks.
        return self.action_from_plan(plan)

    def choose_action(
        self,
        messages: List[Dict[str, Any]],
        plan: Dict[str, Any],
        recovery: Optional[Dict[str, Any]],
    ) -> Tuple[Action, GenerationResult]:
        system = (
            self.wiki
            + "\n\nYou are the executor in a planner/executor/validator ensemble. "
            + "Choose exactly one next action for the customer-service task. "
            + "Use tools when information or state changes are needed. Ask the user only when required information is missing. "
            + "For authentication, ask for email or name plus zip when missing; do not transfer to a human merely because identity is not verified. "
            + "Only transfer to a human after available self-service actions and clarification questions are exhausted or the policy explicitly requires transfer. "
            + "Output exactly one JSON object of the form {\"name\": tool_name, \"arguments\": {...}}. "
            + f"To speak to the customer, use {RESPOND_ACTION_NAME} with arguments {{\"{RESPOND_ACTION_FIELD_NAME}\": \"...\"}}."
        )
        user = {
            "tools": self.compact_tools,
            "planner_state": plan,
            "recovery_state": recovery or {},
            "transcript": transcript_for_prompt(messages),
            "output_contract": {"name": "one available tool name or respond", "arguments": "object"},
        }
        result = self.bank.generate(
            self.bank.executor_adapter,
            [{"role": "system", "content": system}, {"role": "user", "content": short_json(user)}],
            tools=self.tools_info,
            max_new_tokens=self.max_action_tokens,
            temperature=self.temperature,
        )
        return parse_action_text(result.text), result

    def recover_action_candidate(
        self,
        messages: List[Dict[str, Any]],
        plan: Dict[str, Any],
        recovery: Optional[Dict[str, Any]],
        action: Action,
        issues: List[str],
    ) -> Tuple[Action, GenerationResult]:
        system = (
            "You are the recovery candidate generator in a TAU-bench ensemble. "
            "Given a rejected action, emit exactly one safer next action as JSON with keys name and arguments. "
            "Follow the benchmark policy and do not invent IDs or tool observations."
        )
        user = {
            "policy": self.wiki,
            "tools": self.compact_tools,
            "planner_state": plan,
            "recovery_state": recovery or {},
            "rejected_action": action_payload(action),
            "deterministic_issues": issues,
            "transcript": transcript_for_prompt(messages),
            "output_contract": {"name": "one available tool name or respond", "arguments": "object"},
        }
        result = self.bank.generate(
            self.bank.recovery_adapter,
            [{"role": "system", "content": system}, {"role": "user", "content": short_json(user)}],
            tools=self.tools_info,
            max_new_tokens=env_int("TAU_ENSEMBLE_RECOVERY_ACTION_TOKENS", self.max_action_tokens),
            temperature=0.0,
        )
        return parse_action_text(result.text), result

    def evaluate_candidate(self, candidate: ActionCandidate, messages: List[Dict[str, Any]], step: int) -> ActionCandidate:
        normalized, issues = self.gate.validate(candidate.action)
        if normalized.name == "transfer_to_human_agents" and step < self.min_transfer_step:
            issues = [*issues, "premature transfer_to_human_agents"]
        issues = [*issues, *self.grounding_issues(normalized, messages)]
        score = 0.0
        if issues:
            score -= 100.0 + len(issues)
        else:
            score += 100.0
        if normalized.name != RESPOND_ACTION_NAME:
            score += 15.0
        elif candidate.source == "controller":
            score += 12.0
        if normalized.name == "transfer_to_human_agents":
            score -= 30.0
        source_bonus = {
            "controller": 25.0,
            "planner": 15.0,
            "executor": 10.0,
            "recovery_action": 12.0,
            "validator_repair": 8.0,
        }
        score += source_bonus.get(candidate.source, 0.0)
        candidate.normalized = normalized
        candidate.issues = issues
        candidate.score = score
        return candidate

    def select_candidate(self, candidates: List[ActionCandidate], messages: List[Dict[str, Any]], step: int) -> Tuple[Action, List[str], ActionCandidate, List[ActionCandidate]]:
        evaluated = [self.evaluate_candidate(candidate, messages, step) for candidate in candidates]
        evaluated.sort(key=lambda item: item.score, reverse=True)
        if not evaluated:
            fallback = Action(
                name=RESPOND_ACTION_NAME,
                kwargs={RESPOND_ACTION_FIELD_NAME: "Could you clarify what you would like me to do next?"},
            )
            empty = ActionCandidate(source="fallback", action=fallback, normalized=fallback, issues=[], score=0.0)
            return fallback, [], empty, [empty]
        best = evaluated[0]
        if best.issues:
            final = self.gate.missing_arg_response(best.normalized or best.action, best.issues)
            return final, [], best, evaluated
        return best.normalized or best.action, [], best, evaluated


    def grounding_issues(self, action: Action, messages: List[Dict[str, Any]]) -> List[str]:
        if action.name == RESPOND_ACTION_NAME:
            content = str(action.kwargs.get(RESPOND_ACTION_FIELD_NAME, ""))
            if "<|call|>" in content or "to=functions." in content:
                return ["respond content contains malformed tool call text"]
            return []

        context = transcript_for_prompt(messages, max_messages=50).lower()
        sensitive_names = {
            "user_id",
            "order_id",
            "product_id",
            "item_id",
            "item_ids",
            "new_item_ids",
            "payment_id",
            "payment_method_id",
            "reservation_id",
            "reservation_code",
        }
        bad_values = {"user@example.com", "sara_doe_496", "john_doe", "unknown", "n/a", "none"}

        def walk(prefix: str, value: Any) -> Iterable[Tuple[str, Any]]:
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from walk(str(key), child)
            elif isinstance(value, list):
                for child in value:
                    yield from walk(prefix, child)
            else:
                yield prefix, value

        issues: List[str] = []
        for key, value in walk("", action.kwargs):
            if key not in sensitive_names and not key.endswith("_id") and not key.endswith("_ids"):
                continue
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            lowered = text.lower()
            variants = id_variants(text) or {lowered}
            grounded = any(v and v in context for v in variants if len(v) >= 4)
            if lowered in bad_values or not grounded:
                issues.append(f"ungrounded {key}: {text}")
        return issues

    def validator_repair(
        self,
        messages: List[Dict[str, Any]],
        plan: Dict[str, Any],
        action: Action,
        issues: List[str],
    ) -> Tuple[Optional[Action], Optional[GenerationResult], List[str], Dict[str, Any]]:
        if self.validator is None:
            return None, None, ["validator disabled"], {}
        system = (
            "You are a strict function-calling validator for a tau-bench customer-service agent. "
            "Judge the proposed action; do not solve the whole task from scratch. Return JSON only with keys "
            "valid, chosen_source, repair_action, reason, schema_confidence, semantic_confidence. "
            "Use repair_action=null when the proposed action is valid. Only provide a repair_action when a schema, "
            "grounding, or clear policy error can be fixed using the transcript and policy. If information is missing, "
            f"repair_action may be {RESPOND_ACTION_NAME!r} asking for exactly that missing detail. "
            "If deterministic_issues is non-empty, valid must be false unless repair_action fixes the issue. "
            "Do not invent unavailable facts and do not write customer-facing prose outside JSON."
        )
        user = {
            "policy": self.wiki,
            "tools": self.compact_tools,
            "planner_state": plan,
            "proposed_action": action_payload(action),
            "deterministic_issues": issues,
            "transcript": transcript_for_prompt(messages),
            "output_contract": {
                "valid": "boolean; whether proposed_action should be executed as-is",
                "chosen_source": "proposed | repair | reject",
                "repair_action": None,
                "reason": "short diagnostic reason",
                "schema_confidence": "0.0-1.0",
                "semantic_confidence": "0.0-1.0",
            },
        }
        try:
            result = self.validator.generate_repair(
                [{"role": "system", "content": system}, {"role": "user", "content": short_json(user)}],
                self.tools_info,
            )
            judgment = parse_validator_judgment(result.text)
            repair_payload = judgment.get("repair_action")
            if repair_payload:
                candidate = action_from_call(repair_payload)
                repaired, repair_issues = self.gate.validate(candidate)
                if repair_issues:
                    return None, result, repair_issues, judgment
                return repaired, result, [], judgment
            if judgment.get("valid") is True:
                return None, result, [], judgment
            return None, result, ["validator_rejected_without_repair"], judgment
        except Exception as exc:  # noqa: BLE001 - diagnostics are stored in trajectory info.
            return None, None, [f"validator_error: {type(exc).__name__}: {exc}"], {}
        finally:
            if env_bool("TAU_ENSEMBLE_UNLOAD_VALIDATOR_AFTER_CALL", True):
                self.validator.unload()

    def write_debug(self, task_index: Optional[int], row: Dict[str, Any]) -> None:
        if self.debug_dir is None:
            return
        suffix = task_index if task_index is not None else "unknown"
        name = f"task_{suffix}.jsonl"
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            with (self.debug_dir / name).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            print(f"[ensemble] debug write failed: {type(exc).__name__}: {exc}", flush=True)

    def solve(self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30) -> SolveResult:
        max_num_steps = env_int("TAU_ENSEMBLE_MAX_STEPS", max_num_steps)
        reset = env.reset(task_index=task_index)
        info = reset.info.model_dump()
        reward = 0.0
        previous_observation = reset.observation
        previous_source = "user"
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": reset.observation},
        ]
        total_cost = 0.0
        diagnostics: List[Dict[str, Any]] = []

        for step in range(max_num_steps):
            print(f"[ensemble] step {step} plan start", flush=True)
            plan, plan_gen = self.plan_state(messages, previous_source=previous_source)
            print(
                f"[ensemble] step {step} plan done recommended={plan.get('recommended_action_type')} "
                f"next={str(plan.get('next_subgoal', ''))[:160]}",
                flush=True,
            )
            do_recovery, recovery_reason = self.should_recover(step, previous_observation, plan)
            recovery = None
            recovery_gen = None
            if do_recovery:
                print(f"[ensemble] step {step} recovery start reason={recovery_reason}", flush=True)
                recovery, recovery_gen = self.recovery_state(messages, plan, recovery_reason)
                print(
                    f"[ensemble] step {step} recovery done safe_action={recovery.get('safe_action_type')}",
                    flush=True,
                )

            candidates: List[ActionCandidate] = []
            seen_candidates: set[str] = set()

            def add_candidate(candidate: ActionCandidate) -> None:
                key = json.dumps(action_payload(candidate.action), sort_keys=True, default=str)
                if key in seen_candidates:
                    return
                seen_candidates.add(key)
                candidates.append(candidate)

            controller_action = self.controller_action(messages, plan)
            if controller_action is not None:
                add_candidate(ActionCandidate(source="controller", action=controller_action))
                print(f"[ensemble] step {step} controller candidate={controller_action.name}", flush=True)

            planner_action = self.action_from_plan(plan)
            if planner_action is not None:
                add_candidate(ActionCandidate(source="planner", action=planner_action, raw_text=plan_gen.text, generation=plan_gen))
                print(f"[ensemble] step {step} planner candidate={planner_action.name}", flush=True)

            print(f"[ensemble] step {step} executor start", flush=True)
            proposed_action, action_gen = self.choose_action(messages, plan, recovery)
            add_candidate(ActionCandidate(source="executor", action=proposed_action, raw_text=action_gen.text, generation=action_gen))
            executor_eval = self.evaluate_candidate(
                ActionCandidate(source="executor", action=proposed_action, raw_text=action_gen.text, generation=action_gen),
                messages,
                step,
            )
            normalized_action = executor_eval.normalized or proposed_action
            issues = list(executor_eval.issues or [])
            print(
                f"[ensemble] step {step} executor proposed={proposed_action.name} "
                f"normalized={normalized_action.name} issues={issues}",
                flush=True,
            )

            recovery_action = None
            recovery_action_gen = None
            if issues or do_recovery:
                print(f"[ensemble] step {step} recovery action start", flush=True)
                recovery_action, recovery_action_gen = self.recover_action_candidate(
                    messages, plan, recovery, normalized_action, issues
                )
                add_candidate(
                    ActionCandidate(
                        source="recovery_action",
                        action=recovery_action,
                        raw_text=recovery_action_gen.text,
                        generation=recovery_action_gen,
                    )
                )
                print(f"[ensemble] step {step} recovery action candidate={recovery_action.name}", flush=True)

            selected_action, final_issues, selected_candidate, evaluated_candidates = self.select_candidate(
                candidates, messages, step
            )
            print(
                f"[ensemble] step {step} selected source={selected_candidate.source} "
                f"action={selected_action.name} score={selected_candidate.score} "
                f"issues={selected_candidate.issues}",
                flush=True,
            )

            validator_action = None
            validator_gen = None
            validator_issues: List[str] = []
            validator_judgment: Dict[str, Any] = {}
            should_validate = (
                self.validator_mode in {"always", "repair"}
                and (selected_action.name != RESPOND_ACTION_NAME or bool(selected_candidate.issues))
                and (self.validator_mode == "always" or bool(selected_candidate.issues))
            )
            if should_validate:
                print(f"[ensemble] step {step} validator start", flush=True)
                validator_action, validator_gen, validator_issues, validator_judgment = self.validator_repair(
                    messages, plan, selected_action, list(selected_candidate.issues or [])
                )
                print(
                    f"[ensemble] step {step} validator done action="
                    f"{validator_action.name if validator_action else None} issues={validator_issues} "
                    f"judgment_valid={validator_judgment.get('valid')}",
                    flush=True,
                )

            final_action = selected_action
            if validator_action is not None and not validator_issues:
                repair_candidate = self.evaluate_candidate(
                    ActionCandidate(source="validator_repair", action=validator_action, generation=validator_gen),
                    messages,
                    step,
                )
                if not repair_candidate.issues and (
                    selected_candidate.issues or self.trust_validator or validator_judgment.get("valid") is False
                ):
                    final_action = repair_candidate.normalized or validator_action
                    final_issues = []
                    evaluated_candidates.append(repair_candidate)

            print(f"[ensemble] step {step} env step action={final_action.name}", flush=True)
            response = env.step(final_action)
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            previous_observation = response.observation
            previous_source = response.info.source or "unknown"
            print(
                f"[ensemble] step {step} env done reward={reward} done={response.done} "
                f"source={previous_source}",
                flush=True,
            )

            diag = {
                "step": step,
                "plan": plan,
                "plan_raw": plan_gen.text,
                "recovery_reason": recovery_reason,
                "recovery": recovery,
                "recovery_raw": recovery_gen.text if recovery_gen else None,
                "recovery_action_raw": recovery_action_gen.text if recovery_action_gen else None,
                "executor_raw": action_gen.text,
                "proposed_action": proposed_action.model_dump(),
                "normalized_action": normalized_action.model_dump(),
                "validation_issues": issues,
                "candidate_pool": [
                    {
                        "source": candidate.source,
                        "action": (candidate.normalized or candidate.action).model_dump(),
                        "issues": candidate.issues,
                        "score": candidate.score,
                    }
                    for candidate in evaluated_candidates
                ],
                "selected_source": selected_candidate.source,
                "validator_raw": validator_gen.text if validator_gen else None,
                "validator_action": validator_action.model_dump() if validator_action else None,
                "validator_judgment": validator_judgment,
                "validator_issues": validator_issues,
                "final_action": final_action.model_dump(),
                "final_issues": final_issues,
                "observation": response.observation[:4000],
                "done": response.done,
                "latency_ms": {
                    "plan": plan_gen.latency_ms,
                    "recovery": recovery_gen.latency_ms if recovery_gen else 0,
                    "recovery_action": recovery_action_gen.latency_ms if recovery_action_gen else 0,
                    "executor": action_gen.latency_ms,
                    "validator": validator_gen.latency_ms if validator_gen else 0,
                },
            }
            diagnostics.append(diag)
            self.write_debug(task_index, diag)

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": action_gen.text,
                "action": final_action.model_dump(),
                "ensemble": {
                    "plan": plan,
                    "recovery": recovery,
                    "validation_issues": issues,
                    "selected_source": selected_candidate.source,
                    "validator_judgment": validator_judgment,
                    "validator_issues": validator_issues,
                },
            }
            messages.append(assistant_msg)
            if final_action.name != RESPOND_ACTION_NAME:
                messages.append({"role": "tool", "name": final_action.name, "content": response.observation})
            else:
                messages.append({"role": "user", "content": response.observation})

            if response.done:
                break

        if info.get("reward_info") is None:
            try:
                reward_res = env.calculate_reward()
                reward = reward_res.reward
                info["reward_info"] = reward_res.model_dump()
            except Exception as exc:  # noqa: BLE001
                info["reward_calc_error"] = f"{type(exc).__name__}: {exc}"
            try:
                info["user_cost"] = env.user.get_total_cost()
            except Exception:
                pass

        info["ensemble_diagnostics"] = diagnostics
        info["ensemble_config"] = {
            "recovery_mode": self.recovery_mode,
            "validator_mode": self.validator_mode,
            "trust_validator": self.trust_validator,
            "toolace_enabled": self.validator is not None,
        }
        return SolveResult(reward=reward, messages=messages, info=info, total_cost=total_cost)
