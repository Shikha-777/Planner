"""GPT-OSS ensemble adapter for tau2/tau3 half-duplex text evals.

This module plugs the existing local GPT-OSS adapter bank into tau2's
HalfDuplexAgent interface.  It intentionally avoids task-specific scripts:
the agent sees only the conversation, domain policy, and visible tool schemas.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Iterable, Optional

from pydantic import BaseModel

from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.environment.tool import Tool


SYSTEM_PROMPT = """
You are a customer service agent being evaluated in tau3-bench.

Follow the policy exactly. In each turn, do exactly one of:
- respond to the user with text, or
- call one available tool.

Do not output both text and a tool call. Do not invent tool results. Use only
the visible tool schemas. Ask the user for missing information when policy or
tool schemas require it.

Never invent identifiers, names, addresses, dates, zip/postal codes, product
attributes, account details, or other arguments. Tool arguments must come
verbatim from the conversation, policy, or prior tool results.

Do not transfer to a human agent while authentication, order lookup, product
lookup, clarification, or another visible tool can still make progress.
Never expose private reasoning, analysis, commentary, plans, or tool syntax to
the user. If responding with text, write only the customer-facing message.

<policy>
{domain_policy}
</policy>
""".strip()

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
RESPOND_ACTION_FIELD_NAME = "content"
RESPOND_ACTION_NAME = "respond"


@dataclass
class Action:
    name: str
    kwargs: dict[str, Any]


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    adapter: str
    tokens: int


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


def normalize_one_call(value: Any) -> Optional[dict[str, Any]]:
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


def normalize_call_container(value: Any) -> list[dict[str, Any]]:
    value = maybe_json(value)
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    elif isinstance(value, dict) and isinstance(value.get("calls"), list):
        value = value["calls"]
    elif isinstance(value, dict) and ("name" in value or "function" in value or "tool_name" in value):
        value = [value]
    if not isinstance(value, list):
        return []
    calls: list[dict[str, Any]] = []
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


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
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
        for response_field in (RESPOND_ACTION_FIELD_NAME, "message", "response"):
            if response_field in obj and isinstance(obj[response_field], str):
                return Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={RESPOND_ACTION_FIELD_NAME: obj[response_field]},
                )

    content = text.strip()
    if not content:
        content = "Could you clarify what you would like me to do next?"
    return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: content})


def compact_tool_schemas(tools_info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
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


class GPTOSSAdapterBank:
    def __init__(self, executor_path: str, planner_path: str, recovery_path: str) -> None:
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
        device_map_env = os.environ.get("TAU_ENSEMBLE_DEVICE_MAP", "single").strip()
        if device_map_env.lower() in {"single", "cuda:0", "0"}:
            device_map: Any = {"": 0}
        elif device_map_env.lower() in {"none", "cpu"}:
            device_map = None
        else:
            device_map = device_map_env
        print(f"[ensemble] loading GPT-OSS executor adapter: {executor_path}", flush=True)
        print(f"[ensemble] GPT-OSS device_map={device_map!r}", flush=True)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "device_map": device_map,
        }
        if Mxfp4Config is not None:
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        self.model = AutoPeftModelForCausalLM.from_pretrained(executor_path, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(executor_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[ensemble] GPT-OSS hf_device_map={getattr(self.model, 'hf_device_map', None)}", flush=True)
        try:
            print(f"[ensemble] GPT-OSS first_param_device={next(self.model.parameters()).device}", flush=True)
        except Exception:
            pass

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
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
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
        kwargs: dict[str, Any] = {
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

        gen_kwargs: dict[str, Any] = {
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


class PolicyGate:
    def __init__(self, tools_info: list[dict[str, Any]]) -> None:
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

    def _schema_types(self, schema: dict[str, Any]) -> list[str]:
        raw = schema.get("type")
        if isinstance(raw, list):
            return [str(item) for item in raw]
        if isinstance(raw, str):
            return [raw]
        if "enum" in schema:
            return []
        return []

    def _coerce_value(self, value: Any, schema: dict[str, Any]) -> Any:
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

    def _matches_type(self, value: Any, schema: dict[str, Any]) -> bool:
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

    def validate(self, action: Action) -> tuple[Action, list[str]]:
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
        issues: list[str] = []
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
        blocking = [issue for issue in issues if not issue.startswith("dropped unexpected")]
        return Action(name=name, kwargs=cleaned), blocking

    def missing_arg_response(self, action: Action, issues: list[str]) -> Action:
        missing = [issue.split(":", 1)[1].strip() for issue in issues if issue.startswith("missing required arg:")]
        if missing:
            joined = ", ".join(missing)
            content = f"Could you provide the missing information ({joined}) so I can help with that?"
        else:
            content = "Could you clarify the details I need before I continue?"
        return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: content})


class Tau3EnsembleState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[Any]
    step: int = 0


def _jsonish(value: Any, limit: int = 3000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _message_to_line(message: Any) -> str:
    role = getattr(message, "role", None) or "unknown"
    content = getattr(message, "content", None)
    if role == "assistant" and getattr(message, "tool_calls", None):
        calls = [
            {"name": call.name, "arguments": call.arguments}
            for call in (message.tool_calls or [])
        ]
        return "assistant tool_call: " + _jsonish(calls, 4000)
    if role == "tool":
        name = getattr(message, "name", "") or getattr(message, "tool_name", "")
        return f"tool {name}: {str(content or '')[:4000]}"
    if isinstance(message, MultiToolMessage):
        return "\n".join(_message_to_line(m) for m in message.tool_messages)
    return f"{role}: {str(content or '')[:4000]}"


def _transcript(messages: list[Any], limit: int = 24) -> str:
    lines = [_message_to_line(message) for message in messages[-limit:]]
    return "\n".join(line for line in lines if line)


def _strip_harmony(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"\b(analysis|commentary|final)\b\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_internal(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "<|channel|>",
            "we have a user request",
            "the user wants",
            "according to the policy",
            "we need to",
            "private_plan",
            "candidate_action",
        )
    )


def _get_role(message: Any) -> str:
    return str(getattr(message, "role", "") or "")


def _get_content(message: Any) -> str:
    return str(getattr(message, "content", "") or "")


def _grounding_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _assistant_tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or []
    result: list[dict[str, Any]] = []
    for call in calls:
        result.append(
            {
                "name": getattr(call, "name", ""),
                "arguments": getattr(call, "arguments", {}) or {},
            }
        )
    return result


def _respond_message(content: str, raw: Optional[dict[str, Any]] = None) -> AssistantMessage:
    content = _strip_harmony(content) or "Could you clarify what you would like me to do next?"
    return AssistantMessage(
        role="assistant",
        content=content,
        raw_data=raw,
    )


def _tool_message(name: str, arguments: dict[str, Any], raw: Optional[dict[str, Any]] = None) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=name,
                arguments=arguments,
                requestor="assistant",
            )
        ],
        raw_data=raw,
    )


class Tau3EnsembleAgent(LLMConfigMixin, HalfDuplexAgent[Tau3EnsembleState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ) -> None:
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args or {},
        )
        self.openai_tools = [tool.openai_schema for tool in tools]
        self.compact_tools = compact_tool_schemas(self.openai_tools)
        self.gate = PolicyGate(self.openai_tools)
        self.max_plan_tokens = env_int("TAU3_ENSEMBLE_PLAN_TOKENS", 256)
        self.max_action_tokens = env_int("TAU3_ENSEMBLE_ACTION_TOKENS", 384)
        self.max_repair_tokens = env_int("TAU3_ENSEMBLE_REPAIR_TOKENS", 320)
        self.min_transfer_step = env_int("TAU3_ENSEMBLE_MIN_TRANSFER_STEP", 8)
        self.temperature = float((llm_args or {}).get("temperature", env_float("TAU3_ENSEMBLE_TEMPERATURE", 0.0)))
        self.debug_dir = os.environ.get("TAU3_ENSEMBLE_DEBUG_DIR")
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.bank = GPTOSSAdapterBank(
            os.environ.get("TAU_ENSEMBLE_EXECUTOR_ADAPTER", DEFAULT_EXECUTOR),
            os.environ.get("TAU_ENSEMBLE_PLANNER_ADAPTER", DEFAULT_PLANNER),
            os.environ.get("TAU_ENSEMBLE_RECOVERY_ADAPTER", DEFAULT_RECOVERY),
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(domain_policy=self.domain_policy)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> Tau3EnsembleState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only valid tau2 agent history messages."
        )
        return Tau3EnsembleState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
            step=0,
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: Tau3EnsembleState,
    ) -> tuple[AssistantMessage, Tau3EnsembleState]:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Tau3EnsembleAgent supports text half-duplex only.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        assistant_message = self._generate_next_message(state)
        state.messages.append(assistant_message)
        state.step += 1
        return assistant_message, state

    def _planner_note(self, state: Tau3EnsembleState) -> tuple[str, dict[str, Any]]:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a private planner for a tau3-bench customer-service agent. "
                    "Return compact JSON. Do not include gold answers or hidden state."
                ),
            },
            {
                "role": "user",
                "content": short_json(
                    {
                        "policy": self.domain_policy,
                        "tools": self.compact_tools,
                        "transcript": _transcript(state.messages),
                        "schema": {
                            "known_facts": "object",
                            "missing_information": ["string"],
                            "next_subgoal": "string",
                            "preferred_tool_names": ["string"],
                        },
                    },
                    limit=18000,
                ),
            },
        ]
        result = self.bank.generate(
            self.bank.planner_adapter,
            prompt,
            max_new_tokens=self.max_plan_tokens,
            temperature=0.0,
        )
        note: dict[str, Any]
        try:
            decoder = json.JSONDecoder()
            note = {}
            for idx, char in enumerate(result.text):
                if char not in "{[":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(result.text[idx:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    note = parsed
                    break
        except Exception:
            note = {}
        return result.text, note

    def _generate_next_message(self, state: Tau3EnsembleState) -> AssistantMessage:
        start = time.time()
        plan_text, plan = self._planner_note(state)
        action_prompt = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": short_json(
                    {
                        "private_plan": plan,
                        "available_tools": self.compact_tools,
                        "conversation": _transcript(state.messages),
                        "output_format": {
                            "tool_call": {"name": "tool_name", "arguments": {}},
                            "text_response": {
                                "name": "respond",
                                "arguments": {RESPOND_ACTION_FIELD_NAME: "..."},
                            },
                        },
                        "strict_output_rules": [
                            "Return exactly one tool call or one respond action.",
                            "Do not transfer unless the policy cannot be completed by tools or clarification.",
                            "Do not emit private analysis/commentary/final channel text.",
                            "Do not repeat a tool call with identical arguments if its result is already in the conversation.",
                            "Do not invent missing argument values; ask for missing information instead.",
                        ],
                        "tool_call_history": self._tool_call_history(state.messages),
                    },
                    limit=22000,
                ),
            },
        ]
        generation = self.bank.generate(
            self.bank.executor_adapter,
            action_prompt,
            tools=self.openai_tools,
            max_new_tokens=self.max_action_tokens,
            temperature=self.temperature,
        )
        action = parse_action_text(generation.text)
        action, issues = self.gate.validate(action)
        issues.extend(self._unsupported_argument_issues(state, action))
        action, repair_generation, repair_issues = self._repair_if_needed(
            state=state,
            action=action,
            issues=issues,
            generation_text=generation.text,
        )
        if repair_generation is not None:
            generation = repair_generation
            issues = repair_issues
        if issues:
            action = self.gate.missing_arg_response(action, issues)
            action, _ = self.gate.validate(action)

        raw = {
            "planner_raw": plan_text[:2000],
            "executor_raw": generation.text[:2000],
            "latency_ms": generation.latency_ms,
            "adapter": generation.adapter,
            "schema_issues": issues,
        }
        elapsed = time.time() - start
        if action.name == RESPOND_ACTION_NAME:
            content = (
                action.kwargs.get(RESPOND_ACTION_FIELD_NAME)
                or action.kwargs.get("content")
                or action.kwargs.get("message")
                or generation.text
            )
            msg = _respond_message(str(content), raw)
        else:
            msg = _tool_message(action.name, action.kwargs, raw)
        msg.generation_time_seconds = elapsed
        self._write_debug_turn(state, action, raw, elapsed)
        return msg

    def _write_debug_turn(
        self,
        state: Tau3EnsembleState,
        action: Action,
        raw: dict[str, Any],
        elapsed: float,
    ) -> None:
        if not self.debug_dir:
            return
        record = {
            "step": state.step,
            "elapsed_seconds": elapsed,
            "action": {"name": action.name, "arguments": action.kwargs},
            "schema_issues": raw.get("schema_issues", []),
            "adapter": raw.get("adapter"),
            "planner_raw": raw.get("planner_raw", ""),
            "executor_raw": raw.get("executor_raw", ""),
        }
        path = os.path.join(self.debug_dir, "turns.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            print(f"[ensemble] debug write failed: {exc}", flush=True)

    def _tool_call_history(self, messages: list[Any], limit: int = 12) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        pending: Optional[dict[str, Any]] = None
        for message in messages:
            calls = _assistant_tool_calls(message)
            if calls:
                for call in calls:
                    pending = {
                        "name": call.get("name"),
                        "arguments": call.get("arguments") or {},
                        "result": None,
                    }
                    history.append(pending)
                continue
            if pending is not None and _get_role(message) == "tool":
                pending["result"] = _get_content(message)[:1000]
                pending = None
        return history[-limit:]

    def _has_duplicate_tool_call(self, state: Tau3EnsembleState, action: Action) -> bool:
        if action.name == RESPOND_ACTION_NAME:
            return False
        signature = json.dumps(
            {"name": action.name, "arguments": action.kwargs},
            sort_keys=True,
            default=str,
        )
        for item in self._tool_call_history(state.messages, limit=30):
            item_signature = json.dumps(
                {"name": item.get("name"), "arguments": item.get("arguments") or {}},
                sort_keys=True,
                default=str,
            )
            if item_signature == signature and item.get("result") is not None:
                return True
        return False

    def _repair_if_needed(
        self,
        state: Tau3EnsembleState,
        action: Action,
        issues: list[str],
        generation_text: str,
    ) -> tuple[Action, Optional[GenerationResult], list[str]]:
        repair_reasons = list(issues)
        content = action.kwargs.get(RESPOND_ACTION_FIELD_NAME) if action.name == RESPOND_ACTION_NAME else ""
        if action.name == RESPOND_ACTION_NAME and (not isinstance(content, str) or _looks_internal(content)):
            repair_reasons.append("assistant response contains private/internal reasoning or invalid text")
        if action.name.startswith("transfer_to_human") and state.step < self.min_transfer_step:
            repair_reasons.append("transfer proposed before ordinary tools or clarification were exhausted")
        if self._has_duplicate_tool_call(state, action):
            repair_reasons.append("duplicate tool call with identical arguments already has a result")

        if not repair_reasons:
            return action, None, issues

        repair_prompt = [
            {
                "role": "system",
                "content": (
                    "Repair the previous model output for a tau3-bench customer-service agent. "
                    "Return exactly one valid next action as JSON: "
                    "{\"name\":\"tool_name\",\"arguments\":{...}} or "
                    "{\"name\":\"respond\",\"arguments\":{\"content\":\"customer-facing text\"}}. "
                    "Do not include analysis, commentary, markdown, hidden state, or tool results. "
                    "Use only argument values visible in the conversation, policy, or previous tool results. "
                    "Ask the user when required information is missing. "
                    "Do not transfer while a visible tool or clarification can make progress. "
                    "Do not repeat an identical tool call whose result is already provided."
                ),
            },
            {
                "role": "user",
                "content": short_json(
                    {
                        "policy": self.domain_policy,
                        "available_tools": self.compact_tools,
                        "conversation": _transcript(state.messages),
                        "tool_call_history": self._tool_call_history(state.messages),
                        "bad_output": generation_text,
                        "repair_reasons": repair_reasons,
                    },
                    limit=22000,
                ),
            },
        ]
        repaired = self.bank.generate(
            self.bank.recovery_adapter,
            repair_prompt,
            tools=self.openai_tools,
            max_new_tokens=self.max_repair_tokens,
            temperature=0.0,
        )
        repaired_action = parse_action_text(repaired.text)
        repaired_action, repaired_issues = self.gate.validate(repaired_action)
        repaired_issues.extend(self._unsupported_argument_issues(state, repaired_action))
        repaired_content = (
            repaired_action.kwargs.get(RESPOND_ACTION_FIELD_NAME)
            if repaired_action.name == RESPOND_ACTION_NAME
            else ""
        )
        if repaired_action.name.startswith("transfer_to_human") and state.step < self.min_transfer_step:
            repaired_issues.append("repair still proposed early transfer")
        if self._has_duplicate_tool_call(state, repaired_action):
            repaired_issues.append("repair still repeated an identical completed tool call")
        if repaired_action.name == RESPOND_ACTION_NAME and (
            not isinstance(repaired_content, str) or _looks_internal(repaired_content)
        ):
            repaired_issues.append("repair still contains private/internal text")
        if repaired_issues:
            return action, repaired, repaired_issues
        return repaired_action, repaired, []

    def _unsupported_argument_issues(self, state: Tau3EnsembleState, action: Action) -> list[str]:
        if action.name == RESPOND_ACTION_NAME or action.name not in self.gate.tools_by_name:
            return []
        visible_parts = [
            self.domain_policy,
            _transcript(state.messages, limit=40),
            short_json(self.compact_tools, limit=12000),
        ]
        visible = _grounding_key(" ".join(visible_parts))
        if not visible:
            return []

        fn = self.gate.tools_by_name[action.name].get("function", {})
        props = (fn.get("parameters", {}) or {}).get("properties", {}) or {}
        issues: list[str] = []
        free_text_fields = {
            "content",
            "message",
            "response",
            "summary",
            "description",
            "reason",
            "note",
            "notes",
            "comment",
            "comments",
        }
        for field, value in action.kwargs.items():
            if field in free_text_fields:
                continue
            schema = props.get(field, {}) or {}
            if "enum" in schema:
                continue
            if isinstance(value, (dict, list, bool, int, float)) or value is None:
                continue
            raw_value = str(value).strip()
            key = _grounding_key(raw_value)
            if len(key) < 3:
                continue
            candidates = {key}
            if raw_value.startswith("#"):
                candidates.add(_grounding_key(raw_value[1:]))
            if not any(candidate and candidate in visible for candidate in candidates):
                issues.append(f"unsupported invented arg: {field}")
        return issues


def create_tau3_ensemble_agent(tools, domain_policy, **kwargs):
    return Tau3EnsembleAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "local-gptoss-ensemble",
        llm_args=kwargs.get("llm_args") or {},
    )
