#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Mxfp4Config,
    Trainer,
    TrainingArguments,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def common_prefix_len(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def apply_chat_template_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "conversation": messages,
        "tokenize": True,
        "add_generation_prompt": False,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        token_ids = tokenizer.apply_chat_template(**kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        token_ids = tokenizer.apply_chat_template(**kwargs)
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def fallback_prompt(row: dict[str, Any]) -> str:
    if "text" in row:
        return str(row["text"])
    parts = []
    for message in row.get("messages") or []:
        role = message.get("role", "message")
        content = message.get("content") or ""
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def assistant_train_flags(row: dict[str, Any], messages: list[dict[str, Any]]) -> list[bool]:
    raw_flags = row.get("loss_mask")
    if isinstance(raw_flags, list) and len(raw_flags) == len(messages):
        return [bool(value) for value in raw_flags]
    return [message.get("role") == "assistant" for message in messages]


def tokenize_chat_row(
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_length: int,
    truncation_side: str,
) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        input_ids = tokenizer(fallback_prompt(row), add_special_tokens=True).input_ids
        labels = list(input_ids)
    else:
        tools = row.get("tools") if isinstance(row.get("tools"), list) else None
        input_ids = apply_chat_template_ids(tokenizer, messages, tools)
        labels = [-100] * len(input_ids)
        flags = assistant_train_flags(row, messages)

        for index, message in enumerate(messages):
            if message.get("role") != "assistant" or not flags[index]:
                continue
            prefix_ids = apply_chat_template_ids(tokenizer, messages[:index], tools)
            upto_ids = apply_chat_template_ids(tokenizer, messages[: index + 1], tools)
            if len(upto_ids) > len(input_ids):
                continue
            if input_ids[: len(upto_ids)] == upto_ids:
                start = common_prefix_len(prefix_ids, upto_ids)
                end = len(upto_ids)
            else:
                start = min(len(prefix_ids), len(input_ids))
                end = min(len(upto_ids), len(input_ids))
            if start < end:
                labels[start:end] = input_ids[start:end]

    if len(input_ids) > max_seq_length:
        if truncation_side == "left":
            input_ids = input_ids[-max_seq_length:]
            labels = labels[-max_seq_length:]
        else:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "label_tokens": sum(1 for label in labels if label != -100),
    }


def build_dataset(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_seq_length: int,
    truncation_side: str,
    min_label_tokens: int,
    max_samples: int | None,
) -> tuple[Dataset, dict[str, Any]]:
    if max_samples:
        rows = rows[:max_samples]
    tokenized = [
        tokenize_chat_row(tokenizer, row, max_seq_length, truncation_side)
        for row in rows
    ]
    kept = [row for row in tokenized if row["label_tokens"] >= min_label_tokens]
    lengths = [len(row["input_ids"]) for row in kept]
    label_counts = [row.pop("label_tokens") for row in kept]
    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(rows) - len(kept),
        "max_seq_length": max_seq_length,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "avg_label_tokens": round(sum(label_counts) / len(label_counts), 2) if label_counts else 0,
    }
    return Dataset.from_list(kept), summary


@dataclass
class CausalLMDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            length = len(feature["input_ids"])
            pad_len = max_len - length
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def trainer_argument_names() -> set[str]:
    return set(inspect.signature(TrainingArguments.__init__).parameters)


def pick_precision() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    return torch.cuda.is_bf16_supported(), not torch.cuda.is_bf16_supported()


def model_load_kwargs(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    native_quantization = getattr(config, "quantization_config", None)
    quantization = None
    if args.tuning_method == "lora" and native_quantization is None and args.load_in_4bit:
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.tuning_method == "lora" and args.device_map:
        kwargs["device_map"] = args.device_map
    if quantization is not None:
        kwargs["quantization_config"] = quantization
        kwargs["dtype"] = "auto"
    elif args.dequantize_mxfp4 and "gpt-oss" in args.model.lower():
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return kwargs


def load_model(args: argparse.Namespace) -> Any:
    if "gpt-oss" in args.model.lower() and not torch.cuda.is_available() and not args.preprocess_only:
        raise RuntimeError(
            "GPT-OSS MXFP4 loading needs a working CUDA GPU. "
            f"torch={torch.__version__}, torch_cuda={torch.version.cuda}"
        )
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.output_router_logits and hasattr(config, "output_router_logits"):
        config.output_router_logits = True
    if hasattr(config, "use_cache"):
        config.use_cache = False

    kwargs = model_load_kwargs(args, config)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config, **kwargs)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.tuning_method == "lora":
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        if args.init_adapter:
            model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
            model.print_trainable_parameters()
            return model
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.sft.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/qwen3-14b-taskdecomp-lora"))
    parser.add_argument("--tuning-method", choices=["lora", "full"], default="lora")
    parser.add_argument("--init-adapter", default="", help="Optional PEFT adapter to continue training.")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-label-tokens", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--optim", default="")
    parser.add_argument("--dequantize-mxfp4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-router-logits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["all-linear"])
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", default="")
    parser.add_argument("--fsdp-use-orig-params", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fsdp-activation-checkpointing", action="store_true")
    parser.add_argument("--deepspeed")
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, train_summary = build_dataset(
        tokenizer,
        read_jsonl(args.train),
        args.max_seq_length,
        args.truncation_side,
        args.min_label_tokens,
        args.max_train_samples,
    )
    val_dataset, val_summary = build_dataset(
        tokenizer,
        read_jsonl(args.validation),
        args.max_seq_length,
        args.truncation_side,
        args.min_label_tokens,
        args.max_validation_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_summary = {"train": train_summary, "validation": val_summary}
    (args.output_dir / "preprocess_summary.json").write_text(
        json.dumps(preprocess_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(preprocess_summary, indent=2), flush=True)

    if args.preprocess_only:
        return
    if len(train_dataset) == 0:
        raise RuntimeError("No train examples with label tokens survived preprocessing.")

    model = load_model(args)
    bf16, fp16 = pick_precision()
    save_steps = args.save_steps if args.save_steps is not None else args.eval_steps
    optim = args.optim or ("adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch")

    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "report_to": ["wandb"] if os.getenv("WANDB_API_KEY") and os.getenv("WANDB_DISABLED") != "true" else [],
        "optim": optim,
    }
    if args.fsdp:
        training_kwargs["fsdp"] = args.fsdp
        fsdp_config: dict[str, Any] = {"use_orig_params": args.fsdp_use_orig_params}
        if args.fsdp_transformer_layer_cls_to_wrap:
            fsdp_config["transformer_layer_cls_to_wrap"] = args.fsdp_transformer_layer_cls_to_wrap
        if args.fsdp_activation_checkpointing:
            fsdp_config["activation_checkpointing"] = True
        training_kwargs["fsdp_config"] = fsdp_config
    if args.deepspeed:
        training_kwargs["deepspeed"] = args.deepspeed

    arg_names = trainer_argument_names()
    if "eval_strategy" not in arg_names and "evaluation_strategy" in arg_names:
        training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")
    training_args = TrainingArguments(
        **{key: value for key, value in training_kwargs.items() if key in arg_names}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) else None,
        data_collator=CausalLMDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
