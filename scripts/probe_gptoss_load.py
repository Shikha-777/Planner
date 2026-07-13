from __future__ import annotations

import os

import torch

from scripts.run_gptoss_capability_plan import load_model


def main() -> None:
    print("visible", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("count", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            index,
            torch.cuda.get_device_name(index),
            round(props.total_memory / 1024**3, 2),
            "GB",
        )
    model, _tokenizer = load_model("openai/gpt-oss-20b")
    print("loaded", type(model).__name__)


if __name__ == "__main__":
    main()
