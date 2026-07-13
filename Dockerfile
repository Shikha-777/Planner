FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /workspace/taskdecomp

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m pip install -e .

CMD ["/bin/bash"]

