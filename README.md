# Goal-Graph Tool Binding for LLM Tool Use

This repository implements a goal-graph runtime pipeline for reliable LLM tool use on BFCL and API-Bank style benchmarks.

The core idea is simple: **do not trust the LLM to directly produce final executable tool calls**. Instead, GPT-OSS is used for semantic understanding, while deterministic Python components perform tool binding, argument grounding, verification, and compilation.

## Pipeline Overview

```text
User request + tool schemas
        |
        v
GPT-OSS semantic frame
        |
        v
Deterministic tool binder
        |
        v
Optional call skeleton recovery
        |
        v
Goal-graph runtime verifier
        |
        v
Compiled tool calls
        |
        v
BFCL / API-Bank adapter
