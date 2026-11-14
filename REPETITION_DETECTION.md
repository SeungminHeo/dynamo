<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend repetition detection (Axiom downstream)

Internal-only: not intended for upstream. The frontend stops a choice whose generated
token IDs end in one N-gram repeated back to back, and reports finish reason
`repetition`. It runs in the Rust `Decoder` (`lib/llm/src/backend/repetition.rs`), so it
works the same for vLLM, SGLang, and TensorRT-LLM workers, and its stop cancels the
upstream engine request, which frees that request's decode slot and KV cache.

## Configuration

Set `DYN_REPETITION_DETECTION` on the **frontend**. It is read once at process start.
Unset, empty, or invalid leaves detection off (invalid values log an error).

One rule, the same shape as vLLM's `SamplingParams.repetition_detection`:

```sh
DYN_REPETITION_DETECTION='{"min_pattern_size": 8, "max_pattern_size": 256, "min_count": 4}'
```

Or a list of rules, for example a stricter count for short patterns:

```sh
DYN_REPETITION_DETECTION='[{"max_pattern_size": 7, "min_count": 32},
                           {"min_pattern_size": 8, "max_pattern_size": 256, "min_count": 4}]'
```

| Field | Meaning |
|---|---|
| `min_pattern_size` | Smallest pattern length in tokens; `0` or omitted means 1. |
| `max_pattern_size` | Largest pattern length, 1 to 4096. |
| `min_count` | Back-to-back occurrences that stop the choice, at least 2. |

A choice stops when, for some pattern length `L` covered by a rule, its last
`L * min_count` generated tokens are its last `L` tokens repeated `min_count` times. With
several rules covering the same `L`, the smallest `min_count` applies. Clients cannot
change or disable the rule per request.

## Behaviour

- Same gate as other stop conditions: no stop until `min_tokens` have been generated.
  An explicit stop token or stop string on the same token keeps its own finish reason.
- The stopping token's text is emitted; nothing after it is.
- Per-token cost is linear in the largest `max_pattern_size`.
- Finish reason `repetition` is treated as truncation, like `length`: collected
  (non-streaming) responses drop incomplete tool calls and incomplete structured output.
  Streaming clients have already received the deltas, exactly as with `length`, and must
  treat a `repetition` finish as an interrupted turn. Anthropic Messages reports
  `end_turn`.
- A migrated request restarts detection from the new attempt's generated tokens.

## Coverage

Covered: the default `--dyn-chat-processor dynamo` path when workers stream token IDs
(vLLM, SGLang with `skip_tokenizer_init`, TensorRT-LLM).

Not covered:
- `--dyn-chat-processor vllm|sglang` (the Python processors bypass the Rust `Decoder`).
- Chunks that carry only engine-decoded text without token IDs.

## Tuning

Short patterns repeat legitimately in markdown table rules, code indentation, and JSON
arrays of identical records. Start with long patterns and a moderate count, replay
representative traffic, and inspect responses that end with `repetition` before
lowering either bound.
