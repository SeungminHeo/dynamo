<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend repetition detection (Axiom downstream)

Internal-only: not intended for upstream. The frontend stops a choice whose generated
token IDs end in one N-gram repeated back to back, with the internal finish reason
`repetition` (clients see `length`; see Behaviour). It runs in the Rust `Decoder` (`lib/llm/src/backend/repetition.rs`), so it
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
- A repetition stop is a truncation. OpenAI clients see `finish_reason: "length"`,
  Responses `incomplete_details.reason: "max_output_tokens"`, and Anthropic Messages
  `stop_reason: "max_tokens"`. OpenAI has no repetition value, and gateways such as
  LiteLLM (verified with 1.106.6) rewrite unknown reasons to `stop`, which would present
  a cut-off turn as complete. Collected (non-streaming) responses drop incomplete tool
  calls and incomplete structured output; streaming clients have already received the
  deltas and must treat the turn as interrupted.
- The cause stays visible to the operator: the request trace keeps the backend reason
  `repetition`, the frontend logs the request ID at INFO, and a client that asks with
  `nvext.extra_fields: ["detailed_finish_reason"]` gets `nvext.detailed_finish_reason:
  "repetition"`. An engine that reports `repetition` itself is handled the same way.
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
representative traffic, and inspect requests whose backend reason is `repetition` (request
trace or the INFO log) before lowering either bound.
