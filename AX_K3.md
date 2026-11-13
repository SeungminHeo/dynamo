<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K3 `ax_k3` downstream maintenance

Internal-only: not intended for upstream. The parser lives in the personal
frontend-crates fork. On `axiom/release/1.7.0` the pin is branch
`ax-k3-unified-0.10.2` (commit `6f884824`, based on tag `dynamo-parsers-v2-v0.10.2`,
which `release/1.7.0` pins); `axiom/main` still pins `ax-k3-unified-0.9.6`. It is pinned by commit as a `dynamo-parsers-v2`
patch in both Cargo roots (`Cargo.toml` and `lib/bindings/python/Cargo.toml`)
and both lockfiles.

## Configuration

Rust frontend with `--dyn-chat-processor dynamo`, and either
`--dyn-tool-call-parser ax_k3 --dyn-reasoning-parser ax_k3` or
`--dyn-tool-call-parser ax_k3 --dyn-reasoning-parser deepseek_r1` (the same
unified family). Aliases: `axk3`, `ax-k3`. `ax_k3` is a v2 unified family, so
`DYN_PARSER_VERSION=1` is not supported.

## Behaviour

The specification is the model team's `axk3_tool_parser.py` with
`call_group_policy="all"`.

- Reasoning: the template opens `<think>`; only the first `</think>` ends it,
  and tool markup inside the thought stays reasoning. A prompt that already ends
  with `</think>` starts in the answer. Requests cannot turn thinking off.
- Calls: `<tool_call>NAME<|parameter|>KEY\nVALUE<|/parameter|></tool_call>`.
  Every closed, named call is executed. Each is streamed whole (name and
  complete arguments) as soon as its `</tool_call>` arrives. Text between calls
  is content; whitespace-only gaps are dropped.
- Values: string-typed parameters are raw text. Other values keep the model's
  spelling (`1.50` stays `1.50`) with `json.dumps` separators.
- Malformed output:
  - A key line missing its newline is repaired from the schema or dropped.
  - A value missing `<|/parameter|>` runs to `</tool_call>`.
  - A call without `</tool_call>` is returned as content and is never executed.
- `tool_choice` required/named use guided JSON. With `none`, markup never
  becomes a call. The reference's logit ban on `<tool_call>` is not ported.
- Not ported: the history re-render hook `decode_arguments_verbatim`.

## Validation

```sh
# frontend-crates fork
cargo test -p dynamo-parsers-v2 --lib ax_k3
# 5516 fuzzed cases instead of the checked-in 1216
AX_K3_REFERENCE_CASES=<cases.json> cargo test -p dynamo-parsers-v2 --lib reference_cases
# this repository
cargo test --locked -p dynamo-llm --no-default-features --lib preprocessor::ax_k3_tests
```

`parsers/v2/tests/data/ax_k3/gen_reference_cases.py` regenerates the
differential fixture from the reference parser, with vLLM stubbed. Its
`AX_K3_POLICY` setting defaults to `all`.

When Dynamo's `dynamo-parsers-v2` pin moves:
1. Rebase the fork branch onto the matching tag.
2. Rerun the differential.
3. Update the rev in both Cargo roots.

## Build

The `Axiom runtime wheel (personal fork)` workflow builds the runtime wheel by
`workflow_dispatch` on this branch. It checks that `ax_k3` is registered and
records the parser-v2 pin in `build-info.json`.

`axiom/release/1.7.0` builds a 1.7.0 wheel (a push to that branch starts the
workflow). Install it only into an image whose `ai-dynamo` Python package is
1.7.0; never into a 1.6.x image.
