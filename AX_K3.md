<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K3 `ax_k3` downstream maintenance

Internal-only: not intended for upstream. This branch is Dynamo
`release/1.6.1` plus the Axiom changes and `ax_k3`. The parser lives in the personal frontend-crates
fork, branch `ax-k3-unified-0.7.24` (based on tag `dynamo-parsers-v2-v0.7.24`,
the version 1.6.1 and 1.6.2 pin; a port of the 0.9.6 branch `ax-k3-unified-0.9.6` used by
`axiom/test-ax-k3-flash`). It is pinned by commit as a `dynamo-parsers-v2`
patch in both Cargo roots (`Cargo.toml` and `lib/bindings/python/Cargo.toml`)
and both lockfiles. The branch also carries the Axiom Solar Open2 changes
(see `SOLAR_OPEN2.md`).

## Configuration

Rust frontend with `--dyn-chat-processor dynamo`, and either
`--dyn-tool-call-parser ax_k3 --dyn-reasoning-parser ax_k3` or
`--dyn-tool-call-parser ax_k3 --dyn-reasoning-parser deepseek_r1` (the same
unified family). Aliases: `axk3`, `ax-k3`. On 1.6.x the default
`DYN_PARSER_VERSION` (auto) selects `ax_k3`, like DeepSeek V4.1; `2` also works.
`ax_k3` is a v2 unified family, so `DYN_PARSER_VERSION=1` is not supported.

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
# frontend-crates fork (branch ax-k3-unified-0.7.24)
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

This branch builds a 1.6.1 wheel. Install it only into a 1.6.1 image.
