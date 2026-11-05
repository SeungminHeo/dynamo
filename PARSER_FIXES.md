<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Downstream parser maintenance

Work is tracked only in the personal forks: [Solar and repetition](https://github.com/SeungminHeo/dynamo/issues/1), [ordinary JSON arrays](https://github.com/SeungminHeo/dynamo/issues/2).

`main` carries the fixes on the current upstream development base. The
`SeungminHeo/solar-rust-1.6.1` branch produces the separate Dynamo 1.6.1 runtime.
Each branch pins compatible parser and protocol crates from the same immutable
frontend-crates commit, in both Cargo manifests and lockfiles. Do not install a
development-main wheel into a 1.6.1 image.

## Ordinary JSON arrays

The legacy basic JSON parser previously consumed any valid JSON array even if
none of its elements described a tool call. `[5,0,8]` became empty content;
`result: [5,0,8]` lost the array. The affected shared configurations are `hermes`,
`qwen25`, `default`, `nemotron_deci`, `llama3_json`, `mistral`, `phi4`, and `jamba`.
The faulty source is identical in parser releases 8.1.0 and 10.0.7; this build
targets Dynamo 1.6.1, not 1.5.0.

Unmarked arrays producing zero calls now return the original text, including its
whitespace. Explicit tool wrappers and guided JSON retain their existing empty
result semantics. Real calls, raw argument bytes, required/named tool choice,
Solar native parsing, and engine `repetition` termination remain supported.

The compatibility fallback still accepts unmarked objects/arrays that actually
match the tool-call schema. For Hermes, a name-only object is also a call. Mixed
arrays can still yield their valid calls while skipping malformed entries. This
change does not make Hermes marker-only or validate names against requested
tools; those are separate policy changes.

Validation includes parser batch/finalize calls, every UTF-8 two-chunk split,
and Dynamo's postprocessor plus response aggregation with reasoning on/off,
tools absent/auto/none, and whole-response/character-at-a-time delivery. Before
the dependency update the Dynamo regression returns empty content; afterward
it preserves the input and `finish_reason: stop`. The four related Dynamo
integration targets pass (217 tests).

The forward port on upstream development commit `945f785059` passes 101
Solar/repetition/request-trace unit tests and 234 related integration tests.
It uses the current `qwen3` reasoning selector in the regression; the old
`qwen` alias is no longer accepted by main's selector validation.

An additional CPU audit checked five ordinary array forms across all 31 legacy
parser names (155 cases), using a native final envelope for Harmony. All passed
after the fix. This is not a GPU engine benchmark or a full conformance claim.

## Deployment

Rebuild `ai-dynamo-runtime` and replace the frontend image using the immutable
wheel URL and SHA-256 requirements file from its personal GitHub release.
Keep the SGLang/vLLM engine versions and existing Python overlays pinned. This
fix changes no worker transport or engine code; workers need no rebuild when
the frontend is deployed separately. A combined image must replace its runtime
wheel as well. Merely changing Python files does not replace the Rust parser.

After deployment, smoke-test ordinary arrays with tools enabled, with and
without streaming, alongside a real tagged tool call. CPU tests here do not
claim live deployment validation.
