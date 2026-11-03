<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Solar Open2 on Dynamo 1.6.1

Downstream-only implementation tracked in [SeungminHeo/dynamo#1](https://github.com/SeungminHeo/dynamo/issues/1).
The parser fork is pinned by immutable commit in both Cargo roots and lockfiles.
The Python bindings are outside the workspace, so both patches are required.

Configure the Rust frontend with `--dyn-chat-processor dynamo`,
`--dyn-reasoning-parser solar_open2` and `--dyn-tool-call-parser solar_open2`.
The worker must register the intended Solar Jinja template. Use the default
parser-version selection or `DYN_PARSER_VERSION=1`; Solar is a legacy parser,
not a v2/unified family. Keep the worker's Solar parser configuration for its
native guided-decoding gate.

- Missing effort defaults to high; medium/high start in reasoning. Other efforts
  (including low/none) return content directly. `enable_thinking` does not
  override the Solar template. A top-level effort overrides template kwargs.
- `<|think:start|>` / `<|think:end|>` split reasoning from content. Native tool
  calls can also close reasoning when the think-end marker is absent.
- Special tokens survive detokenization so the parser can see its delimiters.
- Native tool arguments are schema-typed, with string fallback on failed conversion.
- Engine `finish_reason: "repetition"` is accepted and preserved in chat/text
  streaming and collected responses. This does not implement repetition detection.
- Tool calls are withheld until a backend `stop`. Truncated, repetition-terminated, cancelled, errored,
  or unterminated responses do not expose executable calls, even if an earlier
  call was complete. Reasoning and pre-call prose still stream.
- Guided required/named calls retain reasoning even when the entire completion
  arrives in one chunk; bare guided JSON is also accepted.
- `reasoning_tokens` uses Dynamo's existing chunk-granular estimate. Low/none
  reports zero; a positive backend count remains authoritative. A chunk mixing
  reasoning and content can overcount; this is not an exact token-boundary count.

## Prompt compatibility

The supplied Solar template was rendered through Rust `dynamo-renderer` 8.0.3
and the Python Transformers Jinja environment. Tool definitions, nested tool
history arguments, JSON schema, and JSON object instructions match byte-for-byte
when supplied the same template inputs.

Dynamo 1.6.1's Python `convert_tools()` adds `strict: false` when the client omitted
it. In the tool probe this alone changes 194 tokens to 199. Rust keeps omission,
matching direct template rendering of the request; this implementation deliberately
does not add the Python-only default to the prompt.

A top-level `response_format` selects guided decoding, and is not automatically
injected into this Jinja template by either frontend. To opt into the template's
output-format instruction, also supply it in `chat_template_kwargs.response_format`.

## Build and installation

The personal-fork `Solar runtime wheel` workflow builds on CPU in a Linux
manylinux_2_28 x86_64 container using Python 3.12. The output has the existing
`cp310-abi3` ABI tag and is compatible with Python 3.12. NIXL uses its existing
runtime-resolved fallback when native development libraries are absent.
The workflow verifies both parser registrations and uploads the wheel and SHA256.
It does not deploy a model or start GPU workloads.

For an existing image that already installs `ai-dynamo==1.6.1` and its matching
runtime, add this layer after that installation (retain its existing Python patches):

```dockerfile
COPY ai_dynamo_runtime-1.6.1-*.whl /tmp/solar-wheel/
RUN python3 -m pip install --no-deps --force-reinstall /tmp/solar-wheel/*.whl \
    && rm -rf /tmp/solar-wheel
```

Keep the worker image and frontend on 1.6.1 for the initial validation. Do not
install a main-branch wheel into a 1.6.1 Python image.

```sh
cargo test --locked -p dynamo-llm --no-default-features --lib preprocessor::solar::tests
cargo test --locked -p dynamo-llm --no-default-features --test test_reasoning_parser
cargo clippy --locked --no-deps -p dynamo-llm --no-default-features --all-targets -- -D warnings
```

Local validation: 989 parser tests and 186 protocol tests pass. The Dynamo
Solar/finish-reason suite passes 76 tests, including collected and streaming
responses. The full macOS Dynamo library run passes 3,122 tests with 13 ignored;
a discovery timing failure passes in isolation. A gRPC metrics-bind shutdown
test still times out in isolation and in the preserved pre-repetition test binary.
Its startup/shutdown implementation is unchanged by this branch.

GPU acceptance remains a separate step: use an isolated deployment, then compare
prompt tokens and valid JSON on the existing workload, reasoning/tool separation,
required/named tool calls, and interrupted outputs. Body byte equality is not a
reliable model parity metric under different batching.
