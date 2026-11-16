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
- Reasoning follows UpstageAI/vllm's public `solar_open2_reasoning_parser.py`
  (tag `v0.22.0-solar-open2`), ported as `SolarOpen2ReasoningParser` in the
  parser fork. Only `<|think:end|>` ends the thought; everything after it is
  content. Text written while thinking, such as `<tool_call>` or
  `<|tool_call:start|>`, stays reasoning.
  - A complete tool-call block written inside the thought moves to content
    after the answer, so it still runs. The stream holds the thought from its
    first `<|tool_call:start|>` until `<|think:end|>`, and the frontend
    finalizes the parser at the end of the stream to deliver moved blocks.
  - A thought cut off before `<|think:end|>` stays reasoning. The reference's
    non-streaming path returns it as content.
- Special tokens survive detokenization so the parser can see its delimiters.
- Native tool arguments are schema-typed, with string fallback on failed conversion.
- Engine `finish_reason: "repetition"` is accepted and reported to clients as a
  truncation (`length`, Responses `max_output_tokens`, Anthropic `max_tokens`);
  `nvext.detailed_finish_reason` keeps `repetition` on request. Frontend detection
  is separate (`DYN_REPETITION_DETECTION`, see `REPETITION_DETECTION.md`); it also
  stops a loop inside a streaming tool argument.
- Native (`auto`) tool calls stream as the model writes them, following the
  reference streaming parser: the name when its line ends, typed values when
  they are complete, and `}` at `<|tool_call:end|>`. String values also stream
  while they are written (beyond the reference, which waits for
  `<|tool_arg:end|>`), so a long argument such as a file body never holds the
  stream. A string value that could still be the literal `null` waits until it
  cannot. Concatenated arguments equal `solar::parse` output; a key repeated in
  one call is streamed twice (the parser keeps the last value).
- Executability is carried by the terminal reason. `tool_calls` is reported only
  after a real backend `stop` with at least one closed call. Truncated
  (`length`), repetition-terminated, cancelled, errored, or unterminated
  responses keep their own reason; streamed deltas are not revoked, and
  clients must not execute calls from such a turn. Collected (non-streaming)
  responses drop every call in that case. A call the grammar rejects after its
  name was sent is never closed, so its arguments never form valid JSON, and
  collected responses drop it. Prose after the first call is dropped.
- Guided `required`/named JSON stays on the shared jail and follows
  `DYN_ENABLE_GUIDED_TOOL_STREAMING`, with the same terminal rule.
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

## Wheel releases and engine images

Keep the parser/protocol source in the frontend-crates fork, the runtime wheel
in this Dynamo fork, and engine changes in their SGLang/vLLM forks. For the same
supported Linux/Python/CUDA environment, both engine images consume the same
verified runtime wheel. Keep their engine packages pinned independently. The
current build targets Linux x86_64 (the repository's x86-64-v3/AVX2 target),
Python 3.12 compatibility and CUDA 13.
The wheel build does not validate engine GPU execution or native NIXL compatibility.

Actions artifacts are temporary build outputs. After a successful wheel run,
prepare a release with Python 3.11+ and an authenticated GitHub CLI:

```sh
python3 container/solar_runtime_release.py RUN_ID --output /tmp/solar-release-RUN_ID
# To prepare and publish in one invocation, use a fresh output directory:
python3 container/solar_runtime_release.py RUN_ID --output /tmp/solar-publish-RUN_ID --publish
```

The helper checks the workflow identity, successful status, wheel checksum,
package version and compatibility tags. It creates a personal-fork prerelease
named `solar-runtime-1.6.1-<source-commit>` with the wheel, `SHA256SUMS`,
`build-info.json`, and `solar-runtime.requirements.txt`. The manifest records source pins, build command, run URL and
an exact wheel download URL. Existing releases/assets are never overwritten.
No PyPI publication is involved.

The package version remains 1.6.1 for this downstream build. Do not identify it
by `pip freeze` alone. Pin `wheel_url` and `wheel_sha256` from `build-info.json`
in the image build inputs, check the checksum, then install the local wheel
with `--no-deps --force-reinstall` after all stock Dynamo package installation.
A later install of the stock runtime would replace the custom parser.

Keep the deployment's engine commit, final image digest, Dynamo source commit,
wheel SHA256 and chat-template SHA256 together in its deployment lock/manifest.
The frontend and worker images should use the same selected Dynamo runtime build.
Rollback by selecting the previous image digest; keep old wheel releases intact.

The release's requirements file pins the wheel URL and hash. Copy the selected
file into either engine image's build context and install it after the stock
Dynamo installation:

```dockerfile
COPY solar-runtime.requirements.txt /tmp/solar-runtime.requirements.txt
RUN python3 -m pip install --require-hashes --no-deps --force-reinstall \
    -r /tmp/solar-runtime.requirements.txt
```

Commit that requirements file alongside the engine image/deployment inputs.
Changing it selects a new runtime build without changing the SGLang/vLLM package.
