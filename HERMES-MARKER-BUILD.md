# Axiom runtime build and branch maintenance

This fork requires `<tool_call>` for native Hermes tool extraction. Unmarked
JSON, including `{"name":"Alice"}` and tool-shaped arrays, remains content.
Explicit guided bare JSON remains supported. Tagged name-only calls retain
empty-argument recovery. Qwen25 keeps its previous behavior. This is a personal
policy change, separate from upstream frontend-crates PR #559.

This runtime also accepts `tools: []` when `tool_choice` is omitted, `none`, or
`auto`. Required/named choices still require actual tools (including tools
supplied through system messages). Strict schema validation is unchanged.
Tracked in personal-fork issue #3.

## Optional strict-schema migration mode

The 1.6.1 runtime backports upstream #17163 (`a0de541ea8`); `axiom/main` already
has that feature. To temporarily accept function schemas that do not meet the
OpenAI strict subset, set this on the **frontend** before process startup:

```dockerfile
ENV DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION=1
```

Or set the same environment variable in the frontend deployment. Unset/false
keeps strict function-schema prevalidation enabled. The setting is latched at
startup; restart the frontend after changing it. Request schema and strict flags
are preserved, not rewritten. Basic validation and backend constraints still
apply, and response_format validation is unchanged. This does not guarantee
OpenAI strict semantics for an incomplete schema. Remove the override after
clients migrate. Tracked in personal-fork issue #4.

## Maintained branches

- `axiom/main`: development patches on the upstream main lineage.
- `axiom/release/1.6.1`: the existing 1.6.1 patch stack; build 1.6.1 images from this line.
- Land shared fixes on `axiom/main`, then cherry-pick selected fixes with `git cherry-pick -x` onto the release line. Release-only adaptations stay on the release line and document why.
- Do not merge the entire development branch into a release branch or force-reset either maintained branch. Review upstream updates on the matching line.
- The old personal `main` and `SeungminHeo/solar-rust-1.6.1` branches are retained as migration references; future Axiom work uses the two branches above.
- Published `axiom-runtime-*` tags and wheel hashes remain immutable. Branch/documentation changes alone do not require rebuilding the published wheel.
- The workflow filename remains `solar-runtime-wheel.yml` for existing GitHub workflow dispatch compatibility; its display name and new artifact names use Axiom.

## Pinned sources

- Dynamo release branch: `axiom/release/1.6.1`
- Release parser/protocol source: `8870bba39bd875abaf266c3e02ac8cd1385ad4af`
- Dynamo `axiom/main` parser/protocol source: `dc397c2236ec16ad2cebf004ea1176c927f88ff7`
- Parser repository: https://github.com/SeungminHeo/frontend-crates

Both workspace and Python binding manifests and lockfiles pin the same source
within each branch. Do not replace these with registry-only dependencies or
reuse the earlier Solar wheel: that wheel does not contain this policy.

## Linux wheel via GitHub Actions

The personal-fork-only `solar-runtime-wheel.yml` builds on manylinux_2_28 x86_64
with Rust 1.96.1 and the CI runtime features. A push to the release branch starts
it automatically. Manual build (also supported on `axiom/main`):

```sh
gh workflow run solar-runtime-wheel.yml --repo SeungminHeo/dynamo --ref axiom/release/1.6.1
gh run list --repo SeungminHeo/dynamo --workflow solar-runtime-wheel.yml --limit 5
# After the selected run succeeds:
gh run download RUN_ID --repo SeungminHeo/dynamo --dir wheel-artifacts
```

The artifact contains the wheel, `SHA256SUMS`, and `build-info.json` with the
Dynamo commit, parser commit, and build run URL. Pin the run/commit and wheel
SHA256; the package version alone does not identify this custom build.

## Build without GitHub Actions

Use this fallback when GitHub Actions is unavailable.

On a Linux x86_64 builder with Rust 1.96.1, protoc 30.2, Python 3.12, clang,
OpenSSL/zlib development headers and `maturin[patchelf]==1.16.0` installed:

```sh
git clone --branch axiom/release/1.6.1 https://github.com/SeungminHeo/dynamo.git
cd dynamo
# Record/pin the exact revision before building.
git rev-parse HEAD
cd lib/bindings/python
CARGO_BUILD_JOBS=3 CUDARC_CUDA_VERSION=13000 \
CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_PROFILE_RELEASE_LTO=false \
maturin build --locked --release --interpreter python3.12 \
  --compatibility manylinux_2_28 \
  --features kv-indexer,slot-tracker,select-service,mm-routing,ais-forward-pass,request-trace-s3 \
  --out ../../../dist
cd ../../..
sha256sum dist/*.whl > dist/SHA256SUMS
```

Use the `quay.io/pypa/manylinux_2_28_x86_64` environment from the workflow for
portable manylinux artifacts. On a newer Linux host, auditwheel may reject the
requested compatibility; do not relabel an incompatible wheel. The workflow's
"Install CPU build tools" step contains the exact container setup commands.
For manually built wheels, copy `dist/` to a `wheel-artifacts/axiom-runtime-linux-x86_64-COMMIT/`
directory before using the Dockerfile snippet below; retain the commit alongside
the checksum. No CUDA device is needed for the wheel build.

## Frontend image

Use the wheel built from the matching Dynamo branch/version. For the 1.6.1
image, keep the existing `ai-dynamo==1.6.1` Python package and overlay this
runtime wheel in the final frontend image stage:

```dockerfile
COPY wheel-artifacts/axiom-runtime-linux-x86_64-*/ /tmp/custom-runtime/
RUN cd /tmp/custom-runtime && \
    sed 's@dist/@@' SHA256SUMS | sha256sum -c - && \
    python3 -m pip install --no-deps --force-reinstall ./ai_dynamo_runtime-*.whl && \
    rm -rf /tmp/custom-runtime
```

Retain the Dynamo Rust chat processor and Hermes parser selection. The patch
has no new CLI flag and does not require a worker rebuild. A frontend configured
with the vLLM/SGLang chat processor will use that engine's parser instead.
Do not force Unified/v2 for Hermes: this patch changes the legacy v1 parser.

Before rollout, test tools=auto ordinary JSON, tagged real calls, required and
named guided calls, with streaming both enabled and disabled. No GPU model or
live deployment qualification is implied by the local Rust regressions.
