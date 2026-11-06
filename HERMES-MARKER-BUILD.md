# Personal Hermes marker policy build

This fork requires `<tool_call>` for native Hermes tool extraction. Unmarked
JSON, including `{"name":"Alice"}` and tool-shaped arrays, remains content.
Explicit guided bare JSON remains supported. Tagged name-only calls retain
empty-argument recovery. Qwen25 keeps its previous behavior. This is a personal
policy change, separate from upstream frontend-crates PR #559.

## Pinned sources

- Dynamo release branch: `SeungminHeo/solar-rust-1.6.1`
- Release parser/protocol source: `8870bba39bd875abaf266c3e02ac8cd1385ad4af`
- Dynamo personal main parser/protocol source: `dc397c2236ec16ad2cebf004ea1176c927f88ff7`
- Parser repository: https://github.com/SeungminHeo/frontend-crates

Both workspace and Python binding manifests and lockfiles pin the same source
within each branch. Do not replace these with registry-only dependencies or
reuse the earlier Solar wheel: that wheel does not contain this policy.

## Linux wheel via GitHub Actions

The personal-fork-only `solar-runtime-wheel.yml` builds on manylinux_2_28 x86_64
with Rust 1.96.1 and the CI runtime features. A push to the release branch starts
it automatically. Manual build (also supported on personal `main`):

```sh
gh workflow run solar-runtime-wheel.yml --repo SeungminHeo/dynamo --ref SeungminHeo/solar-rust-1.6.1
gh run list --repo SeungminHeo/dynamo --workflow solar-runtime-wheel.yml --limit 5
# After the selected run succeeds:
gh run download RUN_ID --repo SeungminHeo/dynamo --dir wheel-artifacts
```

The artifact contains the wheel, `SHA256SUMS`, and `build-info.json` with the
Dynamo commit, parser commit, and build run URL. Pin the run/commit and wheel
SHA256; the package version alone does not identify this custom build.

## Frontend image

Use the wheel built from the matching Dynamo branch/version. For the 1.6.1
image, keep the existing `ai-dynamo==1.6.1` Python package and overlay this
runtime wheel in the final frontend image stage:

```dockerfile
COPY wheel-artifacts/solar-runtime-linux-x86_64-*/ /tmp/custom-runtime/
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
