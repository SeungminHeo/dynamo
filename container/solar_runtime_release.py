# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare or publish a personal-fork Solar wheel release. Requires Python 3.11+ and gh."""

import argparse
import base64
import hashlib
import json
import subprocess
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

REPO = "SeungminHeo/dynamo"
WORKFLOW = ".github/workflows/solar-runtime-wheel.yml"


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True)


def source_file(path: str, commit: str) -> str:
    payload = json.loads(gh("api", f"repos/{REPO}/contents/{path}?ref={commit}"))
    return base64.b64decode(payload["content"]).decode()


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(run_id: int, output: Path) -> tuple[str, str, list[Path]]:
    run = json.loads(gh("api", f"repos/{REPO}/actions/runs/{run_id}"))
    if run["conclusion"] != "success" or run["path"] != WORKFLOW:
        raise ValueError(
            "Only successful Solar runtime wheel workflow runs can be released"
        )
    if run["head_repository"]["full_name"] != REPO:
        raise ValueError("The build must originate in the personal Dynamo fork")
    commit = run["head_sha"]
    artifact = f"solar-runtime-linux-x86_64-{commit}"
    output.mkdir(parents=True, exist_ok=False)
    gh(
        "run",
        "download",
        str(run_id),
        "--repo",
        REPO,
        "--name",
        artifact,
        "--dir",
        str(output),
    )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one wheel, found {len(wheels)}")
    wheel = wheels[0]
    checksum = sha256(wheel)
    recorded = (output / "SHA256SUMS").read_text().splitlines()
    if recorded != [f"{checksum}  dist/{wheel.name}"]:
        raise ValueError("Wheel does not match the build's SHA256SUMS")
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        wheel_name = metadata_name.removesuffix("METADATA") + "WHEEL"
        tags = Parser().parsestr(archive.read(wheel_name).decode()).get_all("Tag")
    if metadata["Name"] != "ai-dynamo-runtime" or metadata["Version"] != "1.6.1":
        raise ValueError("Expected an ai-dynamo-runtime 1.6.1 wheel")
    if not tags or not all(
        tag.startswith("cp310-abi3-manylinux_2_28_x86_64") for tag in tags
    ):
        raise ValueError(f"Unexpected wheel compatibility tags: {tags}")
    manifest = tomllib.loads(source_file("Cargo.toml", commit))
    cargo_config = tomllib.loads(source_file(".cargo/config.toml", commit))
    workflow = source_file(WORKFLOW, commit)
    commands = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("maturin build ")
    ]
    if len(commands) != 1:
        raise ValueError(
            "Expected exactly one wheel build command in the source workflow"
        )
    tag = f"solar-runtime-1.6.1-{commit[:12]}"
    download = f"https://github.com/{REPO}/releases/download/{tag}/{wheel.name}"
    info = {
        "repository": REPO,
        "dynamo_commit": commit,
        "frontend_dependencies": manifest["patch"]["crates-io"],
        "workflow_run": run["html_url"],
        "workflow_source": f"https://github.com/{REPO}/blob/{commit}/{WORKFLOW}",
        "build_command": commands[0],
        "rustflags": cargo_config["target"]["x86_64-unknown-linux-gnu"]["rustflags"],
        "cargo_environment": cargo_config["env"],
        "wheel": wheel.name,
        "wheel_version": metadata["Version"],
        "wheel_tags": tags,
        "wheel_sha256": checksum,
        "wheel_url": download,
        "release_tag": tag,
    }
    info_path = output / "build-info.json"
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    requirements = output / "solar-runtime.requirements.txt"
    requirements.write_text(
        f"ai-dynamo-runtime @ {download} --hash=sha256:{checksum}\n"
    )
    # Release checksums use basenames, so sha256sum -c works in a download directory.
    sums = output / "SHA256SUMS"
    sums.write_text(
        f"{checksum}  {wheel.name}\n{sha256(info_path)}  {info_path.name}\n"
        f"{sha256(requirements)}  {requirements.name}\n"
    )
    notes = output / "release-notes.md"
    notes.write_text(
        f"Downstream Solar Open2 runtime for Dynamo 1.6.1. Source: `{commit}`.\n\n"
        "Includes Solar reasoning/tool parsing and the backend repetition finish reason. "
        "Linux x86_64, built with Python 3.12; cp310-abi3 is compatible with Python 3.12.\n\n"
        "Install this exact wheel after ai-dynamo==1.6.1, preserving the engine packages. "
        "Verify SHA256SUMS first. Package version alone does not identify this custom build.\n\n"
        f"Build and CPU import/registration check: {run['html_url']}. GPU validation is not included.\n"
    )
    print(json.dumps(info, indent=2))
    return tag, commit, [wheel, sums, info_path, requirements]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=int)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for verified assets"
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Create a prerelease in SeungminHeo/dynamo",
    )
    args = parser.parse_args()
    tag, commit, assets = prepare(args.run_id, args.output)
    if args.publish:
        # gh release create fails if a release already exists; never use upload --clobber.
        print(
            gh(
                "release",
                "create",
                tag,
                *[str(asset) for asset in assets],
                "--repo",
                REPO,
                "--target",
                commit,
                "--prerelease",
                "--latest=false",
                "--title",
                f"Solar runtime 1.6.1 ({commit[:12]})",
                "--notes-file",
                str(args.output / "release-notes.md"),
            )
        )


if __name__ == "__main__":
    main()
