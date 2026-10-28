# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Exercise host-side GPU setup with Docker mocked; no GPU or network needed."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "device_flags,expected",
    [
        (None, ["--device", "/dev/dri"]),
        ("--device /dev/dri/renderD128", ["--device", "/dev/dri/renderD128"]),
        (
            "--device /dev/dri/renderD128\n--device /dev/dri/card0\n",
            ["--device", "/dev/dri/renderD128", "--device", "/dev/dri/card0"],
        ),
        (
            "--device\n/dev/dri/renderD128\n--device\n/dev/dri/card0",
            ["--device", "/dev/dri/renderD128", "--device", "/dev/dri/card0"],
        ),
        (
            "\n--device\t/dev/dri/renderD128\n\n--device /dev/dri/card0",
            ["--device", "/dev/dri/renderD128", "--device", "/dev/dri/card0"],
        ),
    ],
)
def test_gpu_setup_preserves_device_arguments(tmp_path, device_flags, expected):
    source = "flydsl-test"
    workspace = tmp_path / "workspace with spaces"
    (workspace / source).mkdir(parents=True)
    wheels = workspace / f"{source}-wheels"
    for role, files in {
        "pr": ["flydsl-0.1.0-test.whl", "fly-opt"],
        "base": ["flydsl-0.1.0-test.whl", "fly-opt"],
        "tools": ["FileCheck"],
    }.items():
        directory = wheels / role
        directory.mkdir(parents=True)
        manifest = []
        for filename in files:
            data = filename.encode()
            (directory / filename).write_bytes(data)
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {filename}\n")
        (directory / "SHA256SUMS").write_text("".join(manifest))

    device_file = tmp_path / "devices"
    if device_flags is not None:
        device_file.write_text(device_flags)
    # Bind the external podinfo file to the fixture without touching /etc.
    helper = Path(__file__).resolve().parents[2] / "scripts/setup_ci_gpu.sh"
    script = tmp_path / "setup_ci_gpu.sh"
    script.write_text(helper.read_text().replace("/etc/podinfo/gha-render-devices", str(device_file)))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['MOCK_DOCKER_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if any('rocm-sdk path --root' in arg for arg in sys.argv):\n"
        "    print('/opt/rocm')\n"
    )
    docker.chmod(0o755)
    log = tmp_path / "docker.jsonl"
    github_env = tmp_path / "github.env"
    container = "flydsl-test-42-1-linux-flydsl-mi325-1"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "MOCK_DOCKER_LOG": str(log),
        "GITHUB_WORKSPACE": str(workspace),
        "GITHUB_ENV": str(github_env),
        "CI_SOURCE_DIR": source,
        "CI_CONTAINER": container,
        "CI_PREPARE_AITER": "false",
        "CI_HIP_VISIBLE_DEVICES": "",
        "DOCKER_IMAGE": "mock-ci-image",
    }
    subprocess.run(["bash", str(script)], env=env, check=True, text=True, capture_output=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == ["rm", "-f", container]
    run = calls[1]
    assert run[0] == "run"
    device_start = run.index("--device=/dev/kfd") + 1
    assert run[device_start : run.index("-v")] == expected
    assert run[run.index("--name") + 1] == container
    assert f"{workspace / source}:/flydsl-test" in run
    assert f"{wheels}:/flydsl-ci-wheels:ro" in run
    assert github_env.read_text() == "ROCM_PATH=/opt/rocm\n"
