# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_rocm_backend(monkeypatch):
    """Load the real backend module without requiring built MLIR bindings."""
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "python" / "flydsl"
    prefix = "_flydsl_rocm_resolver_test"

    packages = {
        prefix: package_root,
        f"{prefix}.compiler": package_root / "compiler",
        f"{prefix}.compiler.backends": package_root / "compiler" / "backends",
        f"{prefix}.runtime": package_root / "runtime",
        f"{prefix}.utils": package_root / "utils",
    }
    for name, path in packages.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)

    runtime_device = types.ModuleType(f"{prefix}.runtime.device")
    runtime_device.get_rocm_arch = lambda: "gfx000"
    runtime_device.get_warp_size = lambda _arch: 32
    monkeypatch.setitem(sys.modules, runtime_device.__name__, runtime_device)

    env = types.ModuleType(f"{prefix}.utils.env")
    monkeypatch.setitem(sys.modules, env.__name__, env)

    base_name = f"{prefix}.compiler.backends.base"
    base_spec = importlib.util.spec_from_file_location(base_name, package_root / "compiler" / "backends" / "base.py")
    base = importlib.util.module_from_spec(base_spec)
    monkeypatch.setitem(sys.modules, base_name, base)
    base_spec.loader.exec_module(base)

    module_name = f"{prefix}.compiler.backends.rocm"
    spec = importlib.util.spec_from_file_location(module_name, package_root / "compiler" / "backends" / "rocm.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rocm(monkeypatch):
    for name in ("FLYDSL_ROCM_TOOLKIT", "ROCM_PATH", "HIP_PATH"):
        monkeypatch.delenv(name, raising=False)
    return _load_rocm_backend(monkeypatch)


def _make_toolkit(root: Path) -> Path:
    (root / "llvm" / "bin").mkdir(parents=True)
    (root / "llvm" / "bin" / "ld.lld.exe").touch()
    (root / "amdgcn" / "bitcode").mkdir(parents=True)
    return root


def _make_split_sdk(monkeypatch, tmp_path) -> Path:
    """Point ROCM_PATH at a TheRock SDK with its toolkit under lib/llvm; returns that dir."""
    sdk = tmp_path / "sdk"
    llvm = sdk / "lib" / "llvm"
    (llvm / "bin").mkdir(parents=True)
    (llvm / "bin" / "ld.lld.exe").write_bytes(b"lld")
    (llvm / "amdgcn" / "bitcode").mkdir(parents=True)
    (llvm / "amdgcn" / "bitcode" / "ocml.bc").write_bytes(b"ocml")
    monkeypatch.setenv("ROCM_PATH", str(sdk))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    return llvm


def _stage_dir(tmp_path: Path, llvm: Path) -> Path:
    digest = hashlib.sha256(str(llvm).encode()).hexdigest()[:12]
    return tmp_path / "local" / "FlyDSL" / "rocm-toolkit-files" / digest


def test_direct_sdk_from_rocm_path(rocm, monkeypatch, tmp_path):
    sdk = _make_toolkit(tmp_path / "sdk")
    monkeypatch.setenv("ROCM_PATH", str(sdk))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    assert rocm._windows_rocm_toolkit() == sdk.resolve().as_posix()
    assert not (tmp_path / "local").exists()


def test_explicit_toolkit_overrides_discovery_and_requires_layout(rocm, monkeypatch, tmp_path):
    explicit = _make_toolkit(tmp_path / "explicit")
    monkeypatch.setenv("FLYDSL_ROCM_TOOLKIT", str(explicit))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "ignored"))

    assert rocm._windows_rocm_toolkit() == explicit.resolve().as_posix()

    monkeypatch.setenv("FLYDSL_ROCM_TOOLKIT", str(tmp_path / "incomplete"))
    with pytest.raises(RuntimeError, match="Invalid FLYDSL_ROCM_TOOLKIT"):
        rocm._windows_rocm_toolkit()


def test_split_sdk_stages_hardlinked_files_once(rocm, monkeypatch, tmp_path):
    llvm = _make_split_sdk(monkeypatch, tmp_path)
    stage = _stage_dir(tmp_path, llvm)

    assert rocm._windows_rocm_toolkit() == stage.as_posix()
    lld = stage / "llvm" / "bin" / "ld.lld.exe"
    bitcode = stage / "amdgcn" / "bitcode" / "ocml.bc"
    assert lld.samefile(llvm / "bin" / "ld.lld.exe")
    assert bitcode.samefile(llvm / "amdgcn" / "bitcode" / "ocml.bc")
    assert not any(p.is_symlink() for p in stage.rglob("*"))

    monkeypatch.setattr(rocm.os, "link", lambda *_a: pytest.fail("restaged a current file"))
    monkeypatch.setattr(rocm.shutil, "copy2", lambda *_a: pytest.fail("restaged a current file"))
    assert rocm._windows_rocm_toolkit() == stage.as_posix()


def test_split_sdk_restages_files_an_sdk_upgrade_replaced(rocm, monkeypatch, tmp_path):
    llvm = _make_split_sdk(monkeypatch, tmp_path)
    rocm._windows_rocm_toolkit()

    # pip replaces files rather than rewriting them in place.
    (llvm / "bin" / "ld.lld.exe").unlink()
    (llvm / "bin" / "ld.lld.exe").write_bytes(b"newer lld")

    stage = Path(rocm._windows_rocm_toolkit())
    assert (stage / "llvm" / "bin" / "ld.lld.exe").read_bytes() == b"newer lld"
    assert not list(stage.rglob("*.tmp"))


def test_split_sdk_copies_when_hardlinks_fail(rocm, monkeypatch, tmp_path):
    llvm = _make_split_sdk(monkeypatch, tmp_path)

    def no_link(*_args):
        raise OSError("cross-device link")

    monkeypatch.setattr(rocm.os, "link", no_link)
    stage = Path(rocm._windows_rocm_toolkit())

    lld = stage / "llvm" / "bin" / "ld.lld.exe"
    assert lld.read_bytes() == b"lld"
    assert not lld.samefile(llvm / "bin" / "ld.lld.exe")


def test_missing_sdk_has_actionable_error(rocm, monkeypatch, tmp_path):
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("HIP_PATH", str(tmp_path / "also-missing"))
    monkeypatch.setattr(rocm.sys, "path", [str(tmp_path / "empty-python-path")])

    with pytest.raises(RuntimeError, match="Could not find ld.lld.exe and AMDGPU bitcode"):
        rocm._windows_rocm_toolkit()


def test_undetected_arch_asks_for_arch(rocm, monkeypatch):
    monkeypatch.setattr(rocm, "env", types.SimpleNamespace(compile=types.SimpleNamespace(arch="")))
    monkeypatch.setattr(rocm, "get_rocm_arch", lambda: "gfx1151")
    assert rocm.RocmBackend.detect_target().arch == "gfx1151"

    monkeypatch.setattr(rocm, "get_rocm_arch", lambda: "")
    with pytest.raises(RuntimeError, match="Set ARCH"):
        rocm.RocmBackend.detect_target()
