"""The project must never pull GPU libraries: PyTorch is the CPU-only build."""

import tomllib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
GPU_MARKERS = ("nvidia", "cuda", "cudnn", "triton")
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _locked_packages() -> list[dict]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return lock["package"]


def test_torch_is_cpu_build():
    assert torch.version.cuda is None
    assert not torch.cuda.is_available()


def test_torch_is_locked_from_cpu_index():
    torch_entries = [pkg for pkg in _locked_packages() if pkg["name"] == "torch"]
    assert torch_entries
    assert all(pkg["source"].get("registry") == CPU_INDEX for pkg in torch_entries)


def test_lockfile_has_no_gpu_packages():
    names = [pkg["name"] for pkg in _locked_packages()]
    assert [name for name in names if any(marker in name for marker in GPU_MARKERS)] == []
