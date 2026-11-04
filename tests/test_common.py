import subprocess
import sys

import pytest

from kartlab.common import io
from kartlab.common.io import atomic_write, read_yaml, write_text_atomic, write_yaml_atomic


def test_atomic_write_replaces_the_file_and_leaves_no_temporaries(tmp_path):
    target = tmp_path / "sub" / "file.txt"
    write_text_atomic(target, "uno\n")
    write_text_atomic(target, "dos con tilde: ñ\n")
    assert target.read_bytes() == "dos con tilde: ñ\n".encode()
    assert [p.name for p in target.parent.iterdir()] == ["file.txt"]


def test_a_failed_write_keeps_the_old_file(tmp_path):
    target = tmp_path / "file.txt"
    write_text_atomic(target, "bueno")

    def broken(tmp):
        tmp.write_text("a medias", encoding="utf-8")
        raise RuntimeError("fallo")

    with pytest.raises(RuntimeError):
        atomic_write(target, broken)
    assert target.read_text(encoding="utf-8") == "bueno"
    assert [p.name for p in tmp_path.iterdir()] == ["file.txt"]


def test_replace_is_retried_while_the_target_is_busy(tmp_path, monkeypatch):
    calls = []
    real_replace = io.os.replace

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError("en uso")
        real_replace(src, dst)

    monkeypatch.setattr(io.os, "replace", flaky)
    monkeypatch.setattr(io, "REPLACE_WAIT", 0.0)
    write_text_atomic(tmp_path / "f.txt", "ok")
    assert len(calls) == 3
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "ok"


def test_yaml_round_trip_keeps_accents_and_order(tmp_path):
    data = {"nombre": "Cañón", "pasos": 20000, "lista": [1, 2]}
    write_yaml_atomic(tmp_path / "c.yaml", data)
    assert read_yaml(tmp_path / "c.yaml") == data
    assert list(read_yaml(tmp_path / "c.yaml")) == ["nombre", "pasos", "lista"]


def test_lower_priority_is_inherited_by_child_processes():
    code = (
        "import subprocess, sys\n"
        "from kartlab.common.system import lower_priority, is_below_normal_priority\n"
        "lower_priority()\n"
        "assert is_below_normal_priority()\n"
        "child = 'from kartlab.common.system import is_below_normal_priority as f; "
        "assert f()'\n"
        "subprocess.run([sys.executable, '-c', child], check=True)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
