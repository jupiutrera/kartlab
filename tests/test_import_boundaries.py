"""Training-side layers must stay headless: no pygame and no rendering code.

Each layer is imported in a fresh interpreter, because this test process may
already have pygame loaded by other tests.
"""

import subprocess
import sys

import pytest

HEADLESS_LAYERS = ["kartlab.common", "kartlab.sim", "kartlab.env", "kartlab.train"]

CHECK = """
import importlib, sys
importlib.import_module(sys.argv[1])
leaked = [
    name for name in sys.modules
    if name.split(".")[0] == "pygame" or name.startswith(("kartlab.render", "kartlab.show"))
]
print("\\n".join(sorted(leaked)))
"""


@pytest.mark.parametrize("layer", HEADLESS_LAYERS)
def test_headless_layer_does_not_import_rendering(layer):
    result = subprocess.run(
        [sys.executable, "-c", CHECK, layer],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert result.stdout.split() == []
