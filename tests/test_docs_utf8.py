"""Docs carry Spanish accents: every Markdown file must be UTF-8 without BOM."""

import codecs
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")])


@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_markdown_is_utf8_without_bom(path):
    data = path.read_bytes()
    assert not data.startswith(codecs.BOM_UTF8)
    data.decode("utf-8")
