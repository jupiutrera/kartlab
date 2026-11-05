"""Well-known project paths. The package runs from the repository (editable install)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
RECORDS_DIR = REPO_ROOT / "records"  # time-trial records (ignored by git)
