"""Command line for track files: ``validate`` and ``export``.

Run with ``uv run python scripts/track_file.py --help``. Exit code 0 means every file
is valid; 1 means a file failed validation; 2 means a usage error or an unwritable file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kartlab.sim.showcase import SHOWCASE
from kartlab.sim.trackfile import (
    TrackFileError,
    TrackSpec,
    load_spec,
    save_spec,
    spec_from_seed,
    spec_from_showcase,
    validate_spec,
)
from kartlab.sim.tracksets import is_eval_seed


def validate_files(paths: list[Path]) -> int:
    """Print a verdict per file; return the exit code."""
    failed = 0
    for path in paths:
        try:
            spec = load_spec(path)
        except TrackFileError as error:
            print(f"ERROR {error}", file=sys.stderr)
            failed += 1
            continue
        report = validate_spec(spec)
        if report.ok:
            print(
                f"OK {path}: «{spec.name}», {report.length:.0f} m, anchura {spec.width:g} m, "
                f"radio mínimo {report.min_radius:.1f} m, hierba mínima {report.gap:.1f} m"
                + (f", {report.crossings} cruce(s) a nivel" if report.crossings else "")
                + (", uno de ellos puente con túnel" if spec.bridge else "")
            )
        else:
            failed += 1
            print(f"ERROR {path}: «{spec.name}» no es válido", file=sys.stderr)
            for problem in report.problems:
                print(f"  - {problem}", file=sys.stderr)
    return 1 if failed else 0


def export_spec(spec: TrackSpec, out: Path) -> int:
    report = validate_spec(spec)
    if not report.ok:  # a spec we produce ourselves should always be valid
        for problem in report.problems:
            print(f"ERROR {problem}", file=sys.stderr)
        return 1
    try:
        save_spec(out, spec)
    except OSError as error:
        print(f"ERROR no se puede escribir {out}: {error.strerror or error}", file=sys.stderr)
        return 2
    print(f"Guardado {out} ({report.length:.0f} m)")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Circuitos en fichero JSON: validar y exportar.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("validate", help="comprueba uno o varios ficheros de circuito")
    check.add_argument("files", nargs="+", type=Path)

    export = sub.add_parser("export", help="guarda un circuito existente como fichero editable")
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", type=int, help="semilla de entrenamiento del generador")
    source.add_argument("--showcase", choices=sorted(SHOWCASE), help="circuito de escaparate")
    export.add_argument("--out", type=Path, required=True, help="fichero .json de salida")
    args = parser.parse_args(argv)

    if args.command == "validate":
        raise SystemExit(validate_files(args.files))
    if args.seed is not None:
        if args.seed < 0:
            print(f"ERROR la semilla {args.seed} no puede ser negativa", file=sys.stderr)
            raise SystemExit(2)
        if is_eval_seed(args.seed):
            print(f"ERROR la semilla {args.seed} está reservada para evaluación", file=sys.stderr)
            raise SystemExit(2)
        spec = spec_from_seed(args.seed)
    else:
        spec = spec_from_showcase(args.showcase)
    raise SystemExit(export_spec(spec, args.out))
