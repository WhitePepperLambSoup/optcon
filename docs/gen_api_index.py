"""Generate docs/API.md from the package's own docstrings.

Run it from anywhere:

    python optcon/docs/gen_api_index.py

The index is generated rather than hand-written so that it cannot drift away
from the code: if a function is added and its summary is missing, it shows up
here with no description, which is the reminder.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import optcon

HEADER = """# API index

Generated from the package docstrings by `docs/gen_api_index.py`; do not edit
by hand. Every public callable is listed with the first line of its summary.
"""


def _first_line(text: str | None) -> str:
    if not text:
        return ""
    for line in inspect.cleandoc(text).splitlines():
        if line.strip():
            return line.strip()
    return ""


def _public_functions(module) -> list[tuple[str, str]]:
    rows = []
    for name, member in sorted(vars(module).items()):
        if name.startswith("_"):
            continue
        if inspect.isfunction(member) and member.__module__ == module.__name__:
            rows.append((name, _first_line(member.__doc__)))
    return rows


def main() -> int:
    package_dir = Path(optcon.__file__).resolve().parent
    modules = [optcon]
    for info in sorted(pkgutil.iter_modules([str(package_dir)]), key=lambda i: i.name):
        if info.name.startswith("_") or info.name in {"benchmarks", "examples", "tests"}:
            continue
        modules.append(importlib.import_module(f"optcon.{info.name}"))

    lines = [HEADER]
    total = 0
    for module in modules:
        rows = _public_functions(module)
        if not rows:
            continue
        title = "optcon" if module is optcon else f"optcon.{module.__name__.split('.')[-1]}"
        lines.append(f"\n## `{title}`\n")
        if module.__doc__:
            lines.append(_first_line(module.__doc__) + "\n")
        for name, summary in rows:
            lines.append(f"- `{name}` - {summary}")
        total += len(rows)

    lines.append(f"\n---\n\n{total} public functions across {len(modules)} modules.\n")
    target = Path(__file__).resolve().parent / "API.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {target} ({total} functions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
