"""Report optional optical-engine status for local runs and CI.

The command distinguishes an unavailable optional engine (SKIP) from a found
engine that fails during import (FAIL). This keeps a bare checkout usable while
preventing broken optional installations from looking like successful skips.
Use --strict in CI to gate only Tier-1 engines with differential adapters.
"""

from __future__ import annotations

import argparse

from optcon.engines.registry import ENGINE_SPECS, engine_status, import_engine

STRICT_ENGINE_NAMES = {
    "miepython",
    "PyMieScatt",
    "lightpipes_forvard",
    "lightpipes_fresnel",
    "tmm_core",
}


def main(strict: bool = False) -> int:
    failures = 0
    print("Optical engine availability")
    print("---------------------------")
    for name in sorted(ENGINE_SPECS):
        status = engine_status(name)
        if status == "ready":
            print(f"[PASS] {name}")
        elif status == "missing":
            print(f"[SKIP] {name}: optional package/source unavailable")
        else:
            if not strict or name in STRICT_ENGINE_NAMES:
                failures += 1
            _, message = import_engine(name)
            gate = " (gating)" if name in STRICT_ENGINE_NAMES else " (survey-only)"
            print(f"[FAIL] {name}{gate}: {message}")
    if failures:
        print(f"\n{failures} gating engine import failure(s) require attention.")
        return 1
    print(
        "\nNo gating engine import failures detected; missing optional engines are explicit skips."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when a gating Tier-1 engine has an import error",
    )
    raise SystemExit(main(strict=parser.parse_args().strict))
