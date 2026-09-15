"""Package hygiene: what an open-source release has to get right.

These tests do not check physics.  They check that the declared metadata, the
public namespace and the importable surface agree with each other, which is
the class of mistake that survives every physics test and then embarrasses a
release.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest

import optcon

PACKAGE_ROOT = Path(optcon.__file__).resolve().parent
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"

MODULES = (
    "units",
    "quantity",
    "checks",
    "specs",
    "errors",
    "elements",
    "gaussian",
    "fresnel",
    "polarization",
    "cavity",
    "fiber",
    "grating",
    "thinfilm",
    "diffraction",
    "beam_quality",
    "noise",
    "laser",
    "interferometry",
    "aberrations",
    "radiometry",
    "propagation",
    "modes",
    "mtf",
    "waveguide",
    "thermal",
    "nonlinear",
    "vector_fields",
    "mueller",
    "engines",
    "examples",
    "benchmarks",
)


def test_every_documented_module_imports():
    for name in MODULES:
        importlib.import_module(f"optcon.{name}")


def test_reproducibility_entry_points_are_importable():
    importlib.import_module("optcon.examples.experiment_05_cavity_thermal_tolerance")
    importlib.import_module("optcon.benchmarks.run_all")


def test_version_is_declared_and_semantic():
    assert re.fullmatch(r"\d+\.\d+\.\d+", optcon.__version__), optcon.__version__


def test_version_matches_the_packaging_metadata():
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert optcon.__version__ == metadata["project"]["version"]


def test_every_name_in_all_exists():
    missing = [name for name in optcon.__all__ if not hasattr(optcon, name)]
    assert missing == []


def test_all_is_sorted_and_free_of_duplicates():
    names = list(optcon.__all__)
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_the_core_package_does_not_import_the_engines():
    """The engine layer is optional; importing optcon must not require it."""
    import subprocess
    import sys

    script = (
        "import sys, optcon; "
        "assert not any(m.startswith('optcon.engines') for m in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(PACKAGE_ROOT.parent),
    )
    assert completed.returncode == 0, completed.stderr


def test_pyproject_declares_the_licence_and_dependencies():
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = metadata["project"]
    assert project["license"] == "MIT"
    assert any(dep.startswith("numpy") for dep in project["dependencies"])
    assert any(dep.startswith("scipy") for dep in project["dependencies"])
    assert "optcon[dev,engines]" not in project["optional-dependencies"]["all"]
    assert set(project["optional-dependencies"]["dev"]).issubset(project["optional-dependencies"]["all"])
    assert set(project["optional-dependencies"]["engines"]).issubset(project["optional-dependencies"]["all"])


def test_an_unknown_unit_name_raises_rather_than_returning_none():
    with pytest.raises(optcon.UnitError):
        optcon.unit("furlong")
