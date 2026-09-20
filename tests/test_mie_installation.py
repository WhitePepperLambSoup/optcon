"""Mie comparisons must work with published packages and no source corpus."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _pymiescatt_is_importable() -> bool:
    """Check availability in the interpreter used by the subprocess test.

    Earlier package-hygiene tests may temporarily add another environment's
    ``site-packages`` directory to ``sys.path``. Metadata discovery can then
    report a distribution that the child interpreter cannot actually import.
    Probe the exact interpreter instead so the skip condition matches the
    execution path below.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import PyMieScatt"],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ.copy(),
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.skipif(
    not _pymiescatt_is_importable(), reason="PyMieScatt is not importable by this interpreter"
)
@pytest.mark.parametrize("medium", [1.0, 1.33, 1.5])
@pytest.mark.parametrize("diameter_nm", [200.0, 1000.0])
def test_installed_pymiescatt_honours_vacuum_wavelength(tmp_path, medium, diameter_nm):
    environment = os.environ.copy()
    environment["OPTCON_CORPUS_ROOT"] = str(tmp_path)
    command = (
        "from optcon import q; from optcon.engines import mie_efficiencies; "
        f"query = dict(m=1.5+0.01j, diameter=q({diameter_nm!r}, 'nm'), "
        f"wavelength=q(550.0, 'nm'), medium_index={medium!r}); "
        "reference = mie_efficiencies(engine='optcon_reference', **query); "
        "candidate = mie_efficiencies(engine='PyMieScatt', **query); "
        "errors = {key: abs(candidate[key].value - reference[key].value) / "
        "max(abs(reference[key].value), 1e-12) for key in reference}; "
        "assert max(errors.values()) < 2e-6, errors"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
