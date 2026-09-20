"""The thin-film adapter must work without a neighboring source checkout."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

from optcon.engines import registry


@pytest.mark.skipif(importlib.util.find_spec("tmm") is None, reason="tmm is not installed")
@pytest.mark.parametrize(
    "command",
    [
        "from optcon.engines.registry import engine_status; "
        "assert engine_status('tmm_core') == 'ready'",
        "import math; from optcon import q; "
        "from optcon.engines import stack_response; "
        "result = stack_response(n_list=[1.0, 1.5], "
        "thicknesses=[math.inf, math.inf], wavelength=q(550.0, 'nm'), "
        "angle=q(0.0, 'rad')); "
        "assert abs(result['R'].value - 0.04) < 1e-14; "
        "assert abs(result['T'].value - 0.96) < 1e-14",
    ],
    ids=["availability", "fresnel_response"],
)
def test_installed_tmm_works_without_a_source_corpus(tmp_path, command):
    environment = os.environ.copy()
    environment["OPTCON_CORPUS_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_tmm_dependency_failure_is_not_hidden_by_the_installed_fallback(monkeypatch):
    imported = []

    def broken_import(module_name):
        imported.append(module_name)
        raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")

    monkeypatch.setattr(registry.importlib, "import_module", broken_import)
    registry.import_engine.cache_clear()
    try:
        assert registry.engine_status("tmm_core") == "error"
        assert imported == ["tmm_core"]
    finally:
        registry.import_engine.cache_clear()
