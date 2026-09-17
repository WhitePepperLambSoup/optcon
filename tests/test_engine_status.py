"""Tests for explicit optional-engine availability reporting."""

from __future__ import annotations

import importlib

import numpy as np

from optcon.benchmarks.engine_status import STRICT_ENGINE_NAMES
from optcon.engines import ENGINE_SPECS, describe_engines, engine_status, registry


def test_reference_engines_are_ready():
    for name in (
        "optcon_reference",
        "optcon_beam_reference",
        "optcon_diffraction",
        "optcon_paraxial",
    ):
        assert engine_status(name) == "ready"


def test_engine_status_uses_explicit_three_state_vocabulary():
    statuses = {engine_status(name) for name in ENGINE_SPECS}
    assert statuses <= {"ready", "missing", "error"}
    assert "ready" in statuses


def test_engine_description_reports_import_errors_without_calling_them_missing():
    report = describe_engines()
    assert "status" in report.splitlines()[0]
    assert "ready" in report
    assert "missing" in report or "error" in report


def test_pymiescatt_shim_restores_scipy_trapz_alias():
    integrate = importlib.import_module("scipy.integrate")
    original = getattr(integrate, "trapz", None)
    had_original = hasattr(integrate, "trapz")
    try:
        if had_original:
            delattr(integrate, "trapz")
        registry._prepare_import_compatibility("PyMieScatt")
        assert hasattr(integrate, "trapz")
        assert np.allclose(integrate.trapz([1.0, 2.0]), np.trapezoid([1.0, 2.0]))
    finally:
        if had_original:
            integrate.trapz = original  # type: ignore[attr-defined]
        elif hasattr(integrate, "trapz"):
            delattr(integrate, "trapz")


def test_import_engine_checks_all_declared_required_modules(monkeypatch):
    spec = registry.EngineSpec(
        name="synthetic",
        domain="test",
        module="synthetic",
        project_path="",
        length_unit="m",
        angle_unit="rad",
        takes_polarization=False,
        summary="test engine",
        required_imports=("synthetic", "synthetic.submodule"),
    )
    monkeypatch.setitem(registry.ENGINE_SPECS, "synthetic", spec)
    imported: list[str] = []

    def fake_import(module_name: str):
        imported.append(module_name)
        return object()

    monkeypatch.setattr(registry.importlib, "import_module", fake_import)
    registry.import_engine.cache_clear()

    assert registry.import_engine("synthetic") == (True, "")
    assert imported == ["synthetic", "synthetic.submodule"]


def test_engine_status_distinguishes_missing_required_module_from_dependency_error(monkeypatch):
    spec = registry.EngineSpec(
        name="synthetic",
        domain="test",
        module="synthetic",
        project_path="",
        length_unit="m",
        angle_unit="rad",
        takes_polarization=False,
        summary="test engine",
        required_imports=("synthetic", "synthetic.submodule"),
    )
    monkeypatch.setitem(registry.ENGINE_SPECS, "synthetic", spec)

    def missing_import(module_name: str):
        if module_name == "synthetic.submodule":
            raise ModuleNotFoundError("No module named 'synthetic.submodule'")
        return object()

    monkeypatch.setattr(registry.importlib, "import_module", missing_import)
    registry.import_engine.cache_clear()
    assert registry.engine_status("synthetic") == "missing"

    def broken_import(module_name: str):
        if module_name == "synthetic.submodule":
            raise ModuleNotFoundError("No module named 'dependency_of_synthetic'")
        return object()

    monkeypatch.setattr(registry.importlib, "import_module", broken_import)
    registry.import_engine.cache_clear()
    assert registry.engine_status("synthetic") == "error"


def test_declared_transitive_dependency_is_reported_as_missing(monkeypatch):
    spec = registry.EngineSpec(
        name="synthetic_dependency",
        domain="test",
        module="synthetic_dependency",
        project_path="",
        length_unit="m",
        angle_unit="rad",
        takes_polarization=False,
        summary="test engine",
        dependency_imports=("optional_dependency",),
    )
    monkeypatch.setitem(registry.ENGINE_SPECS, spec.name, spec)

    def broken_import(module_name: str):
        if module_name == spec.module:
            raise ModuleNotFoundError("No module named 'optional_dependency'")
        return object()

    monkeypatch.setattr(registry.importlib, "import_module", broken_import)
    registry.import_engine.cache_clear()
    assert registry.engine_status(spec.name) == "missing"


def test_undeclared_import_failure_remains_an_error(monkeypatch):
    spec = registry.EngineSpec(
        name="synthetic_broken",
        domain="test",
        module="synthetic_broken",
        project_path="",
        length_unit="m",
        angle_unit="rad",
        takes_polarization=False,
        summary="test engine",
        dependency_imports=("declared_dependency",),
    )
    monkeypatch.setitem(registry.ENGINE_SPECS, spec.name, spec)

    def broken_import(module_name: str):
        if module_name == spec.module:
            raise ModuleNotFoundError("No module named 'undeclared_dependency'")
        return object()

    monkeypatch.setattr(registry.importlib, "import_module", broken_import)
    registry.import_engine.cache_clear()
    assert registry.engine_status(spec.name) == "error"


def test_ml_only_tmm_fast_does_not_block_tier_one_strict_gate():
    assert "tmm_fast" not in STRICT_ENGINE_NAMES
    assert {"tmm_core", "miepython", "PyMieScatt"} <= STRICT_ENGINE_NAMES
