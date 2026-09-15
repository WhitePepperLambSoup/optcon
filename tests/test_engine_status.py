"""Tests for explicit optional-engine availability reporting."""

from __future__ import annotations

import importlib

import numpy as np

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
