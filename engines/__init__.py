"""Adapters that bring the surveyed optical engines under optcon's type layer."""

from .beams import gaussian_beam_radius
from .differential import assert_engines_agree, compare_across_engines, format_report
from .mie import mie_efficiencies
from .psf import airy_psf_radius
from .registry import (
    ENGINE_SPECS,
    EngineSpec,
    available_engines,
    corpus_root,
    describe_engines,
    engine_spec,
    is_available,
)
from .thinfilm import engine_units, stack_response

__all__ = [
    "ENGINE_SPECS",
    "EngineSpec",
    "available_engines",
    "assert_engines_agree",
    "airy_psf_radius",
    "compare_across_engines",
    "corpus_root",
    "describe_engines",
    "engine_spec",
    "engine_units",
    "format_report",
    "gaussian_beam_radius",
    "is_available",
    "mie_efficiencies",
    "stack_response",
]
