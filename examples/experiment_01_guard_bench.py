"""Experiment 01 - does the type layer actually catch physics bugs?

Method: a labelled set of optical/ML mistakes plus a labelled set of
legitimate operations, each run twice - once as ordinary float code and once
through optcon - then count

  * detection rate   : injected bugs that raise a typed error
  * false-alarm rate : legitimate operations that wrongly raise

The float path is what the same author would write in a plain numpy model,
so the gap between the two columns is the value the library adds.

Run:  python -m optcon.examples.experiment_01_guard_bench
"""

from __future__ import annotations

import math

import numpy as np

from optcon import (
    OptConError,
    amplitude_ratio,
    assert_adjoint,
    assert_gradient_matches,
    assert_passive,
    assert_unitary,
    dimensionless,
    power_ratio,
    q,
    sqrt,
)

LAMBDA_NM = 1064.0
LAMBDA_MM = 1.064e-3


def _field_transmission(power_transmission: float) -> float:
    """The physically correct field amplitude for a power transmission."""
    return math.sqrt(power_transmission)


#: The unit the float path implicitly assumes, for cases where both paths
#: produce a number.  Without this, "different value" would be confounded with
#: "same value written in a different unit".
PLAIN_UNITS = {
    "add 1064 nm to 0.3 mm (legal, same dimension)": "mm",
    "sine of a mirror tilt given in mrad (legal)": "1",
    "nm wavelength added into mm algebra (legal, float path mixes units)": "mm",
    "field amplitude from power transmission (legal)": "1",
    "cascading two field amplitudes (legal)": "1",
}


# (category, name, is_bug, plain_float_callable, optcon_callable)
CASES = [
    # -- unit mixing -----------------------------------------------------
    (
        "units", "add 1064 nm to 0.3 mm (legal, same dimension)", False,
        lambda: LAMBDA_NM * 1e-6 + 0.3,
        lambda: q(1064.0, "nm") + q(0.3, "mm"),
    ),
    (
        "units", "add mirror tilt in mrad to a length in mm", True,
        lambda: LAMBDA_MM + 0.3,
        lambda: q(1.064, "mm") + q(0.3, "mrad"),
    ),
    (
        "units", "exponentiate a length", True,
        lambda: math.exp(1.0),
        lambda: np.exp(q(1.0, "mm")),
    ),
    (
        "units", "sine of a bare count instead of an angle", True,
        lambda: math.sin(1.0),
        lambda: np.sin(dimensionless(1.0)),
    ),
    (
        "units", "sine of a mirror tilt given in mrad (legal)", False,
        lambda: math.sin(0.3),
        lambda: np.sin(q(0.3, "mrad")),
    ),
    (
        "units", "nm wavelength added into mm algebra (legal, float path mixes units)", False,
        lambda: LAMBDA_NM + 300.0,
        lambda: q(LAMBDA_NM, "nm") + q(300.0, "mm"),
    ),
    # -- amplitude vs power ----------------------------------------------
    (
        "amplitude", "field amplitude from power transmission (legal)", False,
        lambda: _field_transmission(0.81),
        lambda: sqrt(power_ratio(0.81)),
    ),
    (
        "amplitude", "sqrt applied twice to a transmission", True,
        lambda: math.sqrt(_field_transmission(0.81)),
        lambda: sqrt(amplitude_ratio(0.9)),
    ),
    (
        "amplitude", "power transmission added to a field amplitude", True,
        lambda: 0.81 + 0.9,
        lambda: power_ratio(0.81) + amplitude_ratio(0.9),
    ),
    (
        "amplitude", "cascading two field amplitudes (legal)", False,
        lambda: _field_transmission(0.81) * _field_transmission(0.64),
        lambda: sqrt(power_ratio(0.81)) * sqrt(power_ratio(0.64)),
    ),
    # -- invariant contracts ---------------------------------------------
    (
        "invariant", "lossless quarter-wave plate", False,
        lambda: None,
        lambda: assert_unitary(np.array([[1, 0], [0, 1j]], dtype=complex), name="QWP"),
    ),
    (
        "invariant", "waveplate normalised by the wrong factor", True,
        lambda: None,
        lambda: assert_unitary(
            np.array([[0.9, 0.0], [0.0, 0.9j]], dtype=complex), name="misnormalised QWP"
        ),
    ),
    (
        "invariant", "mirror amplitude gain 1.4 declared passive", True,
        lambda: None,
        lambda: assert_passive(np.array([[1.4]], dtype=complex), name="mirror"),
    ),
    # -- derivative contracts --------------------------------------------
    (
        "derivative", "correct gradient of a quadratic objective", False,
        lambda: None,
        lambda: assert_gradient_matches(
            lambda x: float(np.sum(x**2)), lambda x: 2.0 * x, np.array([0.3, -1.2])
        ),
    ),
    (
        "derivative", "gradient that silently omits one term", True,
        lambda: None,
        lambda: assert_gradient_matches(
            lambda x: float(np.sum(x**2)),
            lambda x: np.array([2.0 * x[0], 0.0]),
            np.array([0.3, -1.2]),
        ),
    ),
    (
        "derivative", "adjoint that drops a coupling term", True,
        lambda: None,
        lambda: assert_adjoint(
            lambda x: np.array([[1.0, 2.0], [0.0, 1.0]]) @ x,
            lambda y: np.array([[1.0, 0.0], [0.0, 1.0]]) @ y,
            np.array([0.5, -0.5]),
        ),
    ),
]


def main() -> int:
    header = (
        f"{'category':11s} {'case':56s} {'plain float':>13s}  "
        f"{'optcon':>19s}  {'typed value':>13s}"
    )
    print(header)
    print("-" * len(header))

    bugs_detected = bugs_total = 0
    false_alarms = clean_total = 0
    divergences = 0
    divergent_names = []
    for category, name, is_bug, plain, guarded in CASES:
        plain_result = None
        try:
            plain_result = plain()
            plain_column = "no error" if plain_result is None else f"{float(plain_result):.6g}"
        except Exception as error:  # the plain path is the control group
            plain_column = type(error).__name__
        raised = False
        typed_result = None
        try:
            typed_result = guarded()
            optcon_column = "ok"
        except OptConError as error:
            optcon_column = type(error).__name__
            raised = True

        typed_column = ""
        reference_unit = PLAIN_UNITS.get(name)
        if typed_result is not None and reference_unit is not None and plain_result is not None:
            typed_value = float(typed_result.to_value(reference_unit))
            typed_column = f"{typed_value:.6g}"
            if not math.isclose(typed_value, float(plain_result), rel_tol=1e-9):
                divergences += 1
                divergent_names.append(name)
                typed_column = f"{typed_value:.6g} <--"

        if is_bug:
            bugs_total += 1
            bugs_detected += raised
        else:
            clean_total += 1
            false_alarms += raised

        mark = "" if raised == is_bug else "   <-- MISMATCH"
        print(
            f"{category:11s} {name:56s} {plain_column:>13s}  "
            f"{optcon_column:>19s}  {typed_column:>13s}{mark}"
        )

    print("-" * len(header))
    print(f"injected bugs detected     : {bugs_detected}/{bugs_total}")
    print(f"false alarms on legal code : {false_alarms}/{clean_total}")
    print(f"silent numeric divergences : {divergences} of {len(PLAIN_UNITS)} comparable cases")
    for divergent in divergent_names:
        print(f"    - {divergent}")
    print(
        "\nThe 'plain float' column is what a numpy-only model produces: an\n"
        "innocent-looking number. Two distinct failure modes show up:\n"
        "  * a bug that raises here would have been a wrong loss curve there;\n"
        "  * a 'silent divergence' raises nothing, yet the typed path answers a\n"
        "    different question because the units were finally respected."
    )
    return 0 if (bugs_detected == bugs_total and false_alarms == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
