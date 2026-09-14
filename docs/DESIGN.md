# optcon: design and rationale

## The problem

Computational optics, and the machine-learning methods now built on top of it,
share four failure modes that no current toolchain catches:

1. **Dimensions live in identifier names.** `L1_mm`, `wavelength_nm` and
   `pump_radius_um` are all `float`. Nothing stops `wavelength_nm + L1_mm`
   from evaluating to a plausible-looking number.
2. **Field amplitude and power are indistinguishable.** A transmission `T`
   and the field amplitude `t = sqrt(T)` are both dimensionless numbers below
   one. Mixing them is a silent square-root error.
3. **Physical invariants are folklore.** Losslessness, passivity and
   reciprocity are properties a model either has or does not have, and they
   are usually asserted in a comment rather than checked.
4. **A derivative is assumed to be a derivative.** A backward pass can be
   smooth, plausible and wrong; training still "works", just toward the wrong
   objective.

These are not library bugs. They are *representation* problems: the objects
the field computes with do not carry the information needed to detect the
mistakes. That is the gap optcon fills.

## What optcon is, and what it is not

optcon is **not** a replacement for an FDTD, FEM, RCWA or ray-tracing solver.
Those are decades of numerical work, and this design treats them as the
substrate rather than the competition.

optcon is the **semantic layer above them**:

```
        applications / experiments
                    |
   elements  gaussian  fresnel  polarization  cavity  fiber  laser
   grating   thinfilm  diffraction  beam_quality  noise  interferometry
   aberrations  radiometry
                    |
   propagation (sampled Field)  <->  modes (Gauss-Hermite basis)
                                                           <- closed-form optics
                    |
   units  quantity  checks  specs                          <- the type layer
                    |
   engines: tmm_core  tmm_fast  miepython  PyMieScatt  LightPipes
            optiland  rayoptics  poppy  prysm  neuroptica  ceviche  ...
                                                           <- heavy solvers
```

The contribution is not any single layer but the fact that they compose: a
quantity carries its dimension and its amplitude order; a solver declares the
units it expects; a contract states what must be true; and independent
implementations are compared against each other under a declared tolerance.

## Layers

### `units` and `quantity`

Two deliberate departures from SI:

* **Angle is its own base dimension.** SI calls the radian dimensionless,
  which makes `mrad` addable to a bare ratio. In alignment code that is almost
  always a bug, so `angle` is tracked separately.
* **An amplitude order accompanies every value.** `None` means untracked and
  keeps all algebra permissive; `0` is a pure ratio (operators, counts); `1`
  is a field amplitude; `2` is a power-like quantity. Multiplication adds
  orders, division subtracts, `sqrt` halves an even order, and addition or
  comparison requires equal orders.

The amplitude order is the part with no counterpart in pint, unyt or
astropy.units, and it is what catches the square-root class of error.

### `checks`

Physics invariants become checkable claims rather than comments:

* `assert_unitary` - lossless
* `assert_passive` - no net gain
* `assert_reciprocal` - symmetric in the amplitude basis
* `check_gradient`, `assert_adjoint` - a derivative that is actually the
  derivative, verified against central differences and the
  `<Jv, w> == <v, J^T w>` dot-product test

This is the scikit-rf treatment of passivity, extended to optics and combined
with automatic-differentiation verification.

### `specs`

Adoption has to be cheap, so the pattern follows what prysm and poppy already
proved works in production optical code: **units at the boundary, plain
numbers inside**. `@checked` reads `Annotated` units off a signature and
converts on the way in and out; the body keeps its existing arithmetic.
`units_of_dataclass` infers units from field-name suffixes so an existing
parameters dataclass can be brought under check without being rewritten.

### Closed-form optics

`elements`, `gaussian`, `fresnel`, `polarization`, `cavity`, `fiber`,
`grating`, `thinfilm`, `diffraction`, `beam_quality`, `noise`, `laser` and
`interferometry` cover the analysis an optical engineer does before reaching
for a solver. Every function takes and returns typed quantities, and every one
is tested against an independent closed form.

### `engines`

The engines surveyed in the surrounding workspace are brought under the type
layer by thin adapters. Each engine's conventions - domain, module, project
path, length unit, angle unit - are recorded as **data** in a registry, so the
adapter can convert at the boundary instead of relying on the caller to
remember. Engines that cannot be imported in a given environment report why
rather than failing obscurely.

Fifteen engines are registered once their dependencies are present, spanning
thin films, Mie scattering, beam propagation, ray tracing, diffraction, FEM
and FDTD solvers, photonic circuits and differentiable lens design. The
registry is what makes the second wave cheap: adding optiland, poppy, prysm,
neuroptica or ceviche is one spec entry plus, where a shared observable
exists, one differential test.

## Methodology: differential testing

Independent implementations of the same physics are the cheapest source of
ground truth a computational field has, and almost nobody uses them.
`compare_across_engines` runs one physical query through several engines,
normalises the outputs through the unit layer, and reports agreement against a
declared tolerance. `assert_engines_agree` is the CI-gate form.

Three kinds of reference are used, in increasing order of strength:

1. **another library** - two independent routes to the same number;
2. **an independent implementation written here** - used when two libraries
   disagree and a tie must be broken;
3. **a closed form** - an exact answer, which needs no adjudication at all.

## Findings

The method has already produced two results that neither library reports.

### `PyMieScatt.MieQ` defaults to an air medium

Experiment 02 found that `miepython` and `PyMieScatt` disagree on `Qext` by
about 0.2%. Adding an independent Mie series written from Bohren & Huffman
(experiment 03) resolved it: `miepython` agrees with the reference to
`1.8e-10`, while `PyMieScatt` deviates by `1.5e-3`.

The cause is a **default**, not the algorithm. `MieQ` ships
`nMedium=1.00027316` and silently divides the refractive index by it. Passing
`nMedium=1.0` reproduces the reference to `6.4e-14`. Series truncation was
ruled out first: three different term counts give the same ten significant
digits.

### LightPipes' convolution propagator is a few percent wide

Experiment 04 compared both LightPipes propagators against the closed-form
Gaussian width `w(z) = w0 sqrt(1 + (z/zR)^2)`. `Forvard` (spectral) matches to
`2.1e-6`; `Fresnel` (convolution) is 2-7% wide across the whole range, and the
error **does not shrink with finer sampling** (256 to 4096 samples all land
near +7%), so it is not a resolution artefact.

Neither library warns the user about either of these.

### What the second wave of engines confirmed

Installing the missing numerical dependencies brought optiland (ray tracing),
neuroptica (photonic circuits) and ceviche (FDFD with an autograd adjoint)
under test. Each validated something optcon states on its own:

* **optiland's paraxial `f2` for a biconvex N-BK7 singlet matches the ABCD
  chain** assembled from `refracting_surface` and `free_space`, to 1e-6. That
  test also caught a real error: the curved-interface matrix needs an
  ``n1/n2`` factor in its lower-right entry, which only becomes visible once
  the surface is combined with a finite thickness.
* **neuroptica's `is_unitary` and `optcon.checks.is_unitary` agree** on four
  random unitary matrices and four deliberately non-unitary ones.
* **ceviche's autograd adjoint passes `check_gradient`** against central
  differences: the derivative verifier applied to somebody else's solver
  rather than to ours.

## Testing philosophy

Tests are written before the implementation and watched to fail. Beyond that,
three habits run through the suite:

* **Closed forms over golden values.** Beam radii, Fresnel coefficients,
  cavity finesse and Airy patterns are checked against textbook relations, not
  against recorded outputs.
* **Cross-module self-consistency.** The cavity spot sizes predicted by the
  `g`-parameter formulas are compared with the eigenmode obtained by solving
  the ABCD round trip; the general mode-overlap integral is compared with the
  coaxial coupling formula. Two routes, one number.
* **Cross-engine agreement.** Where a corpus engine covers the same physics -
  Fresnel coefficients against `tmm_core`, single-layer coatings against
  `tmm_core`, Mie against two libraries - the comparison is part of the suite.

Third-party behaviour that is wrong, or merely surprising, is pinned as a
characterisation test with a comment recording what was measured, so that a
future change upstream is noticed.

## Quality gates

```bash
python -m pytest optcon/tests -q
python -m ruff check --no-cache optcon
python -m mypy optcon
```

Engine-dependent tests skip cleanly when the engine's source is not present,
so the suite is meaningful on a bare checkout. `OPTCON_CORPUS_ROOT` points the
registry at a different corpus.

## Roadmap

* Adapters for `diffractio`, `tracepy` and `pyoptools`; the registry already
  records their conventions, but a shared observable has to be defined first -
  grid alignment for diffraction, surface conventions for ray tracing.
* A field-representation layer took its first step: `propagation.Field` carries
  its grid, and `modes` provides the Gauss-Hermite basis with explicit
  decomposition and reconstruction. Laguerre-Gauss modes and vectorial fields
  are the natural next additions.
* Adapters, not just registry entries, for the engines whose shared observable
  is still undefined: grid alignment for diffraction (`poppy`, `prysm`,
  `diffractio`), surface conventions for ray tracing (`rayoptics`,
  `tracepy`, `pyoptools`), curl conventions for the EM solvers (`ceviche`,
  `femwell`, `A_FMM`).
* An effects layer for stochastic parameters and hardware side effects, which
  is what an RL environment contract needs in order to be checkable.
* A capability-typed sublanguage for agent-generated repair programs,
  replacing the hand-written verb list such systems currently carry.
* Torch and JAX numeric backends; the current carrier is NumPy.
