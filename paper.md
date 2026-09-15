---
title: 'optcon: Dimensional, contract-checked physical computation for optics'
tags:
  - Python
  - optics
  - photonics
  - dimensional analysis
  - differentiable simulation
  - differential testing
  - numerical verification
authors:
  - name: Zhe Su
    orcid: 0009-0008-3047-6452
    affiliation: 1
affiliations:
  - name: Research School of Physics, The Australian National University, Canberra, ACT 2601, Australia
    index: 1
date: 14 September 2026
bibliography: paper.bib
---

# Summary

`optcon` is a Python library for dimensional consistency, physical invariant validation, and cross-engine differential testing in computational optics. While conventional scientific software often treats physical units as documentation or discards them before numerical processing, `optcon` enforces dimensional algebra and tracks field-versus-power provenance directly within arithmetic operations. Physical invariants---losslessness (unitarity), passivity, and reciprocity---and automatic-differentiation adjoints are evaluated as runtime assertions. Alongside its semantic type system, `optcon` provides an analytical optics toolkit (ABCD ray matrices, Gaussian beam propagation, optical resonators, thin-film multilayers, polarization calculus, and separable Hermite-Gauss and Laguerre-Gauss modal expansions) coupled with an automated differential testing harness for external optical engines.

![optcon Semantic Contract and Verification Architecture](docs/figures/fig1_semantic_architecture.png)

# Statement of Need

In computational optics, laser cavity modeling, and differentiable physical simulations, four classes of numerical errors routinely bypass compilers and unit tests:

1. **Dimensional Confusion**: Parameters are represented as bare floating-point scalars. Identifiers such as `wavelength_nm` and `cavity_length_mm` do not prevent erroneous expressions like `wavelength_nm + cavity_length_mm` from evaluating to a plausible but invalid number.
2. **Field Amplitude vs. Power Ambiguity**: Field transmission $t$ and power transmission $T = |t|^2$ are both dimensionless real numbers in $[0, 1]$. Confusing amplitude with power introduces a silent square-root discrepancy ($\sqrt{T}$ vs. $T$) that corrupts downstream calculations or optimization objectives without raising runtime errors.
3. **Unchecked Physical Invariants**: Fundamental physical properties such as energy conservation (unitarity, $M^\dagger M = I$), passivity ($\sigma_{\max}(M) \le 1$), and reciprocity ($M = M^T$) are commonly noted in comments or assumed implicitly rather than verified programmatically across multi-element optical systems.
4. **Adjoint and Gradient Inconsistencies**: In gradient-based inverse design and physics-informed models, backward automatic-differentiation passes can converge toward non-physical local minima if numerical discretizations or phase conventions break operator adjointness [@Hughes2019].

`optcon` addresses these issues via an explicit algebraic type system and runtime contract layer tailored to physical optics.

# State of the Field

Existing scientific software divides broadly into general-purpose unit packages and optical propagation libraries. General dimensional libraries such as `astropy.units` [@Astropy2018], `unyt` [@Goldbaum2018], and `Pint` [@Grecco2020] follow the standard SI convention of treating angles (radians) as dimensionless quantities ($[1]$), allowing planar angles to be added directly to bare scalars without error. Furthermore, none of these libraries track the algebraic distinction between field amplitude and optical power.

Optical simulation packages such as `prysm` [@Dube2019], `POPPY` [@Perrin2012], and `LightPipes` [@Schoenmaker2020] provide diffraction algorithms and wavefront models, but they operate on raw NumPy arrays or convert units only at outer entry points. They lack built-in assertions for energy conservation or automated cross-engine verification.

| Software | Target Domain | Angle Base Dimension | Field vs. Power (`amp_order`) | Invariant Contracts | Adjoint Validation | Cross-Engine Adjudication |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `astropy.units` [@Astropy2018] | Astronomy / General | $\times$ | $\times$ | $\times$ | $\times$ | $\times$ |
| `unyt` [@Goldbaum2018] | Astrophysics / General | $\times$ | $\times$ | $\times$ | $\times$ | $\times$ |
| `Pint` [@Grecco2020] | General Science | $\times$ | $\times$ | $\times$ | $\times$ | $\times$ |
| `POPPY` [@Perrin2012] | Physical Optics | $\times$ | $\times$ | $\times$ | $\times$ | $\times$ |
| `prysm` [@Dube2019] | Physical Optics | $\times$ | $\times$ | $\times$ | $\times$ | $\times$ |
| `LightPipes` [@Schoenmaker2020] | Physical Optics | $\times$ | $\times$ | $\times$ | $\times$ | $\times$ |
| **`optcon` (Ours)** | **Computational Optics** | **$\checkmark$** | **$\checkmark$** | **$\checkmark$** | **$\checkmark$** | **$\checkmark$** |

# Software Design

`optcon` is structured around four components:

## 1. Angle as an Independent Base Dimension
Treating angles as dimensionless scalars permits erroneous expressions, such as adding an alignment tilt ($0.3\text{ mrad}$) to a reflectance coefficient. `optcon` defines `angle` as an independent base dimension alongside length, mass, and time, isolating planar angles from scalar ratios.

## 2. Amplitude Order (`amp_order`) Algebra
`optcon` tracks field versus power provenance by assigning an integer amplitude order $k \in \{0, 1, 2, \dots\}$ to each `Quantity`:
- $k = 0$: Dimensionless scalar or count (round-trip counts, operator matrices);
- $k = 1$: Field amplitude (electric field $E$, transmission amplitude $t = \sqrt{T}$);
- $k = 2$: Power or intensity (irradiance, power transmission $T$).

Arithmetic operations propagate amplitude order according to:
$$\text{order}(a \times b) = \text{order}(a) + \text{order}(b), \quad \text{order}(a / b) = \text{order}(a) - \text{order}(b)$$
$$\text{order}(\sqrt{a}) = \frac{\text{order}(a)}{2} \quad (\text{requires even } \text{order}(a))$$
Operations between mismatched non-zero orders raise an `AmplitudeOrderError`, preventing undetected square-root errors.

## 3. Invariant Contracts and Adjoint Verification
`optcon` converts physical conservation laws into executable contracts:
- `assert_unitary(M)`: Verifies that $M^\dagger M = I$ (energy-preserving, lossless);
- `assert_passive(M)`: Checks that the maximum singular value does not exceed unity ($\sigma_{\max}(M) \le 1$);
- `assert_reciprocal(M)`: Enforces symmetry in the amplitude basis ($M = M^T$);
- `dot_test(forward, adjoint, x)`: Implements Claerbout's adjoint dot-product test [@Claerbout1992], ensuring $\langle Jv, w \rangle = \langle v, J^T w \rangle$ for random probes $v, w$, thereby verifying the correctness of gradients in differentiable optics [@Hughes2019].

## 4. Boundary Verification and Vectorized Execution
To avoid per-element runtime overhead in numerical loops, `optcon` validates dimensions and invariant contracts at function boundaries using `@checked`. Once validated, internal routines execute directly on standard NumPy arrays. Exploiting the spatial separability of Hermite-Gauss modes $u_m(x) u_n(y)$, 2D modal decomposition evaluates via outer matrix products $C = \Delta x^2 (U^\dagger F U^*)^T$, requiring $0.6\text{ ms}$ on a $512 \times 512$ grid. The Laguerre-Gauss module supports arbitrary radial order $p$ and signed azimuthal charge $\ell$, where negative charges denote vortex handedness rather than Python reverse indices.

# Cross-Engine Differential Adjudication

`optcon` provides an adapter interface for external optical solvers to compare outputs across independent implementations and analytical limits [@Siegman1986; @Bohren1983]. Differential testing identified two discrepancies in established optical packages:

| Target System | Independent Engines Compared | Maximum Discrepancy | Adjudication & Finding |
| :--- | :--- | :--- | :--- |
| **Mie Scattering** | `miepython` vs. `PyMieScatt` [@Sumlin2018] | $1.9 \times 10^{-3}$ (0.2%) | Resolved by an independent Mie series: `PyMieScatt` defaults to air ($n_{\text{medium}} = 1.00027316$) rather than vacuum. Explicit specification restores agreement to $6.4 \times 10^{-14}$. |
| **Beam Propagation** | `LightPipes` (`Forvard` vs. `Fresnel`) | $6.9 \times 10^{-2}$ (6.9%) | Comparison with the analytical Gaussian beam $w(z) = w_0 \sqrt{1 + (z/z_R)^2}$ shows `Forvard` (spectral) agrees to $2.1 \times 10^{-6}$, whereas `Fresnel` (convolution) is consistently $2\%\sim 7\%$ wide, independent of grid resolution. |
| **Thick Lens ABCD** | `optcon.elements` vs. `optiland` | $1.0 \times 10^{-6}$ | Demonstrates that curved-surface ray-transfer matrices require an explicit $n_1/n_2$ index scaling factor in the lower-right entry. |
| **Thin-Film Stacks** | `tmm_core` (nm) vs. `tmm_fast` (SI m) [@Byrnes2016] | $4.1 \times 10^{-8}$ | Validates agreement across distinct internal coordinate systems. |

Engine adapters follow a two-tier classification: (1) *Tier-1 differential adapters* (`miepython`, `PyMieScatt`, `LightPipes`, and `tmm`), tested when optional packages are available; and (2) *Tier-2 registry specifications* (`optiland`, `poppy`, `ceviche`), whose conventions are cataloged while shared observables or adapters are being developed. CI runs report missing optional engines as explicit skips and isolate import failures.

![Three-Way Adjudication of Mie Extinction Efficiencies across Independent Engines](docs/figures/fig2_mie_adjudication.png)

![Gaussian Beam Propagation Comparison: Spectral vs Convolution](docs/figures/fig3_beam_propagation_anomaly.png)

# Research Impact Statement

`optcon` supports reproducible numerical modeling and physically constrained optimization across three domains:

1. **Resonator Stability and Transverse Mode Coupling**: As demonstrated in `examples/experiment_05_cavity_thermal_tolerance.py`, `optcon` evaluates tolerance budgets for a Fabry-Perot resonator ($\mathcal{F} \approx 3140$, $\lambda = 1064\text{ nm}$, $L = 100\text{ mm}$) subjected to coupled intracavity perturbations:
   - *Angular Misalignment*: Sweeping mirror tilt $\theta \in [0, 3000]\ \mu\text{rad}$ characterizes fundamental mode coupling into higher-order transverse modes ($\mathrm{TEM}_{00} \to \mathrm{TEM}_{10}, \mathrm{TEM}_{20}$). Numerical 2D modal decompositions on a $256 \times 256$ grid agree with Siegman's paraxial perturbation theory [@Siegman1986] to within $3.3 \times 10^{-16}$.
   - *Intracavity Thermal Lensing*: Round-trip ABCD matrix calculations evaluate the shift in effective stability $g_1^* g_2^*$ and eigenmode waist $w_c$ across the stable regime ($D_{\mathrm{th}} \in [-35, +15]\ \mathrm{m}^{-1}$), validated against symplectic determinant contracts ($\det(M) = 1$).
   - *Two-Dimensional Tolerance Budget*: One-parameter scans yield a 90% $\mathrm{TEM}_{00}$ transmission threshold at tilt $\theta = 640.4\ \mu\mathrm{rad}$ and an asymmetric thermal interval $D_{\mathrm{th}} \in [-16.14, 8.92]\ \mathrm{m}^{-1}$ (conservative marginal limit $|D_{\mathrm{th}}| < 8.92\ \mathrm{m}^{-1}$). The joint 2D contour map evaluates simultaneous tilt and thermal perturbations directly.

![End-to-End Scientific Case Study: High-Finesse Laser Cavity Transverse Mode Leakage and Thermal Tolerance Budget](docs/figures/fig4_cavity_tolerance.png)

2. **Constrained Inverse Design and Reinforcement Learning**: Optical alignment agents and gradient optimizers often explore unbounded parameter domains. Runtime contracts (`assert_passive`, `assert_unitary`) intercept unphysical intermediate states (such as non-unitary scattering in passive media) before gradients update model weights, while `dot_test` verifies adjoint operator consistency [@Claerbout1992; @Hughes2019].
3. **Numerical Benchmarking and Solver Validation**: Differential testing reveals hidden solver defaults (such as ambient air refractive index in Mie scattering) and propagation domain limits (such as convolution kernel aliasing below the critical distance $z_c = N \Delta x^2 / \lambda$).

# AI Usage Disclosure

**Declaration on Generative AI**: During the preparation of this software and manuscript, the authors used large language models (OpenAI ChatGPT/Codex and Google DeepMind Antigravity/Gemini) for code scaffolding, docstring editing, unit test parametrization, and draft manuscript compilation. All physical models, mathematical derivations, software architectures, boundary assertions, numerical benchmarks, and manuscript text were reviewed, edited, and validated by the authors. The authors take full responsibility for the contents of this publication.

# Acknowledgements

The authors acknowledge the open-source optical physics community, whose specialized solvers provided the foundation for differential benchmarking, and the Research School of Physics at The Australian National University.

# References

