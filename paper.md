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
  - name: Author Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Research School of Physics, The Australian National University, Acton, ACT 2601, Australia
    index: 1
date: 14 September 2026
bibliography: paper.bib
---

# Summary

`optcon` is a Python library that provides dimensional safety, invariant contracts, and differential verification for computational optics and machine learning. Unlike conventional scientific packages that treat physical dimensions as cosmetic annotations or strip them during calculation, `optcon` embeds dimensional tracking and field-power provenance directly into arithmetic operations. Furthermore, it turns physical invariants---such as losslessness (unitarity), passivity, and reciprocity---as well as automatic differentiation adjoints into machine-checkable runtime assertions. In addition to serving as a semantic type layer above external heavy solvers, `optcon` provides a closed-form optical toolkit spanning Gaussian beams, ABCD ray matrices, optical cavities, thin-film stacks, polarization calculus, and Fourier diffraction, accompanied by an automated cross-engine differential testing framework.

![optcon Semantic Contract and Verification Architecture](docs/figures/fig1_semantic_architecture.png)

# Statement of Need

In computational optics, laser cavity engineering, and differentiable physical simulations, four critical classes of errors frequently evade traditional compilers and unit testing suites:

1. **Dimensional Confusion**: Optical parameters are commonly represented as bare floating-point numbers (`float`). Variable names such as `wavelength_nm` and `cavity_length_mm` do not prevent erroneous operations like `wavelength_nm + cavity_length_mm` from silently evaluating to a plausible numerical value.
2. **Field Amplitude vs. Power Ambiguity**: In wave optics, field amplitude transmission $t$ and power transmission $T = |t|^2$ are both dimensionless real numbers between $0$ and $1$. Confusing an amplitude ratio with a power ratio produces a silent "square-root error" ($\sqrt{T}$ vs. $T$) that propagates undetected through optimization landscapes.
3. **Unchecked Physical Invariants**: Fundamental principles such as energy conservation (unitarity, $M^\dagger M = I$), passivity ($\sigma_{\max}(M) \le 1$), and reciprocity ($M = M^T$) are routinely asserted in comments or assumed implicitly, but rarely validated programmatically across complex multi-element optical trains.
4. **Adjoint and Gradient Inconsistencies**: In inverse optical design and physics-informed neural networks, a backward automatic-differentiation pass may appear smooth and convergent while silently computing mathematically incorrect gradients due to subtle phase discrepancies or non-adjoint numerical operators [@Hughes2019].

`optcon` addresses these issues by introducing an explicit algebraic type system and invariant contract layer designed specifically for physical optics.

# State of the Field

Existing scientific computing software divides broadly into general-purpose dimensional libraries and optical propagation packages. General dimensional libraries such as `astropy.units` [@Astropy2018], `unyt` [@Goldbaum2018], and `Pint` [@Grecco2020] follow the standard SI convention of treating angles (radians) as dimensionless numbers ($[1]$), allowing planar angles to be added directly to bare scalars without error. Furthermore, no existing unit library tracks the algebraic distinction between field amplitude and power/intensity.

Optical simulation packages such as `prysm` [@Dube2019], `POPPY` [@Perrin2012], and `LightPipes` [@Schoenmaker2020] provide rich diffraction algorithms and wavefront models, but they strip units at function boundaries or manage numbers as bare NumPy arrays, offering no automated runtime contracts for physical conservation laws or cross-engine discrepancy detection.

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

The architecture of `optcon` is centered on four core principles:

## 1. Angle as an Independent Base Dimension
In optical alignment and beam propagation, treating angles as dimensionless numbers is a persistent source of bugs (e.g., adding an alignment tilt of $0.3\text{ mrad}$ to a bare reflectance). `optcon` defines `angle` as an independent base dimension alongside length, mass, and time, preventing accidental mixing between planar angles and dimensionless ratios.

## 2. Amplitude Order (`amp_order`) Algebra
`optcon` associates every `Quantity` with an optional amplitude order $k \in \{0, 1, 2, \dots\}$:
- $k = 0$: Pure ratio or count (e.g., number of round trips, operator matrix);
- $k = 1$: Field amplitude (e.g., electric field $E$, Fresnel transmission amplitude $t = \sqrt{T}$);
- $k = 2$: Power or intensity (e.g., irradiance, power transmission $T$).

The algebraic rules strictly enforce physical consistency:
$$\text{order}(a \times b) = \text{order}(a) + \text{order}(b), \quad \text{order}(a / b) = \text{order}(a) - \text{order}(b)$$
$$\text{order}(\sqrt{a}) = \frac{\text{order}(a)}{2} \quad (\text{requires } \text{order}(a) \text{ to be even})$$
Addition and comparison between different non-zero orders raise an `AmplitudeOrderError`, making silent square-root discrepancies impossible.

## 3. Invariant Contracts and Adjoint Verification
`optcon` converts physical conservation laws into executable contracts:
- `assert_unitary(M)`: Verifies that $M^\dagger M = I$ (energy-preserving, lossless);
- `assert_passive(M)`: Checks that the maximum singular value does not exceed unity ($\sigma_{\max}(M) \le 1$);
- `assert_reciprocal(M)`: Enforces symmetry in the amplitude basis ($M = M^T$);
- `dot_test(forward, adjoint, x)`: Implements Claerbout's adjoint dot-product test [@Claerbout1992], ensuring $\langle Jv, w \rangle = \langle v, J^T w \rangle$ for random probes $v, w$, thereby guaranteeing the mathematical correctness of gradients used in differentiable optics [@Hughes2019].

## 4. Boundary Enforcement with High-Performance Kernel Computation
To ensure minimal computational overhead and seamless interoperability, `optcon` provides the `@checked` decorator. Dimensional assertions and type conversions occur strictly at the boundaries of public functions. Once validated, array operations run as unboxed NumPy arrays with optimized vectorization. Spatial separability of Hermite-Gauss modes $u_m(x) u_n(y)$ reduces 2D modal decompositions to outer matrix products $C = \Delta x^2 (U^\dagger F U^*)^T$, executing in under $0.6\text{ ms}$ on a $512 \times 512$ grid.

# Cross-Engine Differential Adjudication

`optcon` integrates an extensible adapter layer for external optical solvers, enabling automated differential verification against independent implementations and textbook closed forms [@Siegman1986; @Bohren1983]. This adjudication framework has resolved two notable undocumented behaviors in widely used optical software:

| Target System | Independent Engines Compared | Maximum Discrepancy | Adjudication & Finding |
| :--- | :--- | :--- | :--- |
| **Mie Scattering** | `miepython` vs. `PyMieScatt` [@Sumlin2018] | $1.9 \times 10^{-3}$ (0.2%) | Resolved by an independent Mie series: `PyMieScatt` defaults to air ($n_{\text{medium}} = 1.00027316$) rather than vacuum. Explicit specification restores agreement to $6.4 \times 10^{-14}$. |
| **Beam Propagation** | `LightPipes` (`Forvard` vs. `Fresnel`) | $6.9 \times 10^{-2}$ (6.9%) | Comparison with the analytical Gaussian beam $w(z) = w_0 \sqrt{1 + (z/z_R)^2}$ shows `Forvard` (spectral) agrees to $2.1 \times 10^{-6}$, whereas `Fresnel` (convolution) is consistently $2\%\sim 7\%$ wide, independent of grid resolution. |
| **Thick Lens ABCD** | `optcon.elements` vs. `optiland` | $1.0 \times 10^{-6}$ | Demonstrates that curved-surface ray-transfer matrices require an explicit $n_1/n_2$ index scaling factor in the lower-right entry. |
| **Thin-Film Stacks** | `tmm_core` (nm) vs. `tmm_fast` (SI m) [@Byrnes2016] | $4.1 \times 10^{-8}$ | Validates agreement across distinct internal coordinate systems. |

Engine adapters follow a two-tier classification: (1) *Tier-1 Fully Supported Adapters* (`miepython`, `PyMieScatt`, `LightPipes`, `tmm`), continuous-integration tested with graceful degradation when optional packages are absent; and (2) *Tier-2 Architectural Specifications* (`optiland`, `poppy`, `ceviche`), providing standardized schemas for downstream solver integration.

![Three-Way Adjudication of Mie Extinction Efficiencies across Independent Engines](docs/figures/fig2_mie_adjudication.png)

![Gaussian Beam Propagation Comparison: Spectral vs Convolution](docs/figures/fig3_beam_propagation_anomaly.png)

# Research Impact Statement

`optcon` is engineered to serve as a foundational validation infrastructure for computational optics and photonics research:

1. **Laser Cavity & Resonator Alignment**: In complex Fabry-Pérot or ring cavities, small angular misalignments or thermal lensing perturbations can lead to numerical instability. `optcon` provides closed-form stability parameters ($0 \le g_1 g_2 \le 1$), Gouy phase calculations, and beam-waist matching with strict dimensional tracking.
2. **Reinforcement Learning & Inverse Design**: Automated optical alignment agents and differentiable inverse design algorithms frequently operate over unbounded action spaces. By wrapping physical simulations in `assert_passive` and `assert_unitary` guardrails, `optcon` traps unphysical gradient states before they contaminate neural network weights.
3. **Reproducibility & Differential Benchmark**: By quantifying discrepancies between disparate optical simulation packages, `optcon` provides researchers with objective validation criteria, eliminating hidden configuration artifacts in reported literature.

# AI Usage Disclosure

**Author Contributions**: The conceptual architecture, physical formulation (including the independent angle dimension and amplitude order monoid), mathematical proofs, and core algorithmic decisions were conceived and verified by the authors. 

**Generative AI Disclosure**: Generative AI tools (including Google DeepMind Antigravity / Gemini) were employed as pair-programming and refactoring assistants to aid in code drafting, docstring expansion, test coverage amplification, and initial LaTeX/Markdown compilation. In accordance with JOSS Editorial Policies, the authors independently audited, executed, reviewed, and experimentally validated all code, proofs, numerical benchmarks, and manuscript text, and assume full scientific and legal responsibility for the integrity and reproducibility of all submitted materials.

# Acknowledgements

The authors acknowledge the open-source optical physics community, whose specialized solvers provided the foundation for differential benchmarking, and the Research School of Physics at The Australian National University.

# References

