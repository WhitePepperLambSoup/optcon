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

`optcon` is a Python library that provides dimensional safety, invariant contracts, and differential verification for computational optics and machine learning. Unlike conventional scientific packages that treat physical dimensions as cosmetic annotations or strip them during calculation, `optcon` embeds dimensional tracking and field-power provenance directly into arithmetic operations. Furthermore, it turns physical invariants---such as losslessness (unitarity), passivity, and reciprocity---as well as automatic differentiation adjoints into machine-checkable runtime assertions. In addition to serving as a semantic type layer above external heavy solvers, `optcon` provides a closed-form optical toolkit spanning Gaussian beams, ABCD ray matrices, optical cavities, thin-film stacks, polarization calculus, and Fourier diffraction, accompanied by a cross-engine differential testing framework.

![optcon Semantic Contract and Verification Architecture](docs/figures/fig1_semantic_architecture.png)

# Statement of Need

In computational optics and differentiable physical simulations, four critical classes of errors frequently evade traditional compilers and testing suites:

1. **Dimensional Confusion**: Optical parameters are commonly represented as bare floating-point numbers (`float`). Variable names such as `wavelength_nm` and `cavity_length_mm` do not prevent erroneous operations like `wavelength_nm + cavity_length_mm` from silently evaluating to a plausible numerical value.
2. **Field Amplitude vs. Power Ambiguity**: In optical physics, field amplitude transmission $t$ and power transmission $T = |t|^2$ are both dimensionless real numbers between $0$ and $1$. Confusing an amplitude ratio with a power ratio produces a silent "square-root error" ($\sqrt{T}$ vs. $T$) that propagates undetected through optimization landscapes.
3. **Unchecked Physical Invariants**: Fundamental principles such as energy conservation (unitarity), passivity ($\|M\|_2 \le 1$), and reciprocity are routinely asserted in comments or assumed implicitly, but rarely validated programmatically across complex element chains.
4. **Adjoint and Gradient Inconsistencies**: In inverse design and physics-informed neural networks, a backward automatic-differentiation pass may appear smooth and convergent while silently computing incorrect gradients due to subtle phase discrepancies or non-adjoint numerical operators.

Existing unit packages, such as `astropy.units` [@Astropy2018] and `unyt` [@Goldbaum2018], follow the SI convention of treating angles (radians) as dimensionless ratios, allowing angles to be added directly to bare scalars. Crucially, no general dimensional library tracks the distinction between field amplitudes and power quantities. `optcon` addresses these issues by introducing an explicit algebraic type layer and invariant contract system designed specifically for optical physics.

# Core Architecture & Innovations

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
`optcon` converts physical laws into executable contracts:
- `assert_unitary(M)`: Verifies that $M^\dagger M = I$ (energy-preserving, lossless);
- `assert_passive(M)`: Checks that the maximum singular value does not exceed unity ($\sigma_{\max}(M) \le 1$);
- `assert_reciprocal(M)`: Enforces symmetry in the amplitude basis ($M = M^T$);
- `dot_test(forward, adjoint, x)`: Implements Claerbout's adjoint dot-product test [@Claerbout1992], ensuring $\langle Jv, w \rangle = \langle v, J^T w \rangle$ for random probes $v, w$, thereby guaranteeing the mathematical correctness of gradients used in differentiable optics [@Hughes2019].

## 4. Boundary Enforcement with Plain Internal Numbers
To ensure performance and ease of adoption, `optcon` provides the `@checked` decorator. Following the design principle established by `prysm` [@Dube2019] and `POPPY` [@Perrin2012], units are validated and converted at function boundaries, while function bodies compute with bare, high-performance NumPy arrays.

# Cross-Engine Differential Testing

`optcon` integrates a registry and adapter layer for external optical solvers, enabling multi-engine differential testing against independent implementations and textbook closed forms [@Siegman1986; @Bohren1983]. This approach has uncovered notable undocumented behaviors in established libraries:

| Target System | Independent Engines Compared | Maximum Discrepancy | Adjudication & Finding |
| :--- | :--- | :--- | :--- |
| **Mie Scattering** | `miepython` vs. `PyMieScatt` [@Sumlin2018] | $1.9 \times 10^{-3}$ (0.2%) | Resolved by an independent Mie series: `PyMieScatt` defaults to air ($n_{\text{medium}} = 1.00027316$) rather than vacuum. Explicit specification restores agreement to $6.4 \times 10^{-14}$. |
| **Beam Propagation** | `LightPipes` (`Forvard` vs. `Fresnel`) | $6.9 \times 10^{-2}$ (6.9%) | Comparison with the analytical Gaussian beam $w(z) = w_0 \sqrt{1 + (z/z_R)^2}$ shows `Forvard` (spectral) agrees to $2.1 \times 10^{-6}$, whereas `Fresnel` (convolution) is consistently $2\%\sim 7\%$ wide, independent of grid resolution. |
| **Thick Lens ABCD** | `optcon.elements` vs. `optiland` | $1.0 \times 10^{-6}$ | Demonstrates that curved-surface ray-transfer matrices require an explicit $n_1/n_2$ index scaling factor in the lower-right entry. |
| **Thin-Film Stacks** | `tmm_core` (nm) vs. `tmm_fast` (SI m) [@Byrnes2016] | $4.1 \times 10^{-8}$ | Validates agreement across distinct internal coordinate systems. |

Engine integrations adhere to a two-tier support classification: (1) *Tier-1 Fully Supported Adapters* (`miepython`, `PyMieScatt`, `LightPipes`, `tmm`), which are covered by continuous differential verification tests with automated graceful skips when optional packages are absent; and (2) *Tier-2 Architectural Specifications* (`optiland`, `poppy`, `ceviche`), defining target schemas for community-driven solver extensions.

![Three-Way Adjudication of Mie Extinction Efficiencies across Independent Engines](docs/figures/fig2_mie_adjudication.png)

![Gaussian Beam Propagation Comparison: Spectral vs Convolution](docs/figures/fig3_beam_propagation_anomaly.png)

# Performance Highlights

By exploiting the spatial separability of the Hermite-Gauss modal basis $u_m(x) u_n(y)$, `optcon` converts 2D mode evaluations into outer matrix products:
$$C = \Delta x^2 \left( U^\dagger F U^* \right)^T$$
On a $512 \times 512$ grid with 3rd-order modes, this optimization achieves a **460x speedup** ($0.277\text{ s} \to 0.0006\text{ s}$) and reduces peak memory from $16.0\text{ MiB}$ to $0.1\text{ MiB}$. Fresnel propagation transfer functions are similarly evaluated with separable 1D exponentials and `scipy.fft`, executing in under $8\text{ ms}$.

# Author Contributions & AI Disclosure

**Author Contributions**: The conceptual architecture, physical formulation (including the independent angle dimension and amplitude order monoid), mathematical proofs, and core algorithmic decisions were conceived and verified by the authors. 

**Generative AI Disclosure**: Generative AI tools (including Google DeepMind Antigravity / Gemini) were employed as pair-programming and refactoring assistants to aid in code drafting, docstring expansion, test coverage amplification, and initial LaTeX/Markdown compilation. In accordance with JOSS Editorial Policies, the authors independently audited, executed, reviewed, and experimentally validated all code, proofs, numerical benchmarks, and manuscript text, and assume full scientific and legal responsibility for the integrity and reproducibility of all submitted materials.

# Acknowledgements

The authors acknowledge the open-source optical physics community, whose specialized solvers provided the foundation for differential benchmarking, and the Research School of Physics at The Australian National University.

# References
