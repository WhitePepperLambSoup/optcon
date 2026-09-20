# API index

Generated from the package docstrings by `docs/gen_api_index.py`; do not edit
by hand. Every public callable is listed with the first line of its summary.


## `optcon.aberrations`

Wavefront aberrations: Zernike modes, Seidel terms, and their RMS.

- `defocus_coefficient` - Zernike defocus coefficient for a given peak-to-valley wavefront.
- `noll_to_nm` - Convert a Noll index ``j`` to the pair ``(n, m)``.
- `radial_polynomial` - Zernike radial polynomial ``R_n^|m|(rho)``.
- `seidel_spherical_decomposition` - Coefficients of ``amplitude * rho^4`` in Noll order.
- `strehl_from_zernikes` - Marechal estimate of the Strehl ratio from a coefficient vector.
- `wavefront_pv` - Peak-to-valley wavefront error, evaluated on a disc.
- `wavefront_rms` - RMS wavefront error of a coefficient vector, in nanometres.
- `zernike` - The Noll-normalised Zernike mode ``Z_n^m``.
- `zernike_modes` - Every ``(n, m)`` up to ``max_order``, in Noll order.

## `optcon.beam_quality`

Beam quality: M squared and the beam parameter product (ISO 11146).

- `beam_parameter_product` - ``BPP = M^2 lambda / pi``, the waist-radius-times-half-angle invariant.
- `beam_radius_with_m_squared` - ``w(z)`` for a beam of the given ``M^2``.
- `divergence_with_m_squared` - Half angle of a non-ideal beam, ``M^2 lambda / (pi w0)``.
- `m_squared_from_parameters` - ``M^2 = pi w0 theta / lambda`` from a measured waist and half angle.
- `m_squared_from_widths` - ``M^2`` from two measured widths and the separation, inverting the hyperbola.
- `rayleigh_range_with_m_squared` - Effective Rayleigh range of a non-ideal beam, ``pi w0^2 / (M^2 lambda)``.

## `optcon.cavity`

Two-mirror optical cavities.

- `circulating_power` - Steady-state circulating power on resonance, relative to the input.
- `finesse` - Finesse of a symmetric cavity with the given mirror reflectance.
- `free_spectral_range` - Mode spacing of a cavity, ``c / (n * path)``.
- `g_parameters` - The two stability parameters ``g_i = 1 - L/R_i``.
- `is_stable_cavity` - Whether the cavity has a confined, finite-size mode.
- `linewidth` - Full width at half maximum of a resonance, ``FSR / F``.
- `photon_lifetime` - Energy decay time of the cavity, ``1 / (2 pi linewidth)``.
- `quality_factor` - ``Q = frequency / linewidth``.
- `round_trip_gouy_phase` - Gouy phase accumulated per round trip, ``arccos(sqrt(g1 g2))``.
- `spot_size_at_mirrors` - The eigenmode radius on each mirror.
- `stability_product` - ``g1 * g2``; the cavity is usable when this lies in ``[0, 1)``.
- `transverse_mode_spacing` - Frequency spacing between successive transverse modes.
- `waist_size` - Radius of the cavity eigenmode at its waist.

## `optcon.checks`

Physical invariant contracts for optical operators.

- `assert_adjoint` - Declare ``adjoint`` the adjoint of ``forward``; raise if it is not.
- `assert_gradient_matches` - Declare a gradient correct; raise if it is not the real derivative.
- `assert_passive` - Declare an operator gain-free; raise if it amplifies.
- `assert_reciprocal` - Assert matrix symmetry in the caller's matched reciprocal basis.
- `assert_unitary` - Declare an operator lossless; raise if it is not.
- `check_gradient` - Compare a supplied gradient against central differences.
- `dot_test` - Check a supplied adjoint with the inner product of the input space.
- `finite_difference_gradient` - Central-difference gradient of a scalar function.
- `finite_difference_jacobian` - Central-difference Jacobian of a vector-valued function.
- `gain_above_unity` - How far the largest singular value exceeds unity (0 for passive).
- `is_passive` - 
- `is_reciprocal` - 
- `is_unitary` - 
- `reciprocity_error` - ``max|M - M^T|``; evaluates to zero when the operator matrix is symmetric.
- `singular_values` - Singular values of the operator, in descending order.
- `unitarity_error` - ``max|M^H M - I|``; zero exactly when the operator is lossless.

## `optcon.diffraction`

Diffraction limits: the Airy pattern, resolution, and the Strehl ratio.

- `airy_intensity` - Normalised Airy intensity ``(2 J1(u) / u)^2``, equal to one on axis.
- `airy_radius` - Radius of the first dark ring of the Airy disc, ``1.22 lambda f/#``.
- `angular_resolution` - Rayleigh criterion ``1.22 lambda / D``, in radians.
- `diffraction_limited_spot` - Diameter of the Airy disc, ``2.44 lambda f/#``.
- `encircled_energy` - Fraction of the Airy pattern's power inside radius ``u``.
- `f_number` - The ``f/#`` of a system.
- `gaussian_far_field_half_angle` - Divergence of a Gaussian mode, ``lambda / (pi w0)``.
- `rms_wavefront_from_strehl` - Invert the Marechal approximation.
- `strehl_ratio` - Marechal approximation ``exp(-(2 pi sigma / lambda)^2)``.

## `optcon.elements`

Ray-transfer (ABCD) matrices for the elements an optical bench is built from.

- `chief_ray_trace` - Trace ``(y, theta)`` through ``matrix``, with ``theta`` in radians.
- `compose` - Multiply elements **in beam order**.
- `curved_mirror` - A spherical mirror: ``[[1, 0], [-2/R, 1]]`` in the reflective convention.
- `dielectric_interface` - Refraction at a plane interface: ``[[1, 0], [0, n1/n2]]``.
- `effective_focal_length` - ``-1/C``: where a collimated input comes to a focus.
- `flat_mirror` - A plane mirror: the identity in the paraxial approximation.
- `free_space` - Propagation over ``distance``: ``[[1, d], [0, 1]]``.
- `is_marginally_stable` - Whether a round trip sits exactly on the stability boundary.
- `is_stable` - Whether a round-trip matrix is stable, i.e. ``|(A + D)/2| <= 1``.
- `magnification` - Transverse magnification ``A`` of a system that images (``B = 0``).
- `refracting_surface` - Refraction at a **curved** interface: ``[[1, 0], [-Phi/n2, n1/n2]]``.
- `stability` - Half the trace, ``(A + D)/2``.
- `stability_status` - ``"stable"``, ``"marginal"`` or ``"unstable"``, for reporting.
- `thin_lens` - An ideal thin lens: ``[[1, 0], [-1/f, 1]]``.

## `optcon.evidence`

Evidence requirements for promoting numerical results to decisions.

- `evaluate_claim` - Evaluate a claim while retaining its audit metadata and evidence trace.
- `evaluate_evidence` - Evaluate whether ``record`` satisfies the required evidence categories.
- `require_decision_ready` - Return a passing result or raise a diagnostic contract failure.

## `optcon.fiber`

Step-index optical fibre.

- `acceptance_angle` - The largest external half angle that is still guided, in radians.
- `coupling_loss_db` - Coupling efficiency expressed as a positive loss in decibels.
- `cutoff_wavelength` - The wavelength above which the fibre guides only one mode.
- `is_single_mode` - Whether only the LP01 mode is guided.
- `mode_field_diameter` - Marcuse's approximation to the mode field diameter.
- `mode_field_radius` - Half the mode field diameter, in the core radius's own unit.
- `numerical_aperture` - Half-angle acceptance cone, ``sqrt(n_core^2 - n_clad^2)``.
- `propagation_constant` - ``beta`` of the fundamental mode, via Gloge's approximation.
- `v_number` - Normalised frequency ``V = 2 pi a NA / lambda``.

## `optcon.fox_li`

Fox-Li cavity diffraction integral operator for open resonators.

- `fox_li_operator` - Construct the discretized Nystrom Fredholm integral operator matrix.
- `fox_li_power_iteration` - Simulate physical wave relaxation inside the open resonator via power iteration.
- `fresnel_number` - Fresnel number N_F = a^2 / (lambda * L) of the open resonator.
- `solve_fox_li_modes` - Solve for the lowest-loss transverse eigenmodes of the open resonator.

## `optcon.fresnel`

Fresnel reflection and transmission at a plane dielectric interface.

- `amplitude_coefficient` - The field amplitude coefficient ``r`` or ``t``, as an amplitude ratio.
- `brewster_angle` - The angle at which p-polarised light is not reflected.
- `critical_angle` - The angle beyond which total internal reflection occurs.
- `fresnel_coefficients` - The four amplitude coefficients ``r_s``, ``r_p``, ``t_s``, ``t_p``.
- `reflectance` - Fraction of incident power reflected, as a power ratio.
- `snell` - Refraction angle at a plane interface, in radians.
- `transmittance` - Fraction of incident power transmitted, as a power ratio.
- `unpolarized_reflectance` - Mean of the s and p reflectances, for unpolarized light.

## `optcon.gaussian`

Gaussian beams described by the complex q parameter.

- `beam_diameter` - Convenience wrapper: ``2 w``.
- `beam_radius` - Field radius ``w`` of the beam described by ``q``.
- `divergence_half_angle` - Far-field half angle ``theta = lambda / (pi w0)``, in radians.
- `gouy_phase` - Gouy phase relative to the waist, in radians.
- `mode_matching_efficiency` - Power coupled between two Gaussian modes given by their q parameters.
- `mode_overlap` - Power coupling between two coaxial Gaussian modes at the same plane.
- `position_of_waist` - Distance from the current plane back to the waist (negative upstream).
- `propagate` - Apply an ABCD system to a beam: ``q' = (A q + B) / (C q + D)``.
- `q_from_waist` - The q parameter ``distance`` downstream of a waist.
- `radius_of_curvature` - Wavefront radius of curvature; infinite at the waist.
- `rayleigh_range` - ``z_R = pi w0^2 / lambda``.
- `self_consistent_mode` - The q parameter that maps onto itself after one ``round_trip``.
- `waist_of` - The waist radius that this q parameter corresponds to.

## `optcon.grating`

Diffraction gratings.

- `angular_dispersion` - ``d(theta)/d(lambda)`` for the given order, in radians per nanometre.
- `blaze_angle` - Littrow angle: incidence and diffraction coincide, ``sin(theta) = m lambda / 2d``.
- `free_spectral_range_of_grating` - Spacing between successive orders: ``lambda / m``.
- `grating_equation` - Diffracted angle for one order, in radians.
- `lines_per_mm_from_period` - The inverse of :func:`period_from_lines_per_mm`.
- `period_from_lines_per_mm` - Grating period ``d = 1 / (lines per mm)``.
- `resolving_power` - ``R = m N = lambda / delta-lambda``.
- `wavelength_resolution` - The smallest resolvable wavelength difference.

## `optcon.interferometry`

Interferometry: fringe visibility, coherence, and the Fabry-Perot etalon.

- `coherence_length` - ``L_c = c / delta-nu``.
- `coherence_length_from_wavelength_spread` - ``L_c = lambda^2 / delta-lambda``, the wavelength-domain equivalent.
- `coherence_time` - ``tau_c = 1 / delta-nu``.
- `etalon_transmission` - Airy transmission ``1 / (1 + F sin^2(delta/2))``.
- `fringe_visibility` - ``(Imax - Imin) / (Imax + Imin)``.
- `peak_to_minimum_ratio` - Contrast of an etalon, ``1 + F``.
- `two_beam_contrast` - Visibility of the fringes produced by two beams, as a power ratio.
- `two_beam_intensity` - Intensity of two-beam interference, ``I1 + I2 + 2 sqrt(I1 I2) cos(k dOPD)``.
- `visibility_from_intensities` - Visibility of two-beam interference with unequal beam intensities.

## `optcon.laser`

Laser physics for a four-level gain medium.

- `output_power` - Output power above threshold; zero below it.
- `photon_lifetime_cavity` - Cavity photon lifetime ``2L / (c delta)``.
- `relaxation_oscillation_frequency` - Damping frequency of the spiking oscillations, ``(1/2pi) sqrt((r-1)/(tau_p tau_c))``.
- `saturation_intensity` - ``I_sat = h nu / (sigma tau)``.
- `slope_efficiency` - Fraction of the pump power above threshold that becomes output.
- `threshold_gain` - Small-signal gain per metre needed to reach threshold.
- `threshold_pump_power` - ``P_th = I_sat A delta / 2`` for a four-level medium.

## `optcon.modes`

Gauss-Hermite and Laguerre-Gauss modes for sampled optical fields.

- `decompose` - Project ``field`` onto the requested modal basis up to ``max_order``.
- `hermite_gauss` - The normalised ``HG_mn`` mode, sampled on coordinates ``x`` and ``y``.
- `laguerre_gauss` - The normalised ``LG_p^l`` mode: a vortex of charge ``l`` when ``l != 0``.
- `mode_content` - Modes sorted by descending power fraction in the requested basis.
- `mode_power_fractions` - Fraction of the field's power carried by each mode.
- `reconstruct` - Rebuild a field from coefficients in the requested modal basis.

## `optcon.mtf`

Modulation transfer function of an imaging system.

- `circular_aperture_mtf` - Diffraction MTF of an unaberrated circular pupil.
- `cutoff_frequency` - Highest spatial frequency a circular pupil transmits, ``1/(lambda f/#)``.
- `mtf_from_psf` - Modulation transfer function from a sampled point spread function.

## `optcon.mueller`

Mueller calculus: the intensity-domain counterpart of Jones calculus.

- `apply_mueller` - ``S' = M S``.
- `degree_of_polarization` - ``sqrt(S1^2 + S2^2 + S3^2) / S0``.
- `mueller_depolarizer` - Partial depolarisation that keeps ``S0`` and shrinks the rest.
- `mueller_from_jones` - The Mueller matrix of a non-depolarising element described by Jones.
- `mueller_polarizer` - An ideal linear polariser.
- `mueller_retarder` - An ideal waveplate: lossless, so ``S0`` is untouched.
- `mueller_rotator` - A polarisation rotator.
- `stokes_vector` - Stokes vector of a Jones state, as a plain 4-vector.

## `optcon.nlse`

Generalized Nonlinear Schrodinger Equation (G-NLSE) Split-Step Fourier Solver.

- `gaussian_pulse` - Create a transform-limited Gaussian pulse A(T) = sqrt(P0) * exp(-T^2 / (2 T0^2)).
- `soliton_parameters` - Compute fundamental soliton scales: L_D, L_NL, z_0, and peak power P_0.
- `soliton_pulse` - Create a fundamental hyperbolic secant soliton pulse A(T) = sqrt(P0) * sech(T / T0).
- `solve_nlse` - Propagate pulse along distance using the symmetric Split-Step Fourier Method (SSFM).

## `optcon.noise`

Detector noise: photon energy, shot noise, signal to noise, NEP.

- `dark_current_noise` - Shot noise with the dark current included; the two add in quadrature.
- `noise_equivalent_power` - Optical power that produces a signal equal to the noise.
- `photon_energy` - ``h c / lambda``.
- `photons_per_second` - Photon arrival rate for a given optical power.
- `quantum_efficiency_from_responsivity` - The inverse of :func:`responsivity`.
- `responsivity` - Amperes per watt, ``eta q lambda / (h c)``.
- `shot_noise_current` - RMS shot noise ``sqrt(2 q (I + I_dark) B)``.
- `signal_to_noise` - Shot-noise-limited SNR of an ideal detector.

## `optcon.nonlinear`

Second-harmonic generation: phase matching and effective interaction length.

- `coherence_length` - ``lambda / (4 [n(2w) - n(w)])``; infinite under perfect matching.
- `effective_interaction_length` - ``sin(delta-k L / 2) / (delta-k / 2)``.
- `phase_mismatch` - ``delta-k`` in inverse length units.
- `quasi_phase_matching_period` - Poling period ``2 L_coh`` that restores the sign every coherence length.

## `optcon.paraxial`

Paraxial focal length of a singlet, computed by several engines.

- `thick_lens_focal_length` - Effective focal length of a symmetric biconvex singlet.

## `optcon.polarization`

Jones calculus: polarisation states and the elements that act on them.

- `apply` - Apply an operator to a state vector.
- `degree_of_polarization` - ``sqrt(S1^2 + S2^2 + S3^2) / S0``; exactly one for a pure state.
- `extinction_ratio` - Best-to-worst transmission ratio of a polariser pair.
- `half_wave_plate` - A half-wave plate: rotates linear polarisation by twice its angle.
- `intensity` - ``|Ex|^2 + |Ey|^2`` as a power ratio.
- `jones_circular` - Unit-amplitude circular polarisation.
- `jones_linear` - Unit-amplitude linear polarisation at ``angle``.
- `polarizer` - An ideal linear polariser; passive, idempotent, and not unitary.
- `quarter_wave_plate` - A quarter-wave plate: turns linear at 45 degrees into circular.
- `retarder` - A waveplate of retardance ``delta`` whose slow axis is at ``angle``.
- `rotator` - A polarisation rotator.  Unitary, but not reciprocal.
- `stokes` - The four Stokes parameters, unnormalised, as plain floats.

## `optcon.propagation`

A scalar field on a sampled grid, and FFT propagation of it.

- `gaussian_field` - A TEM00 Gaussian beam of the given 1/e^2 radius, sampled on a grid.
- `intensity` - ``|u|^2``, the quantity that becomes a power ratio once normalised.
- `max_propagation` - Distance beyond which the transfer function is undersampled.
- `power` - Total power, in units of the grid cell area.
- `propagate` - Propagate ``field`` by ``distance`` along the optical axis.
- `second_moment_radius` - The D4-sigma radius, ``2 sqrt(<x^2>)``, taken from the x marginal.

## `optcon.quantity`

Dimensional quantities with amplitude/power provenance.

- `amplitude_ratio` - A field amplitude ratio (order 1), e.g. ``t`` where ``T = |t|^2``.
- `dimensionless` - A pure ratio (order 0): counts, operators, plain numbers.
- `power_ratio` - A power ratio (order 2), e.g. ``T`` from a transmission spec.
- `q` - Build an untracked quantity: ``q(300, "mm")``.
- `sqrt` - Square root with the amplitude-order rule applied.

## `optcon.radiometry`

Radiometry and blackbody radiation.

- `etendue` - Throughput ``A Omega``; conserved by lossless optics.
- `irradiance` - Power per unit area on a surface, ``W/m^2``.
- `luminous_flux` - Photometric flux ``Phi = K P``, in lumens.
- `planck_exitance` - Spectral exitance of a Lambertian blackbody, ``pi B_lambda``.
- `planck_radiance` - Spectral radiance of a blackbody, ``W m^-2 sr^-1 m^-1``.
- `radiance` - Power per unit area per unit solid angle, ``W m^-2 sr^-1``.
- `radiance_invariance` - Ratio ``A2 Omega2 / (A1 Omega1)``; one for a lossless system.
- `solid_angle_cone` - Solid angle of a cone, ``2 pi (1 - cos(theta))``.
- `stefan_boltzmann` - Total exitance ``sigma T^4``.
- `wien_displacement` - Wavelength of peak spectral radiance, ``b / T``.

## `optcon.specs`

Annotate an existing signature, get checked units at the boundary.

- `checked` - Check declared units on the way in and on the way out.
- `to_quantities` - Turn a physics dataclass instance into typed quantities.
- `units_of_dataclass` - Infer a unit for each field from its name suffix.

## `optcon.thermal`

Thermal lensing and Gaussian aperture truncation.

- `gaussian_aperture_transmission` - Fraction of a Gaussian beam's power that clears a circular aperture.
- `gaussian_clipping_loss` - Fraction of a Gaussian beam's power lost at a circular aperture.
- `thermal_lens_focal_length` - Focal length of the lens an absorbed pump creates.

## `optcon.thinfilm`

Thin film design: quarter-wave stacks, anti-reflection coatings, Bragg mirrors.

- `anti_reflection_thickness` - Quarter-wave thickness ``lambda / (4 n)``.
- `bragg_reflectance` - Peak reflectance of a quarter-wave ``(HL)^N H`` stack.
- `ideal_anti_reflection_index` - The single-layer index that nulls reflection: ``sqrt(n0 * ns)``.
- `optical_thickness` - ``n t``: the thickness that matters for phase.
- `quarter_wave_stack_reflectance` - Convenience wrapper for an air-incident ``(HL)^N H`` stack.
- `single_layer_reflectance` - Reflectance of one film on a substrate, at normal incidence.

## `optcon.units`

Dimensional algebra for physical quantities.

- `all_registered_symbols` - Yield every bare unit symbol known to the registry (no prefixes).

## `optcon.vector_fields`

Vector fields: a spatial profile that also carries a polarisation state.

- `analyzer_transmission` - Intensity transmitted by a linear analyzer at ``analyzer_angle``.
- `from_scalar` - Give a scalar field a uniform linear polarisation at ``angle``.
- `intensity_map` - ``|Ex|^2 + |Ey|^2`` at every point.
- `polarization_map` - Azimuth of the local linear polarisation, in radians.
- `propagate_vector` - Propagate both components; the polarisation state is carried along.
- `stokes_map` - Stokes parameters at every point, with the same convention as the

## `optcon.waveguide`

Guided modes of a symmetric slab waveguide.

- `confinement_factor` - Fraction of a mode's power carried inside the core.
- `effective_indices` - Effective indices of every guided TE mode, fundamental first.
- `normalized_frequency` - ``V = k0 a NA``, with ``a`` the half thickness.
- `number_of_modes` - How many TE modes a symmetric slab with this ``V`` supports.
- `single_mode` - Whether only the fundamental mode is guided, ``V < pi/2``.

---

219 public functions across 34 modules.
