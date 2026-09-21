# Compositional Evidence Contracts and Decision-Stability Gates

## Goal

Make `optcon`'s evidence-promotion method a reproducible computational method rather than a manually populated Boolean gate, while preserving the existing public API and adding a reader-friendly project explanation.

## Motivation and Contribution Boundary

The current implementation already demonstrates that finite numerical output, dimensional validity, local physical contracts, reference comparison, provenance, and downstream actionability are distinct evidence categories. Its main innovation risk is that `EvidenceRecord` can look like five hand-written flags. This change makes the evidence boundary explicit and compositional, then tests whether decision conclusions are stable against declared numerical and reference error.

The change does not claim a universal accuracy certificate, formal verification of arbitrary optical software, or a new propagation algorithm. It provides a typed evidence graph, a fail-closed promotion evaluation, and a scalar stability margin for declared decision boundaries.

## Design

### 1. Composable evidence graph

Add immutable evidence nodes to the existing `evidence.py` module. Each node has a stable identifier, one or more evidence categories, a source description, a scope description, and optional parent node identifiers. `EvidenceGraph` stores nodes, validates identifiers and parent references, and computes the transitive category closure for any target node.

The graph API will expose:

- `add(node)` for explicit construction;
- `available_categories(node_id)` for transitive evidence closure;
- `missing_categories(node_id, requirements)` for a deterministic blocking certificate;
- `evaluate(node_id, claim)` for an `EvidenceDecision` that retains the graph trace.

The existing `EvidenceRecord`, `EvidenceRequirements`, `EvidenceClaim`, `evaluate_evidence`, and `evaluate_claim` APIs remain valid. A compatibility helper converts the graph closure into an `EvidenceRecord`; existing examples and benchmark rows therefore remain reproducible while new code can avoid manually duplicating Boolean records.

Graph semantics are deliberately simple and auditable:

1. A node contributes only the categories explicitly attached to it.
2. A target inherits the union of categories attached to its ancestors.
3. Missing required categories remain blocking; categories are non-compensatory.
4. Parent references must exist and cycles are rejected.
5. Scope remains explicit metadata; the graph does not silently infer that two different physical scopes are equivalent.

### 2. Decision-stability margin

Add a small `decision_stability.py` module with a scalar threshold primitive. A declared decision supplies a value, a boundary, a conservative non-negative error budget, and a direction (`at_least` or `at_most`). The result reports:

- signed distance to the decision boundary;
- total error budget;
- residual stability margin;
- `stable` status, true only when the residual margin is strictly positive.

The module also provides a two-candidate choice margin based on the gap between the best and second-best candidate. This covers threshold decisions and discrete design choices without pretending that all optical decisions share one uncertainty model.

`evaluate_claim` gains an optional stability result. When supplied, evidence completeness and positive stability are both required for `decision_ready`; callers that omit stability retain the current behavior. A failed stability condition is reported separately from missing evidence so an unstable but well-documented result cannot be mistaken for an under-documented result.

### 3. Independent mutation and decision benchmark

Add a separate benchmark module with a frozen manifest of at least 24 cases: faults and valid controls across Mie conventions, thin-film layer definitions, sampled-beam propagation, and decision thresholds. The benchmark will use independent mutation functions and a small set of independent analytic or package-backed reference functions rather than reusing the existing held-out manifest.

The benchmark will report row-level CSV data and aggregate:

- fault detection rate;
- valid-control acceptance rate;
- incorrect-decision promotion rate;
- blocking-category attribution;
- runtime per case.

It will be explicitly described as an author-constructed robustness benchmark, not as a population-level sensitivity estimate. Its purpose is to demonstrate that the new graph and stability layers generalize beyond the three original decision-impact examples.

### 4. Documentation and manuscript integration

Update public API documentation, the reproducibility guide, and the CPC manuscript's contribution, method, results, discussion, limitations, and data-availability wording. Generate machine-readable graph and stability benchmark outputs through the existing figure/evidence generation workflow.

Create `docs/project_explainer.html` as a self-contained, dependency-free page for non-specialist readers. It will explain the problem, the layered solution, three concrete examples, what the evidence gate can and cannot guarantee, and how to reproduce the project. The page will use plain language, short sections, accessible colors, and no domain knowledge beyond basic “input, calculation, result” concepts.

## Error Handling

- Duplicate graph node identifiers raise `ValueError`.
- Missing parent identifiers raise `ValueError` at insertion time.
- Cycles raise `ValueError` when detected during graph validation/evaluation.
- Negative or non-finite stability budgets raise `ValueError`.
- Invalid threshold directions raise `ValueError`.
- Stability failures never raise by default; they return a structured unstable result so callers can report the reason and decide whether to block promotion.

## Testing Strategy

Tests will be written before implementation for:

- graph closure, missing-category certificates, duplicate/missing/cyclic nodes, and monotonicity;
- threshold and choice stability margins, invalid inputs, and integration with claim evaluation;
- independent benchmark manifest size, controls, fault detection, promotion blocking, and deterministic manifest hash;
- HTML document existence, required headings, no manuscript-only paths, and plain-language content markers.

The existing full test suite, Ruff, Mypy, strict engine status, complete benchmark suite, figure generation, and LaTeX build remain release gates.

## Acceptance Criteria

1. Existing evidence tests and callers pass unchanged.
2. New graph and stability tests pass with deterministic outputs.
3. The independent benchmark contains at least 24 frozen cases, accepts every valid control, detects every declared fault, and reports no faulty decision as stable and evidence-complete.
4. The CPC manuscript reports the new method without overstating it as a formal accuracy certificate.
5. `docs/project_explainer.html` opens as a standalone local file and explains the system to a non-specialist reader.
6. All generated records identify the source revision and remain reproducible from a clean supplement archive.
