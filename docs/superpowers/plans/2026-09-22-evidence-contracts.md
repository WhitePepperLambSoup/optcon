# Compositional Evidence Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compositional evidence graph, decision-stability margins, an independent 24-case protocol benchmark, and a plain-language HTML project explainer without breaking existing optcon callers.

**Architecture:** Extend `evidence.py` with immutable graph nodes and transitive category closure while keeping the current Boolean APIs as compatibility wrappers. Add `decision_stability.py` for threshold and two-choice margins, then let claim evaluation optionally require a stable margin. Add a separate benchmark module that constructs fresh graph/stability cases across four optical decision families and exports row-level CSV plus aggregate metrics. Update manuscript/reproducibility outputs and create a self-contained HTML explainer.

**Tech Stack:** Python 3.11+, dataclasses, NumPy/SciPy, pytest, Ruff, Mypy, CSV, existing Matplotlib/LaTeX generation scripts, standalone HTML/CSS.

## Global Constraints

- Preserve the existing `EvidenceRecord`, `EvidenceRequirements`, `EvidenceClaim`, `evaluate_evidence`, `evaluate_claim`, and `require_decision_ready` behavior for callers that do not opt into graph or stability evaluation.
- Use only the existing runtime dependencies; do not add a new required package.
- Keep graph scope metadata explicit; never infer that two physical scopes are equivalent.
- A stability result is conservative: a zero or negative residual margin is unstable.
- The benchmark is author-constructed robustness evidence, not a population-level sensitivity estimate.
- Use ASCII source text unless an existing file already requires another encoding.
- Run focused tests after each implementation task and the complete verification suite before claiming completion.

---

### Task 1: Add the compositional evidence graph

**Files:**
- Modify: `evidence.py`
- Modify: `tests/test_evidence.py`
- Modify: `__init__.py`

**Interfaces:**
- `EvidenceNode(node_id: str, categories: frozenset[str], source: str, scope: str, parents: tuple[str, ...] = ())` is an immutable graph node.
- `EvidenceGraph(nodes: Iterable[EvidenceNode] = ())` stores validated nodes.
- `EvidenceGraph.add(node) -> None` inserts a node and rejects duplicate IDs or unknown parents.
- `EvidenceGraph.available_categories(node_id) -> tuple[str, ...]` returns the ordered transitive category closure.
- `EvidenceGraph.as_record(node_id) -> EvidenceRecord` converts the closure to the compatibility record.
- `EvidenceGraph.missing_categories(node_id, requirements) -> tuple[str, ...]` returns the blocking categories.
- `EvidenceGraph.evaluate(node_id, claim, stability=None) -> EvidenceDecision` evaluates the claim with graph provenance.

- [ ] **Step 1: Write failing graph tests**

Add tests in `tests/test_evidence.py` that assert:

```python
def test_evidence_graph_inherits_categories_from_parents():
    graph = EvidenceGraph(
        [
            EvidenceNode("semantic", frozenset({"semantic"}), "unit test", "field"),
            EvidenceNode("reference", frozenset({"independent_reference"}), "TMM", "stack"),
            EvidenceNode("result", frozenset({"numerical"}), "solver", "stack", ("semantic", "reference")),
        ]
    )
    assert graph.available_categories("result") == (
        "semantic", "numerical", "independent_reference"
    )
```

Also cover missing-category certificates, compatibility conversion to `EvidenceRecord`, duplicate IDs, unknown parents, cycle rejection through constructor validation, and the monotonicity of adding a parent evidence node.

- [ ] **Step 2: Run the focused tests and verify the intended red failure**

Run:

```text
python -m pytest tests/test_evidence.py -q
```

Expected: the new graph tests fail with import or attribute errors because the graph types do not yet exist; the pre-existing evidence tests remain runnable.

- [ ] **Step 3: Implement graph types and closure**

In `evidence.py`, add `_REQUIREMENT_NAMES` validation helpers, the frozen `EvidenceNode`, and `EvidenceGraph`. Use a DFS with `visiting` and `visited` sets for closure so cycles raise `ValueError`. Keep node categories restricted to `_REQUIREMENT_NAMES`; return categories in that stable order. Add `graph_node_id: str | None = None` and `evidence_categories: tuple[str, ...] = ()` to `EvidenceDecision` with defaults so old constructors remain valid. Make `EvidenceGraph.evaluate` call the existing evaluator and fill those trace fields.

- [ ] **Step 4: Export the graph API**

Add `EvidenceGraph` and `EvidenceNode` to the package's public import surface and sorted `__all__` in `__init__.py`.

- [ ] **Step 5: Run focused tests and static checks**

Run:

```text
python -m pytest tests/test_evidence.py -q
python -m ruff check evidence.py tests/test_evidence.py __init__.py
python -m mypy evidence.py
```

Expected: all focused tests pass and the three commands exit with code 0.

- [ ] **Step 6: Commit the graph implementation**

```text
git add evidence.py tests/test_evidence.py __init__.py
git commit -m "feat: add compositional evidence graph"
```

### Task 2: Add decision-stability margins and claim integration

**Files:**
- Create: `decision_stability.py`
- Modify: `evidence.py`
- Create: `tests/test_decision_stability.py`
- Modify: `tests/test_evidence.py`
- Modify: `__init__.py`

**Interfaces:**
- `ErrorBudget(numerical: float = 0.0, reference: float = 0.0, parameter: float = 0.0)` exposes a non-negative `total` property.
- `DecisionStability(decision_value, boundary, budget, signed_distance, margin, stable, direction)` is the threshold result.
- `threshold_stability(value, boundary, *, numerical_error=0.0, reference_error=0.0, parameter_error=0.0, direction="at_least") -> DecisionStability` computes the conservative residual margin.
- `ChoiceStability(selected, runner_up, gap, error_budget, margin, stable, maximize)` is the two-choice result.
- `choice_stability(values: Mapping[str, float], *, error_budget=0.0, maximize=True) -> ChoiceStability` selects the best and runner-up and subtracts the declared budget.
- `evaluate_claim(claim, record, stability=None)` keeps old behavior when `stability` is omitted and blocks readiness when a supplied result is unstable.

- [ ] **Step 1: Write failing stability tests**

Create `tests/test_decision_stability.py` covering positive, zero, and negative margins, both threshold directions, component-wise error budgets, non-finite/negative inputs, choice gaps, ties, and too few candidates. Extend `tests/test_evidence.py` with:

```python
def test_claim_with_unstable_result_is_not_decision_ready():
    claim = EvidenceClaim("near-boundary", "scalar", "threshold", "margin > 0", "test", "test", "demo")
    record = EvidenceRecord(True, True, True, True, True)
    stability = threshold_stability(1.0, 1.0, reference_error=0.01)
    result = evaluate_claim(claim, record, stability)
    assert result.decision_ready is False
    assert result.failed_requirements == ()
    assert result.stability is stability
    assert result.blocking_reasons == ("stability",)
```

- [ ] **Step 2: Run the focused tests and verify the red failure**

Run:

```text
python -m pytest tests/test_decision_stability.py tests/test_evidence.py -q
```

Expected: the new module/imports and optional claim field fail before implementation.

- [ ] **Step 3: Implement stable threshold and choice calculations**

Implement finite-value validation in `decision_stability.py`. For `at_least`, use `signed_distance = value - boundary`; for `at_most`, use `boundary - value`; set `margin = signed_distance - budget.total` and `stable = margin > 0`. For choices, sort deterministically by value and key, compute the best-runner gap, and require `margin > 0`; reject ties when the budget is zero only if the resulting margin is not positive.

- [ ] **Step 4: Integrate stability into evidence decisions**

Import the stability result type into `evidence.py`, add an optional `stability` field and a `blocking_reasons` property to `EvidenceDecision`, and update `evaluate_claim` so evidence failures and stability failure are reported independently. Keep `status` as `ready` or `blocked` for compatibility.

- [ ] **Step 5: Export and test the stability API**

Export the new public names in `__init__.py`, then run:

```text
python -m pytest tests/test_decision_stability.py tests/test_evidence.py -q
python -m ruff check decision_stability.py evidence.py tests/test_decision_stability.py tests/test_evidence.py __init__.py
python -m mypy decision_stability.py evidence.py
```

- [ ] **Step 6: Commit the stability implementation**

```text
git add decision_stability.py evidence.py tests/test_decision_stability.py tests/test_evidence.py __init__.py
git commit -m "feat: add decision stability margins"
```

### Task 3: Add the independent protocol benchmark

**Files:**
- Create: `benchmarks/compositional_protocol.py`
- Create: `tests/test_compositional_protocol.py`
- Modify: `benchmarks/run_all.py`
- Modify: `private-paper/docs/generate_figures.py`
- Create: `private-paper/docs/data/compositional_protocol_summary.csv` through the generator
- Create: `private-paper/docs/data/compositional_protocol_cases.csv` through the generator

**Interfaces:**
- `PROTOCOL_MANIFEST` contains 24 frozen cases: 12 controls and 12 faults across `mie`, `thinfilm`, `beam`, and `decision` families.
- `run_protocol_benchmark(output_path=None, summary_path=None) -> list[ProtocolCaseResult]` evaluates every case and writes optional CSV files.
- `protocol_summary(rows) -> ProtocolSummary` returns detection rate, control acceptance rate, faulty promotion rate, and runtime statistics.
- `main() -> int` runs the benchmark and prints a concise summary.

- [ ] **Step 1: Write failing benchmark tests**

Create tests that assert the manifest has exactly 24 unique IDs, 12 controls, 12 faults, four families, a stable manifest hash, every control is accepted, every fault is detected or blocked, and `faulty_decision_ready == 0`. Assert CSV headers and deterministic summary fields.

- [ ] **Step 2: Run the focused benchmark tests and verify the red failure**

Run:

```text
python -m pytest tests/test_compositional_protocol.py -q
```

Expected: import failure because the new benchmark module does not yet exist.

- [ ] **Step 3: Implement independent case builders**

Use fresh case definitions rather than importing `HELDOUT_MUTATION_MANIFEST`. Each case should build an `EvidenceGraph`, an `EvidenceClaim`, and either `threshold_stability` or `choice_stability`. Include four families with three controls and three faults each. Faults must vary one missing evidence category, one unstable margin, or one mismatched graph parent; controls must provide complete evidence and positive margin. Keep all numerical values finite and store the mutation description, blocking reasons, stability margin, and elapsed seconds.

- [ ] **Step 4: Implement CSV output and aggregate metrics**

Write case-level columns `case_id,family,is_fault,decision_ready,detected,stable,margin,blocking_reasons,manifest_hash,elapsed_seconds` and summary columns `total_cases,faults,controls,detected_faults,accepted_controls,faulty_decision_ready,fault_detection_rate,control_acceptance_rate,incorrect_promotion_rate,manifest_hash`. Sort rows by manifest order and use a SHA-256 hash of the canonical manifest JSON.

- [ ] **Step 5: Add the benchmark to reproducibility execution**

Add the benchmark to `benchmarks/run_all.py` as a named passing experiment and call it from `generate_figures.py` so the two CSV files are regenerated with the other evidence data. Keep the output under the existing `private-paper/docs/data` path.

- [ ] **Step 6: Run benchmark tests and inspect outputs**

Run:

```text
python -m pytest tests/test_compositional_protocol.py -q
python -m optcon.benchmarks.compositional_protocol
```

Expected: 24/24 cases satisfy expected outcomes, 12/12 faults detected, 12/12 controls accepted, and 0 faulty decisions promoted.

- [ ] **Step 7: Commit the benchmark**

```text
git add benchmarks/compositional_protocol.py tests/test_compositional_protocol.py benchmarks/run_all.py private-paper/docs/generate_figures.py
git commit -m "feat: benchmark compositional promotion protocol"
```

### Task 4: Update manuscript, reproducibility docs, and create the plain-language HTML guide

**Files:**
- Create: `docs/project_explainer.html`
- Modify: `README.md`
- Modify: `docs/REPRODUCIBILITY.md`
- Modify: `private-paper/docs/CPC_METHODOLOGY.md`
- Modify: `private-paper/docs/paper_cpc.tex`
- Modify: `tests/test_package.py`
- Create: `tests/test_project_explainer.py`

- [ ] **Step 1: Write failing HTML/documentation tests**

Add tests that read `docs/project_explainer.html` and assert it contains the title, sections `What problem does it solve?`, `How the project checks results`, `Three easy examples`, `What it does not guarantee`, and `How to reproduce it`, plus no `paper_cpc.tex`, `.bib`, or private review path. Add a package test that checks the file exists and the benchmark module imports.

- [ ] **Step 2: Run the documentation tests and verify the red failure**

Run:

```text
python -m pytest tests/test_project_explainer.py tests/test_package.py -q
```

Expected: the new HTML test fails because the file does not yet exist.

- [ ] **Step 3: Write the standalone HTML guide**

Create a single HTML file with embedded CSS and no external assets. Explain in plain language that the project checks whether an optical calculation is meaningful before it is used for a design choice. Use sections for the problem, the four checking layers, the evidence graph, stability margin, three examples, the limits, and reproduction. Use one compact diagram made with semantic HTML/CSS boxes rather than SVG or JavaScript. Keep the page readable when opened directly from disk.

- [ ] **Step 4: Update technical documentation and manuscript positioning**

Describe the graph closure and stability margin as the new method contribution, add the benchmark outputs and limitations, and explicitly retain the non-claims about universal accuracy, formal proof, and independent population-level sensitivity. Add exact reproduction commands and point readers to the HTML guide.

- [ ] **Step 5: Run documentation tests and lint**

Run:

```text
python -m pytest tests/test_project_explainer.py tests/test_package.py -q
python -m ruff check .
python -m mypy .
```

### Task 5: Full verification, paper build, and supplement refresh

**Files:**
- Regenerate: `private-paper/docs/data/compositional_protocol_cases.csv`
- Regenerate: `private-paper/docs/data/compositional_protocol_summary.csv`
- Regenerate: `private-paper/docs/figures/*` and figure manifest as needed
- Regenerate: `private-paper/docs/paper_cpc.pdf`
- Create: new `private-paper/optcon-cpc-supplement-*.zip`

- [ ] **Step 1: Run the complete source verification**

Run:

```text
python -m pytest tests -q
python -m ruff check .
python -m mypy .
python -m optcon.benchmarks.engine_status --strict
python -m optcon.benchmarks.run_all
```

- [ ] **Step 2: Regenerate figures and compile the manuscript**

Run:

```text
python private-paper/docs/generate_figures.py
pdflatex -interaction=nonstopmode -halt-on-error paper_cpc.tex
bibtex paper_cpc
pdflatex -interaction=nonstopmode -halt-on-error paper_cpc.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_cpc.tex
```

Inspect `paper_cpc.log` for errors, undefined citations, and overfull boxes, and inspect the generated PDF page count.

- [ ] **Step 3: Build and verify a clean supplement**

Run:

```text
powershell -NoProfile -ExecutionPolicy Bypass -File .\private-paper\build_submission_supplement.ps1
```

Extract the newest archive into a fresh temporary directory, verify the top-level `optcon` folder, verify every line in `SHA256SUMS.txt`, confirm no manuscript source/PDF/bibliography is present, install `.[dev,engines]`, and rerun the full tests, benchmark suite, and figure generator from the extracted archive.

- [ ] **Step 4: Review the final diff and commit release updates**

Run:

```text
git diff --check
git status -sb
git diff --stat HEAD~5..HEAD
```

Commit any final generated-source or documentation updates with:

```text
git add README.md docs private-paper/docs benchmarks tests evidence.py decision_stability.py __init__.py
git commit -m "feat: add compositional evidence and decision stability"
```

