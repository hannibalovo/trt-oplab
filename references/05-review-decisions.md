# 05. Review: make a decision from evidence

A result is an experiment, not a win, until its end-to-end acceptance gate
passes. Predeclare the target workload and accuracy tolerances.

## Gate table

| Stage | Minimum objective evidence | Common incorrect conclusion |
|---|---|---|
| Baseline | real workload + environment + hotspot evidence | `LayerInfo` is a runtime latency profile |
| Capture | Q/K/V/O + mask/cache/layout/provenance and replay check | Debug Engine outputs are bitwise production Engine internals |
| Local | finite numeric check + kernel-only AND full-call latency | faster kernel implies faster system |
| Integration | new Engine + LayerInfo diff + same-input numeric comparison | replaced Layer means adjacent tactics untouched |
| E2E | matched continuous replay + quality guardrail + wall latency | single forward allclose implies multi-chunk correctness |

## Failure decision tree

- **Wrong numerics in local** → audit boundary, scale/mask, shape/strides, RoPE,
  history-vs-current KV, dtype, initial state; do not profile a broken candidate.
- **Fast kernel, slow full-call** → quantify pack/layout conversion, synchronization,
  scratch and dispatch; consider different operator boundary.
- **Fast local, slow Engine** → inspect new reformat, missing grouped fusions,
  tactic shift, Plugin overhead and cross-block changes; A/B with same fixture.
- **Good single-forward, failed multi-chunk** → verify commit/discard, KV
  ownership, state lifecycle, cumulative numeric drift and duplicate views.
- **No improvement but correctness passes** → reject or narrow the targeted
  workload; do not claim general speedup.

## Review rubric (human decision, not auto-scored)

- Hypothesis falsifiable? Clear benchmark boundary and baseline?
- Input and outputs truly matched? Shapes/dtypes/cache semantics preserved?
- Numerics tolerance predeclared and evaluated on relevant inputs?
- Local K/O/I cost reported without double-counting?
- Structural diff and actual Nsys GPU timeline both inspected?
- End-to-end application (generate+commit, quality) passes?
- Candidate and full evidence reproducible from pinned versions/config?

Outcomes: **accept** / **iterate** / **reject** / **blocked** with explicit
reason, linked artifacts and open risks. Fill `templates/decision.md`.

## Generalization claims

One snapshot ≠ one workload distribution. One-layer Plugin ≠ whole-Engine proof.
One full-cache B-state ≠ empty/partial/ring-buffer KV coverage. Report the
validated scope. Prefer a *workload-specific* win over a false universal claim.

## v0.3: explicit reported-vs-measured gates

The project validator's `passed=true` is **local numerical / behavior evidence**.
The offline `helpers/evidence_audit.py` only audits reported JSON checks and
cannot verify raw captured tensors. A reduced example cannot be compared to a
full-size original TRT kernel timing as a speedup baseline. An experiment's
`local` stage may be marked PASS when benchmark coverage is recorded and
comparison explicitly `unavailable`, but the decision must remain non-accepting
until real integrated Engine and end-to-end measurements meet acceptance.
`integration` requires `numeric_validation_pass=true`, `nan_inf_count=0` and
measured baseline/candidate Engine costs. Do not equate `max_abs_error` to a
per-element `atol + rtol*abs(ref)` gate.
