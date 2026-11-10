# 06. How to grow the workbench without building an agent bureaucracy

## Separation of concerns

- **SKILL.md** = navigation, trigger, exit conditions, pitfalls (short).
- **references/** = one stage's conceptual checklist and specific invariants.
- **helpers/** = deterministic actions with input validation, tests, explicit errors.
- **project adapters/** = paths/API/contracts to LingBot, vLLM, SGLang, GR00T etc.
- **experiments/** = immutable input snapshots, logs, profiles, build hashes,
  decisions and negative results.

Do not put 1000-line reference docs in SKILL; do not write an AI prompt for
something reliably done by a 30-line parser; do not hide experimental evidence
inside agent chat history.

## What to automate and what not

Automate: environment fingerprints, File/LayerInfo checks, shape metadata
capture, comparisons, evidence hashing, result tables and cross-run diffs.

Do **not** blindly automate: selecting a mathematically equivalent capture
boundary, declaring numerical tolerance, deciding plugin fusion boundary,
interpreting NCU cause, claiming accuracy and final shipping approval.
Those steps need explicit reasoning and experimental evidence.

## When to promote to a separate Skill

Only after repeated independent workflows have different triggers, owners,
inputs and acceptance gates. Example: a standalone architecture-gated
`trt-plugin-integration` skill may become justified if many teams share it;
until then, keep integration as a Reference and reusable scripts.

## Hidden cost model: optimizing experimentation throughput

A team's long-run optimization rate depends on more than per-kernel speed:

```
experiment_value ≈ useful_verified_findings / total_engineer_and_GPU_cost
```

Therefore measure **time-to-root-cause**, false-positive performance wins,
re-run reproducibility, GPU-build iteration cost and onboarding time. A Skill
that increases documentation but not these metrics is not succeeding.

## Evolution protocol

After each finished case, record:
- new invariant violated;
- smallest reproducible failing fixture;
- repeatable diagnosis to automate;
- decision boundary altered.

Only promote a lesson from an individual experiment into shared `references/`
after it has been verified or explicitly marked as a hypothesis.
