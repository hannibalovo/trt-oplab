---
name: trt-oplab
description: >-
  Evidence-driven GPU operator extraction, standalone replay, kernel/operator
  benchmarking, TensorRT integration, and end-to-end regression review.
  Use when optimizing Attention/FFN/KV operations in an existing inference engine,
  evaluating a plugin boundary, or deciding whether a local speedup is real.
---

# Operator Optimization Workbench

Purpose: make **one optimization hypothesis** reproducible and reviewable. This
skill is the *navigation and gatekeeping entrypoint*, not a substitute for code,
measurements, existing profiling skills, or the project's runtime.

## Core rule

Do not conflate these results:
1. captured **reference numerical output** with the exact production-engine internal output;
2. **kernel-only latency** with full operator-call latency;
3. operator-call latency with **integrated engine latency**;
4. engine latency with **application end-to-end latency**;
5. TensorRT **Inspector layer** with one CUDA kernel launch.

Explicitly label unknowns and never report a performance win without its
measurement boundary, workload, units, and comparison artifact.

## Quickstart (8 steps; v0.3)

1. **Define** target workload, hypothesis, metric and fallback; record the
   production path and runtime call boundary. If initialization can be lazy,
   define whether it belongs to startup or first-request cost. See
   `references/01-scope-baseline.md` and `references/09-engine-lifecycle.md`.
2. **Inspect** the baseline Engine/trace, find target ONNX node(s), TRT layer(s),
   and actual GPU kernel(s). `python helpers/layerinfo.py summary BASE.json`.
3. **Extract** authentic inputs and reference outputs with shape/dtype/stride,
   scale/mask, cache state and provenance. `references/02-capture-contract.md`.
4. **Replay and validate** with the project's existing real-data harness (do not
   rewrite it). Keep frozen Q/K/V/O + FP32 math reference, source/manifest SHA,
   tolerance, negative tests, caller output guards, repeat and perturbation
   checks. `references/07-validation-harness.md`.
   For offline JSON report consistency: `python helpers/evidence_audit.py --help`.
5. **Measure** on original production-shaped data, not a reduced validation
   fixture. Record candidate K and complete O; only compare to original TRT at
   demonstrably matched boundaries. For paged KV, first state the Engine ABI
   and cache lifecycle contract. See `references/08-benchmark-protocol.md` and
   `references/10-paged-kv-contract.md`.
6. **Choose boundary** deliberately; integrate only after local evidence.
   ONNX rewrite + Plugin/ordinary TRT graph; see `references/04-integration.md`.
7. **Compare engines** and rerun application replay. `python helpers/layerinfo.py diff BASE.json CANDIDATE.json`.
8. **Decide** accept/iterate/reject based on correctness, local, integration,
   end-to-end evidence and uncertainty. See `references/05-review-decisions.md`.

## Experiment state (machine-readable)

Initialize: `python helpers/experiment.py init --out RUN_DIR --hypothesis "..." --target "..."`

Check: `python helpers/experiment.py status RUN_DIR`

Update only with real artifacts: `python helpers/experiment.py stage RUN_DIR --name capture --status pass --evidence relative/path.pt --note "..."`

Stages: `baseline → capture → local → integration → e2e`.
Each passed stage requires existing evidence file(s), a non-empty explanation,
and required numeric checks when applicable. v0.3 uses explicit
`numeric_validation_pass=true`, not scalar `max_abs_error <= atol`: the
project validator must implement the actual per-element mixed tolerance rule.
`benchmark_comparison_status=unavailable` never implies a speedup. A human/agent *attests* the evidence;
the helper verifies records, not mathematical truth.

## Reference index

- `references/01-scope-baseline.md` — workload, target hotspot, baseline, cost model
- `references/02-capture-contract.md` — extraction boundary, tensor provenance, replay fidelity
- `references/03-local-harness.md` — local replay, correctness, three-tier latency
- `references/04-integration.md` — Plugin boundary, graph diff, runtime registration
- `references/05-review-decisions.md` — gates, failure attribution, accept/reject rules
- `references/06-governance.md` — how skills/helpers/experiments evolve, avoiding overengineering
- `references/07-validation-harness.md` — real tensor fixtures, candidate interface, negative controls, report-strength limits
- `references/08-benchmark-protocol.md` — matched production-shape K/O/I measurement, original-TRT baseline constraints
- `references/09-engine-lifecycle.md` — Engine initialization, cold/hot file cache, graph setup, first-use and steady-state boundaries
- `references/10-paged-kv-contract.md` — dense versus paged Engine ABI, page-table/cache invariants and staged integration checks

## Helper index

- `helpers/layerinfo.py` — **offline**, stdlib-only Inspector summary and conservative MHA diff
- `helpers/experiment.py` — experiment state, evidence checks, append-only stage history
- `helpers/evidence_audit.py` — offline JSON **report consistency** audit; no GPU, tensor values, numerical recomputation or performance proof

## Critical lessons

- In production TensorRT, insertion of an output or a Debug Tensor may alter fusion,
  layout, numerics, or tactics. A debug Engine is **not automatically the original** Engine.
- A Plugin can replace a compute region but often prevents native optimization *through*
  its body. Smallest Plugin is not automatically fastest; largest is not either.
- A standalone replay should not become an accidental claim that full-Engine
  inference or multi-step KV semantics have passed.
- A validator PASS on a **sliced contiguous** Q/K/V fixture is not proof of
  production-strided operation, performance, Plugin integration or E2E quality.
- Do not call same-named layers identical across workload modes or distinct
  Engine hashes; compare workload, source, shape and Tactic.
- Never package raw project tensors, models, proprietary Engine summaries or
  credentials in a portable Skill installer.
- Don't build a multi-skill orchestrator until repeated independent workstreams
  truly require separate ownership. Reuse existing Nsight/KernelWiki skills by links.
