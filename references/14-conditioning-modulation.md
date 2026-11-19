# 14. Conditioning / AdaLN: hypothesis triage before graph surgery

A fused TensorRT Myelin kernel may prepare per-block time-embedding modulation slots (shift, scale, gate), apply normalization to current hidden states and expose many logical Split/Transpose outputs. It is **not** a pure layout-copy kernel merely because the name mentions Transpose/Slice.

## Partition the dependency graph

```text
time step → time embedding → block-specific scale/shift/gate
                                      ↓
current hidden state → normalization → modulation → downstream blocks
```

Do **not** cache the entire fused Layer if it contains dynamic hidden-state normalization. Time steps change within a denoising schedule. First establish whether the *same position in the schedule* repeats across chunks and whether any other runtime condition changes the modulation output.

## Two different hypotheses

**A. Schedule reuse:** precompute small per-block modulation vectors for recurring schedule positions, reuse across chunks. Compare preparation cost, compact cache size, new bindings and lost fusion against existing fused kernel cost. If the cheap precompute becomes expensive external input traffic, reject.

**B. Broadcast/materialization:** apparent tensors with a token dimension may be broadcast views or physically materialized. Shape×dtype logical size alone does not establish writes. Check supported byte counters and producer/consumer fusion/alias information. A debug allocation probe can alter fusion and requires a control.

Do **not** combine A and B in one unexplained candidate change. If no direct evidence that either saves more than graph splitting costs, retain the original Engine.

## When a fast candidate is numerically wrong

A compact modulation precompute may make the rebuilt **Main Engine** substantially faster but fail correctness in recurrent or stateful outputs. The relevant gate includes **all** declared outputs and cache/commit state, not just the most visible prediction tensor. A single-step prediction within tolerance does **not** license multi-chunk replay if its K/V outputs fail.

Two close precompute variants agreeing with each other do **not** prove either matches the baseline's internal time-modulation tensor. Precompute arithmetic and altered Main Engine fusion/tactics remain separate possible error sources.

A particularly useful warning: if the new **whole Forward** saves more time than the original fused target layer consumed, do **not** attribute the full gain to deleting that layer. The rebuild may have changed other tactics, memory planning or fusion boundaries.

## Minimum numerical attribution experiment

1. Freeze identical original input and output fixtures; keep engine, ONNX and external-weight hashes. Separate schedule/time input invariance from graph correctness.
2. Capture modulation intermediates **in a diagnostic Engine**, clearly marking that enabling an output or debug tensor can change the original tactic. This identifies plausible values, not automatically the historical runtime's exact internals.
3. Run a **boundary-matched control** Main Engine: preserve the new binding/Gather/broadcast interface while feeding values computed from a trusted/reference path. Compare with the precomputed candidate through the **same** Main graph to isolate precompute numerics from changes in Main computation.
4. Independently compare a no-graph-surgery baseline and the boundary-matched control on same inputs. Map ONNX source scopes, fused layers, tactics and kernels; differences here implicate graph/tactic effects rather than reuse alone.
5. Bisect earliest divergence (individual modulation slot → first affected block hidden state → projected/current K/V → final output), preserving FP32/BF16 casting order and broadcast semantics. Do not infer cause from aggregate final-output metrics.
6. Only after **all** outputs satisfy the existing numerical acceptance rule may the candidate progress to multi-step and multi-chunk stateful replay. Do not loosen tolerance to rescue an apparent latency win.

For a temporal embedding with multiple frame/time positions, a precompute interface accepting one timestep per batch is valid **only** when equality across all temporal positions is explicitly enforced. Deterministic scheduler source code alone is not proof that every historical call had identical timestep bytes; verify at the execution binding boundary when possible.

## Evidence classification

- **Measured:** original/candidate engine timing, output error tensors, candidate precompute agreement and resource usage.
- **Inferred:** repeated schedule positions, unchanged internal fused computation, causes of new tactic selection.
- **Unknown without controls:** exact original internal activation bits and which changed fusion/tactic first caused the error.

A faster but invalid candidate is a **rejected/blocked prototype**, never a deployable speedup. Keep the negative result as a reusable diagnostic pattern; do not expose private traces, exact benchmark numbers or internal model artifacts in this public Skill.

## Decision outputs

For each candidate report: precise ONNX provenance, dependency classification (static/schedule-dependent/hidden-state-dependent), repeated-input evidence, physical-traffic evidence or unknown, expected gross savings, added memory/copy/binding/tactic costs, correctness gate, and minimum isolated A/B. This page records **hypotheses to test**, not an established optimization win.
