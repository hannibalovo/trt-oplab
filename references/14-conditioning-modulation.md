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

## Decision outputs

For each candidate report: precise ONNX provenance, dependency classification (static/schedule-dependent/hidden-state-dependent), repeated-input evidence, physical-traffic evidence or unknown, expected gross savings, added memory/copy/binding/tactic costs, correctness gate, and minimum isolated A/B. This page records **hypotheses to test**, not an established optimization win.
