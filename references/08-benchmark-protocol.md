# 08. Benchmark protocol: no local PASS implies speedup

## Workload identities (do not mix)

- A **reduced correctness fixture** is valid for algebraic/caller behavior checks
  only; it is NOT a baseline for actual TensorRT kernel performance.
- A **real production-shape workload** must preserve Batch×Heads, query length,
  KV length, dtype, strides, Q/K/V packing and masks, execution state and stream.
- Different modes can have different Sq and KV state despite similar layer
  names. Pin mode and Engine hash; do not join by name or index alone.

## Three levels of cost

| Level | Boundary | Valid comparison |
|---|---|---|
| K | Candidate standalone GPU kernel(s), resident inputs | candidate variants on identical shapes and launch circumstances; original TRT K only if independently measured on same workload/boundary |
| O | Complete replacement operator call | all required Q/K/V packing, allocation, dispatch, copy, synchronization; baseline and candidate only if boundaries match |
| I | Engine / application replay | matched original vs rebuilt candidate Engines and replay, including new reformat/fusion/tactic changes |

**A per-layer TRT Engine Inspector JSON has no GPU execution latency.** An
original `trtexec --dumpProfile`/Nsys report may include surrounding work and
must be mapped to actual GPU launches before interpreting as a like-for-like
K baseline. Do not invent baseline latency if original kernel binary cannot
be replayed standalone.

## Recording minimums

- Machine: GPU, clock/power/thermal state if accessible, CUDA/TRT/driver build,
  Python/torch, plugin/commit and Engine hashes, optimization profile.
- Workload: B/H/Sq/Sk/D, dtype, actual byte strides/layout, causal/mask/scale,
  history length and valid slots, real workload mode and commit state; distinguish
  contig fixture from original storage.
- Protocol: warmup, repeated measurements, p50/p90 or distribution, units,
  synchronization, chosen CUDA stream, CUDA Graph on/off, concurrent GPU load;
  separate one-time compile/setup from steady-state unless relevant to application.
- Numerical: frozen acceptance rule and correct input/output, no NaN/Inf;
  benchmark correctness checked against matching fixture, not only random data.
- Same input/history/chunk boundary for **baseline vs candidate** full Engine;
  regression guardrails for output, KV updates, and repeated-step/commit state,
  not just one forward.

## Multi-arm paired replay

- Name each arm and list its exact changes: Engine artifact, runtime path, KV
  preparation/binding, graph mode, and any replaced operator. If an arm changes
  FFN execution, KV binding, and CUDA Graph execution together, call it a
  **composite candidate**. Do not attribute its result to FFN alone without an
  FFN-only control.
- Pair arms on identical inputs, initial cache state, prediction/commit/reset
  sequence, and workload configuration. Record whether later chunks consume
  predicted actions or demonstration/reference actions; these are different
  workloads and must not be mixed in one comparison.
- Randomize and balance execution order across paired blocks (for three arms,
  rotate the order). Preserve every run and the order. Repeated chunks within a
  single replay are not independent run-level samples.
- Define exclusion rules before running. A confirmed external I/O or system
  fault overlapping a measurement may invalidate that run or paired block;
  preserve the raw result and reason. Never exclude a run just because it is
  slow or unfavorable. Report valid pair count and limitations.

## Split setup, replay, and timing clocks

When startup or first-use cost is in scope, use the endpoint definitions and
phase attribution protocol in [`09-engine-lifecycle.md`](09-engine-lifecycle.md).
The table below lists report boundaries only; it does not redefine K/O/I.

Within integration (I), report distinct boundaries instead of one ambiguous
"inference time":

| Measurement | Boundary to record |
|---|---|
| Engine initialization | host wall interval for load/deserialize/context setup; split file read from deserialize only when actually instrumented |
| CUDA Graph setup | capture/instantiation or graph-build interval, measured separately from Engine initialization |
| Warmup | count and time, with whether it is included in first-use cost |
| KV preparation | pack/copy/page-table preparation at the call boundary, including required allocation, dispatch, and synchronization |
| Engine Forward | declared Engine invocation(s); include mode and invocation count |
| Chunk E2E | application Generate, Commit, and their total for the declared chunk range |

For each value label the clock as host wall time, CUDA Event elapsed time, or
GPU kernel-busy time. These are different measures. CUDA Event intervals can
include stream waits and launch gaps; nested event ranges must not be added, and
event elapsed time is not GPU kernel-busy time. Use Nsight Systems or another
explicit kernel-activity measurement for kernel-busy claims.

Report both first-use cost and replay cost after removing only explicitly
measured one-time setup. State the actual chunk horizon. When a candidate adds
setup but has a positive paired steady-state saving, an illustrative break-even
is `added setup cost / paired saving per chunk`. Mark it **not established** if
the saving is within noise, changes sign across pairs, or has too few valid
pairs to support the estimate. Do not extrapolate a break-even point as a
general performance law.

## Numerical identity and task acceptance

Keep these claims separate: bitwise identity on tested inputs, declared
tolerance-based numerical agreement, cache lifecycle/metadata correctness,
and task-quality acceptance. A passing `allclose` test or agreement with
another implementation does not establish task quality. Report the reference,
threshold source, measured outputs/cache state, and any unavailable quality
gate. For multi-chunk claims, validate the relevant per-chunk outputs and
effective cache contents, not just one forward.

## Sequencing

1. Reproduce any *reported* small-fixture PASS in the source project environment.
2. Capture **production-shape** fixtures, plus representative short/mid/full KV
   and candidate supported modes/heads; preserve physical strides or report
   where storage must be copied.
3. Benchmark candidate K and O under matched real workloads. Identify whether
   each baseline measurement is exact, estimated or unavailable.
4. Only then choose the Plugin boundary and rebuild (see `04-integration.md`).
5. Compare new Engine runtime, GPU timeline and full application replay.

If O cannot be fairly compared before integration, call it **unavailable** and
use integration as a measured experimental hypothesis. Do not label it a win.

For a storage-only or lazy-initialization hypothesis, compare byte-identical
Engines, record actual Page Cache evidence, and report first-use and warmed
replay separately. A new process alone does not establish cold cache; see
`09-engine-lifecycle.md`.

For a three-arm replay, use one harness and a balanced randomized order. Compare
Generic, the cache-only candidate, and the combined candidate as separate arms;
the combined arm is not an FFN-only comparison if it also changes binding or
execution mode. A concise anonymized example and its evidence limits are in
[`examples/ffn-kv-pack-case-study.md`](../examples/ffn-kv-pack-case-study.md).

## Report fields

Use `templates/benchmark_report.json` as a human/machine-readable **template**.
A missing number stays `null`; never fill it with an invented estimate.
