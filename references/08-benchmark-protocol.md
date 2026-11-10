# 08. Benchmark protocol: no local PASS implies speedup

## Workload identities (do not mix)

- A **reduced correctness fixture** is valid for algebraic/caller behavior checks
  only; it is NOT a baseline for actual TensorRT kernel performance.
- A **real production-shape workload** must preserve Batch×Heads, query length,
  KV length, dtype, strides, Q/K/V packing and masks, execution state and stream.
- Video and Action can have different Sq and KV state even if named TensorRT
  `_gemm_mha_v2_myl0_17`. Pin mode and Engine hash, do not join by index alone.

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
  history length and valid slots, real video/action/commit mode; distinguish
  contig fixture from original storage.
- Protocol: warmup, repeated measurements, p50/p90 or distribution, units,
  synchronization, chosen CUDA stream, CUDA Graph on/off, concurrent GPU load;
  separate one-time compile/setup from steady-state unless relevant to application.
- Numerical: frozen acceptance rule and correct input/output, no NaN/Inf;
  benchmark correctness checked against matching fixture, not only random data.
- Same scene/history/chunk boundary for **baseline vs candidate** full Engine;
  regression guardrails for output, KV updates, repeated denoise/commit, not just one forward.

## Sequencing

1. Reproduce the *reported* small-fixture PASS on company hardware.
2. Capture **production-shape** fixtures, plus representative short/mid/full KV
   and candidate supported modes/heads; preserve physical strides or report
   where storage must be copied.
3. Benchmark candidate K and O under matched real workloads. Identify whether
   each baseline measurement is exact, estimated or unavailable.
4. Only then choose the Plugin boundary and rebuild (see `04-integration.md`).
5. Compare new Engine runtime, GPU timeline and full application replay.

If O cannot be fairly compared before integration, call it **unavailable** and
use integration as a measured experimental hypothesis. Do not label it a win.

## Report fields

Use `templates/benchmark_report.json` as a human/machine-readable **template**.
A missing number stays `null`; never fill it with an invented estimate.
