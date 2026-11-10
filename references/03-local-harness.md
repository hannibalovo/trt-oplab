# 03. Local harness: correctness and 3-level measurement

## Goal

Develop an operator without rebuilding the 30-layer engine after each edit.
A local harness is a stand-alone executable/script loading the captured
Q/K/V (or FFN input + fixed weights), invoking a reference and candidate,
checking numerics, warming up, and measuring a defined cost boundary.

It does **not** have to be a compiled `.cu`: Triton JIT, CUDA/C++, cuDNN,
or a standalone TensorRT engine are all possible candidates. A Plugin is
needed only when choosing to integrate a custom op into the TRT graph.

## The three measurements (report all separately)

- **K** kernel-only: one device kernel's elapsed time for fixed device-resident
  inputs, with warmup, repeats, and clear stream/synchronization boundaries.
- **O** full operator call: *necessary* format conversion, prepare page tables,
  KV pack/reformat/copy, host dispatch, and synchronization on real inputs.
- **I** integration: changed Engine and full application replay on matched
  workloads. I is **not** measured within the local harness.

Capture potential setup costs explicitly. Do not include one-time compilation
in K/O steady-state unless the user-facing workload requires it. Keep scratch
buffers, stream ownership, CUDA graph mode, concurrent load, and clocks stable.

## Correctness

- Compare each output using agreed max-abs/max-rel/RMSE or another appropriate
  metric. For low-precision outputs, thresholds must be *declared*, not changed
  after seeing errors.
- Check NaN/Inf explicitly, plus output shape/dtype, mask, positional semantics.
- Include multiple real states: KV empty/partial/full and noncontiguous history;
  video/action modes when the candidate claims to support both.
- Maintain untouched reference fixture and candidate code revision.

## Hypothesis bookkeeping

Form two independent hypotheses:

H1: candidate **core** kernel has a speed advantage at the target shape.
H2: necessary conversion + execution retains an advantage on the real path.

If H1 passes but H2 fails, do **not** reflexively integrate. Investigate layout
changes, a wider fused region, or reject the candidate. An explicit experimental
reason may still justify an integration test, but record the override.

## Example measurement record

```json
{
  "workload": "action-B2-Sq32-Sk8160-bf16",
  "numeric": {"max_abs_error": null, "threshold": null, "nan_inf_count": null},
  "kernel_only_ms": {"baseline": null, "candidate": null, "repetitions": null},
  "full_operator_ms": {"baseline": null, "candidate": null, "repetitions": null},
  "context": {"gpu": null, "trt_version": null, "cuda_version": null,
              "cache_layout": null, "stream": null, "cuda_graph": null}
}
```

Null means *not measured*. Never fabricate latency, shape, or GPU capability.

## v0.3 numerical gate and benchmarking evidence

See `07-validation-harness.md` for per-element mixed atol/rtol acceptance and
negative tests. The experiment helper's `numeric_validation_pass=true` is an
**attestation linked to a saved project-validator report**, not a computed
check; `max_abs_error <= atol` is NOT a valid replacement. For the local stage,
include `candidate_kernel_ms` and `candidate_operator_ms`, plus
`benchmark_comparison_status=matched|unavailable`. If `matched`, also supply
`baseline_operator_ms` and `workload_boundary_matched=true`. If `unavailable`,
local PASS does **not** assert a speedup; it allows a clearly labeled measured
integration hypothesis. Details in `08-benchmark-protocol.md`.
