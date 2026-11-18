# 10. From ONNX semantics to module-level GPU cost

## Required evidence chain

```text
model source and actual dynamic dependencies
  → ONNX nodes, inputs, outputs and consumers
  → Inspector TRT Layer + engine SHA + tactic
  → actual Nsys CUDA launches and durations
  → targeted NCU resources and counters
  → Self-Attention / Cross-Attention / FFN / Other / Mixed / Unknown
```

Never infer a runtime kernel from a TensorRT Layer name alone. One Inspector layer may fuse many ONNX nodes; launches may not be one-to-one. Keep one trace, one target forward, one Engine hash and explicit shape/cache state.

## Cost-accounting rules

- Classify **all active CUDA work** at the chosen boundary. Assign a kernel duration once, not once per ONNX node it contains.
- Prefer semantic origin and consumer topology: Self-Attention QKV / SDPA / output, Cross-Attention Q / text K/V / SDPA / output, FFN gate/up/activation/down, norms/conditioning, layout/output.
- Leave fused cross-category kernels under **Mixed** and unmapped launches under **Unknown**; report their shares. Do not distribute a mixed kernel by ONNX node count.
- Record mapping confidence (exact, group-level, inferred, unknown); compare engines by unique ONNX semantics, not Layer index.
- Keep IProfiler callback medians, Nsys individual launch durations, CUDA Events and CPU wall in **separate columns**. The sum of per-layer medians is not a median full Forward. A sum of GPU launch durations is not application wall time.
- Measure first/partial/full history separately. An attention-heavy steady state may not represent short-context operation.

## Logical shape is not physical traffic

A fused conditioning kernel may expose many large logical `Transpose/Split` outputs. `shape × dtype` describes **logical** volume, not DRAM writes or allocated memory. Strides, pointer aliasing and materialization are unproven without a supported layout/allocator diagnostic. A view-legal ONNX expression is not proof TensorRT kept a view.

Similarly NCU's generic Memory Throughput or L2 Throughput **percentage is not DRAM bytes**. Collect supported byte counters where possible; if unsupported, write "unknown." Profiling/debug tensors can change fusion, so compare debug builds with controls.

## Minimal deliverables

- `layer_semantics.csv`: mode, engine_sha, ONNX source group, TRT layer/tactic, actual kernel, classification, confidence, and measured timing provenance.
- `module_cost.csv`: category/subcategory, duration, denominator, unknown/mixed share, measurement method.
- a short "top opportunities" table that distinguishes **cost observed** from **cost removable**.

Choose the next optimization by **avoidable cost**, not largest raw kernel alone. A large GEMM with no faster replacement may rank below a smaller subgraph provably invariant across forwards.

See `11-invariant-subgraph-cache.md` and `12-gemm-negative-results.md`.
