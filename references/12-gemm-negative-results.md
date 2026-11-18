# 12. GEMM investigation and when to stop

## What a large GEMM hotspot does—and does not—prove

A large FFN down projection (e.g. BF16 matrix product with a wide reduction dimension) can dominate module cost. This identifies a **potential payoff ceiling**, not evidence of inefficient implementation.

Start with authentic M/N/K, transposition, physical strides, weights, activation distribution, epilogue, precision/accumulation and the current Engine's tactic. A standalone random-matrix GEMM can test implementation mechanics but is **not** a production comparison.

## Comparison gate

1. Record actual Engine kernel mapping and its measured latency at a pinned workload.
2. Capture true activation+weights if doing so does not change the target; when internal capture requires graph surgery/debug outputs, explicitly label the changed Engine and any unobservable baseline.
3. Compare library and candidate GEMMs under matched input, stream, warmup, timing boundary and correct epilogue. PyTorch and cuBLASLt may share underlying library kernels; do not present them as independent algorithmic innovations.
4. Separate Nsys single-launch durations, IProfiler layer callbacks, CUDA Event ranges and caller wall time. **Never divide an Event candidate by an IProfiler TRT median and assert a speedup.**
5. Only integrate if an unambiguous advantage survives full operator cost and numerical checks; otherwise record a clean **negative result**.

## NCU counter interpretation

- High dynamic shared memory and low theoretical occupancy can be an intentional tile-reuse tradeoff, not root cause.
- Register usage, shared memory allocation, achieved/theoretical occupancy and generic SM/memory throughput are correlates; causality needs a controlled tactic/tile experiment.
- Vendor-specific tensor-pipe metrics may be unavailable, misleading or constrained by architecture/clock/cluster behavior. Check the **exact metric definition**, device support, replay mode and disassembly before concluding tensor hardware is idle.
- Unsupported CUDA Graph capture (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`) is not proof a GEMM is slow; use a legal synchronized/event or profiler fallback, and disclose unmatched timing scopes.
- On shared accelerators, do not kill others' processes or exceed a declared free-memory safety threshold to collect NCU.

## Stop rule

If no candidate has convincing same-boundary performance and correct numerics, **do not rebuild an Engine just to force a winner**. Record what was tested, which comparisons remain noncomparable, what NCU suggests but does not prove, and why the branch is paused. Re-evaluate only with new evidence (new tactic, profiling support, changed shape or a demonstrably lower integration overhead).
