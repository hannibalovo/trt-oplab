# 13. Stateful inference benchmarks: a common contract, not one universal runner

## Three distinct questions

1. **Cold start:** process/model/Engine load, compilation, graph capture, cache construction, warmup and first response. Disclose all relevant preparation.
2. **Ready-to-replay task:** pin the readiness contract; measure the actual complete multi-step replay including **required** first-use work. A lazy cache miss/graph build inside the run is not steady-state.
3. **Steady state:** all declared one-time setup already done; measure a specified window, still including necessary per-forward copy/pack/staging/dispatch/sync.

Don't silently exclude a cost because it makes the optimization look worse. Record model load and diagnostic prewarm separately even if the headline scope excludes them.

## Matched workload contract

Pin and report: model/Engine hashes, mode(s), shape/precision, seed, denoising steps, guidance/CFG layout, history length and page state, prediction/commit schedule, cache reset and state ownership, runtime flags, GPU/driver/TRT, concurrency, sampling method, windows and inclusion rules.

Use **AB/BA reversed paired trials** and reproducible inputs. Treat startup precompute or graph creation explicitly: whether it is timed, whether it is amortized at task level, how many actual misses/hits occurred, and whether cached values survive across chunks/tasks.

## Reporting pitfalls

- `generate + commit` is a **different scope** from only generate or one Transformer Forward.
- A named full-cache window must match exact chunk indices; "last N chunks" is not interchangeable with "last N after full capacity."
- Host enqueue time and GPU CUDA Event time are different clocks; report both, **do not add them blindly**.
- Sum of Nsys launch durations, IProfiler callbacks, CUDA Event ranges and CPU wall are not interchangeable.
- Run heavy correctness dump/hash tests **separately** from performance; unbounded I/O can overwhelm the measured run.
- GPU idle/thermal/load drift and runner startup order can dominate marginal changes; show both paired reductions, not only an average.
- A cache-hit-only speedup is not a full task speedup. A kernel win does not imply call/Engine/E2E win.
- Different specialized runners are acceptable if they obey one explicit benchmark contract. Forcing one runner can hide candidate-specific lifecycle costs.

## Required evidence

Keep immutable per-run logs, per-chunk metrics, setup/precompute/miss-hit counters, engine/config checksums, correctness and state audit, and AB/BA summary. Separate measured, inferred and unknown quantities.

See `08-benchmark-protocol.md` for kernel/operator/Engine scopes and `templates/system_optimization_review.md` for a concise review template.

## Engine initialization detail

This page defines task-level Cold/Ready/Steady accounting and complete task
cost. For the internal serialized-Engine phases—file read, Runtime creation,
deserialization, context/profile/shape/binding setup, allocation, warmup and
CUDA Graph capture/instantiate—see
[`16-engine-startup-lifecycle.md`](16-engine-startup-lifecycle.md). Use the
same endpoint definitions here; Reference 16 decomposes the initialization
inside them rather than defining a competing task lifecycle.
