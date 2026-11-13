# 09. TensorRT Engine lifecycle and cold-start measurement

## Scope

An Engine file on disk is only one phase of serving. This reference separates
file I/O, TensorRT setup, first-use work, GPU execution, and application latency
for an already-built serialized Engine. Include Engine building/tactic search
only when the deployed application actually builds at startup.

```text
process start → application setup → read Engine bytes → create Runtime
  → deserialize Engine → create context → select profile / set shapes / bind I/O
  → allocate buffers → optional warmup → optional graph capture/instantiate
  → Ready → first request → steady requests
```

Instrument the phases present in the actual call path. A wrapper named
`load_model()` is not evidence that all its time is deserialization.

## Declare the user-visible endpoint

| Endpoint | Boundary | Include |
|---|---|---|
| Startup-to-Ready | process launch to declared Ready signal | Every phase required before the service accepts the target workload |
| Ready-to-first-result | first accepted request to its complete result/commit | Any lazy Engine, context, buffer, warmup, or graph work after Ready |
| Launch-to-first-result | process launch to first complete result | Startup plus the first request |
| Steady state | repeated requests after declared setup | Per-request distribution and sample count |

Moving lazy initialization before Ready redistributes time between endpoints;
it does not remove the cost. Name the product SLO and report the other endpoints
when a change shifts work between them. See `08-benchmark-protocol.md` for K/O/I
and paired replay measurement boundaries.

## Time phases separately

Use a monotonic host clock for synchronous host phases. Record Engine hash,
workload/profile, byte count, process, and exact start/end events for each phase.

1. **File read:** time the bytes actually passed to TensorRT; report size and
   effective throughput. A cached read is not physical-device throughput.
2. **Runtime and deserialization:** time Runtime creation and deserialization
   separately where material. Deserialization time is not file-read time.
3. **Context and bindings:** measure context creation, profile selection, shape
   setup, address binding, allocation, and required format conversion.
4. **First use:** record lazy plugin/module loading, workspace/allocator setup,
   first enqueue, and warmup inputs/count. Do not call runtime work tactic search
   unless the application actually builds an Engine there.
5. **CUDA Graph:** separate warmup, capture, instantiate, and replay. State
   whether setup is before Ready or paid by the first request.
6. **Inference:** keep application host wall, CUDA Event elapsed, and GPU
   kernel-busy time as separate clocks. CUDA Events measure elapsed stream time
   and can include gaps; use an activity trace such as Nsight Systems for
   kernel-busy attribution.

If a host-blocking call occurs between Event markers, the interval can include
device idle time while the host is blocked. Correlate event ranges with the CPU
timeline and GPU activity; do not label such a span GPU compute time.

## Cold and hot file-cache comparison

A new process does not imply a cold file cache. Linux Page Cache is shared
across processes; prior hashing, copying, scanning, or a previous run can warm
it. A hot-cache read also includes allocation and memory-copy work.

- Verify source and destination devices/filesystems with system tools such as
  `findmnt`, `lsblk`, and `df`; do not infer media from a directory name.
- Copy the same Engine to an isolated destination, verify source/target SHA-256,
  and never overwrite the production artifact.
- For a cold Page Cache trial, use targeted per-file eviction only where safe
  and supported; check residency immediately before timing (for example with
  `mincore`). Eviction is advisory. If residency is unknown, label it unknown.
- For hot-cache trials, read the full file before timing and verify residency.
- Alternate or randomize storage order across independent processes. Record
  every round, intended cache state, concurrent I/O, failures, and retries.
  Never combine cold and hot samples into one average.
- Do not use global `drop_caches` on a shared host for a benchmark. Page-cache
  state does not prove controller/device cache state.

## Attribution and validity

With byte-identical Engine hashes, unchanged runtime/build, inputs, context,
profile, and execution path, an improvement limited to file read or setup is a
startup/first-use improvement. It is not a faster Attention, FFN, or GPU kernel.
Compare warmed replay separately. If Engine hashes differ, classify the result
as a different Engine comparison because tactics, graph, precision, or build
settings may differ.

A failed/short read, hash mismatch, deserialization failure, or context failure
is an unsuccessful initialization attempt, not a successful latency sample.
Retain it with phase, error, bytes read, retry/fallback behavior, and production
handling. A successful timing claim also needs the intended cache state (or an
explicit unknown), observed system activity, and applicable output/state checks.

## Diagnostic sequence

1. Choose the endpoint and identify which setup phases are eager or lazy.
2. Trace the actual application call path and add phase timestamps at those
   boundaries.
3. Verify file identity, size, device, cache state, free space, and competing
   I/O before copying or benchmarking.
4. Run cold and hot initialization separately; preserve raw rounds and errors.
5. Replay identical inputs and compare output and relevant committed state.
6. Use a system timeline to distinguish host blocking, stream gaps, GPU work,
   and overlapping I/O. Profile individual kernels only after identifying a
   measured GPU hotspot.
7. Repeat first-request and warmed steady-state comparisons independently.

## Common misreadings

| Misreading | Interpretation |
|---|---|
| “New process means cold.” | Page Cache survives process exit; verify or mark unknown. |
| “Deserialize time is Engine load time.” | Read, deserialize, context, bindings, and warmup are separate phases. |
| “CUDA Event reports GPU busy.” | It reports stream elapsed time; confirm busy intervals in a system trace. |
| “Storage improved inference throughput.” | A read/init-only change establishes startup or first-use impact; measure steady state separately. |
| “Graph replay is request latency.” | Replay excludes capture/instantiate and may exclude warmup. |
| “An I/O error is a slow sample.” | It is a failed initialization unless production recovers; retain it separately. |

## Brief case

See [`examples/engine-lifecycle-case-study.md`](../examples/engine-lifecycle-case-study.md)
for a short anonymized observation and its evidence limits.

## Minimum report fields

- Endpoint definition, call path, and eager/lazy lifecycle phases.
- Engine hash/size, runtime versions, workload/profile, source/target device and
  filesystem.
- Cache method and residency evidence, or `unknown`; concurrent system activity.
- Per-round file read, deserialize, context, binding/allocation, warmup, graph
  setup, inference, errors, and retries.
- Startup, first-result, and steady-state distributions with clock domains and
  inclusion rules.
- Output/state checks and whether compared Engine files were byte-identical.
