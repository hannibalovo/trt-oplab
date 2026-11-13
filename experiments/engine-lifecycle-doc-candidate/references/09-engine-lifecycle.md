# 09. TensorRT Runtime Lifecycle and Cold-Start Measurement

## Purpose

A TensorRT Engine can be present on disk while the application still has substantial work before its first useful result. This reference separates artifact I/O, TensorRT runtime setup, first-use work, GPU execution and application latency so a storage or lifecycle change is not mistaken for a kernel optimization.

This page covers an already-built serialized Engine. Engine building and tactic search are separate workloads and must not be mixed into runtime initialization unless the product builds Engines on startup.

## Lifecycle and timing boundaries

```text
process start
  → application/model setup
  → read serialized Engine bytes
  → create TensorRT Runtime
  → deserialize Engine
  → create ExecutionContext(s)
  → select profile / set shapes / bind addresses / allocate buffers
  → optional warmup and lazy runtime initialization
  → optional CUDA Graph capture + instantiate
  → Runtime Ready
  → first request / first Engine use
  → subsequent requests (steady state)
```

Instrument the boundaries that actually exist in the application. A wrapper called `load_model()` may include several phases above; do not label it “deserialize” without evidence.

Report these application-level quantities separately:

| Metric | Start → end | What it answers |
|---|---|---|
| **Startup-to-Ready** | process launch → the service's declared Ready signal | How long before the service can accept the defined workload? Include every initialization step required by that Ready contract. |
| **Ready-to-End (first result)** | first request accepted after Ready → complete first result/commit | What does the first real user request pay? Lazy Engine, context, buffer, warmup, or graph setup after Ready belongs here. Record idle/scheduling separately if included. |
| **Steady state** | repeated completed requests after initialization/warmup → per-request distribution | What does a reused Engine/context/graph cost under the declared production workload? Report p50/p90/p95 and sample count. |

Also report process-launch-to-first-result when that is the user-visible objective. Moving lazy initialization before Ready can lower Ready-to-End latency while increasing Startup-to-Ready; it redistributes work rather than removing it. State which quantity is the product SLO.

### Why lazy Engine loading lands in the first Generate

Many adapters cache one Runner per mode and create it on demand:

```python
if mode not in runners:
    runners[mode] = TensorRTEngineRunner(engine_path)
return runners[mode](inputs)
```

If the first Video/Action transformer call happens inside `infer_generate`, the first call synchronously opens and reads the Engine, deserializes it, creates its context and then executes. A timer around the outer Generate therefore includes those one-time operations. Later Generate calls reuse the Runner and do not pay them again. Verify this from the call path and per-phase timestamps; a long first request alone does not prove Engine I/O was the cause.

## Separate the setup stages

Use a monotonic host clock (`perf_counter_ns`) for synchronous host phases. Emit an event for each phase with Engine hash, path, bytes, workload/profile and process ID.

1. **File open and read** — time the actual read that produces the byte buffer passed to TensorRT. Record byte count and effective MiB/s. A `read()` may be served from Linux Page Cache; it is not automatically physical-device throughput.
2. **TensorRT Runtime creation** — time separately if material. Keep its lifetime valid for deserialization according to the TensorRT API used.
3. **Engine deserialization** — time `deserialize_cuda_engine` on the bytes. This is host wall time for that API call, not the time to read the plan and not GPU kernel time.
4. **ExecutionContext creation** — time `create_execution_context` separately. Record how many contexts are made and the peak memory behavior. Context creation can allocate resources and may fail independently of deserialization.
5. **Profile, shapes, I/O and buffers** — record optimization-profile selection, dynamic shapes, tensor-address binding, buffer allocation and any format conversion. Include only work required by the production path and label it.
6. **First-use setup / warmup** — record any first enqueue, plugin/module loading, allocator or workspace initialization and exact warmup inputs/shape/count. TensorRT tactics are normally selected at build time; do not describe runtime setup as tactic search unless that is what the code does.
7. **CUDA Graph setup** — if used, separate warmup, capture, graph instantiate and replay. See below.
8. **Actual inference** — report application wall, CUDA Event elapsed, and GPU kernel-busy time as different metrics. Use Nsight Systems to identify host gaps and GPU work; use Nsight Compute for a selected kernel, not whole-request attribution.

For a required synchronized phase, document where synchronization occurs. A CUDA Event measures device-stream time between markers, not arbitrary CPU work. However, an Event pair surrounding a call that blocks the host can include GPU idle gaps before later work is enqueued. A first `execute_async_v3` span that includes Engine loading must not be called “GPU compute time.” Correlate Events with the CPU timeline and Nsight Systems kernel intervals.

## Cold and hot file-cache protocol

A new Python process is **not** proof of a cold file cache. Linux Page Cache is shared across processes, and a prior hash, copy, scanner, or earlier benchmark can warm it. Conversely, a hot-file-cache read is not a storage-device measurement.

For a storage comparison:

1. Verify the actual source and target devices/filesystems with `findmnt`, `lsblk` and `df`; do not infer SSD/HDD from a directory name. Record model, transport, `ROTA`, mount, filesystem and available space.
2. Copy the exact Engine files to an isolated destination; never overwrite the production artifact. Compare SHA-256 before timing. Keep source and target hashes in the report.
3. For **cold Page Cache**, use targeted per-file eviction only when safe and supported (for example `POSIX_FADV_DONTNEED`). Treat it as advisory. Check residency immediately before the timed read (for example with `mincore`) and label the result *cold Page Cache* only when the check supports that claim. Never run global `drop_caches` on a shared host for a benchmark.
4. For **hot Page Cache**, deliberately read the entire target file and verify residency before timing. Record the pre-read time separately. A warm cache can still incur Python allocation and memory-copy costs.
5. Measure repeated independent processes in randomized or alternating storage order. Each round must prepare and verify the intended cache state. Report per-round raw data and a summary; do not silently discard slow or failed rounds.
6. Record competing storage I/O and GPU activity. A cold read can be affected by USB/RAID/network storage, filesystem layers, throttling, shared-device work and hardware caches. “Cold Page Cache” does not guarantee an empty drive/controller cache or a physically cold device.

If page-cache state cannot be inspected or controlled, report it as **unknown**. Do not call the timing cold. Do not combine cold and hot samples in one average.

## CUDA Graph lifecycle

Graph setup belongs to the boundary where the application actually performs it:

- If graph warmup, capture and instantiate are required before the service declares Ready, count and report them in Startup-to-Ready.
- If capture happens lazily on the first request, count it in Ready-to-End.
- If measuring an already prepared graph replay, report replay latency separately as Steady state. Do not include capture in one arm and omit it from the other.
- Capture and replay must use compatible Engine context, stream, tensor addresses, shapes and memory lifetimes. For dynamic shapes or multiple profiles, state whether there is one graph per shape/profile and whether first use of a new key triggers another capture.
- Report warmup/capture/instantiate time and graph replay time as separate phases. A CUDA Graph is not a way to erase startup cost from the user-visible metric.

If production does not use CUDA Graphs, do not introduce graph setup into the benchmark merely to obtain a faster steady-state number.

## Storage wins versus GPU-kernel wins

Storage placement changes how quickly Engine bytes reach host memory. It does not, by itself, change the serialized Engine's tactics or the GPU kernels it launches.

- With identical Engine SHA-256, GPU, runtime versions, inputs, context/profile and execution path, an HDD-to-NVMe improvement confined to cold file read / initialization is a **startup or first-request improvement**.
- Do not call lower first-Generate wall time a faster Attention, FFN or GPU kernel.
- Compare warmed forward/steady-state distributions separately. If those shift, investigate clocks, thermal state, concurrent GPU work, allocator state, streams, graph mode and measurement boundaries before attributing the change to storage.
- If Engine hashes differ, this is no longer a storage-only comparison: tactics, graph structure, precision and build settings may also differ.
- On systems where file cache competes for host or unified memory, clearing Engine file cache after deserialization may matter to memory availability. Measure it as an explicit lifecycle action; do not assume it improves GPU execution.

## I/O errors and sample validity

A failed read, short read, hash mismatch, deserialize failure or context-creation failure is not a valid successful initialization-latency sample. Preserve the failed attempt and report its phase, error code/message, bytes read, retry behavior and whether production would retry or fail startup. Do not remove the row and quietly rerun it.

For a sample to support a performance claim, require at least:

- all expected bytes read and successful Engine deserialization/context creation;
- source/target artifact hash match for a storage-only comparison;
- intended Page Cache state verified, or explicitly marked unknown;
- no unrelated workload invalidating the comparison, with observed GPU and storage activity recorded;
- successful output and relevant state/correctness checks for any E2E run.

If production includes retries, timeouts or fallback behavior, benchmark that behavior as its own end-to-end scenario. A failed attempt may be operationally important but is not a latency sample for successful initialization.

## Suggested diagnostic sequence

1. Define the product boundary: Startup-to-Ready, Ready-to-End, process-launch-to-first-result, or Steady state.
2. Inspect the actual lifecycle call path. Check whether Engine loading/context/graph setup is eager or lazy and which API call occurs inside the first request.
3. Check device, path, file size/hash, free space, concurrent processes and I/O before a large copy or benchmark.
4. Instrument read, deserialize, context, I/O setup, warmup, graph setup and enqueue independently with host timing. Add CUDA Events only for clearly scoped stream work.
5. Run cold-cache and hot-cache samples separately with evidence of residency. A fresh process alone is insufficient.
6. Replay the same fixed production input on byte-identical Engine artifacts from each storage location. Compare output, committed state and Engine hashes.
7. Use Nsight Systems to separate CPU blocking, stream idle gaps, actual GPU kernels and overlapping I/O. Use Nsight Compute only after a specific GPU kernel remains a measured hotspot.
8. Repeat the Ready-to-End test and the warmed steady-state test independently. Publish raw rounds, distributions and exact endpoint definitions.

## Common misreadings

| Misreading | Correct interpretation |
|---|---|
| “New process means cold.” | Page Cache survives process exit; verify or mark unknown. |
| “The first Generate took 100 s, so the Transformer is slow.” | The first call may lazily read/deserialise Engines, create contexts or capture a graph. Attribute phases before profiling kernels. |
| “Deserialize time is Engine load time.” | File read, deserialization, context creation, buffer setup and warmup are distinct phases. |
| “CUDA Event reports GPU busy.” | It reports stream elapsed between markers and can span idle gaps caused by host-side blocking. Confirm kernel-busy intervals in Nsight Systems. |
| “NVMe improved inference throughput.” | If only read/init changed and Engine hashes match, the demonstrated win is cold startup/first-use latency. Measure warmed requests before claiming throughput. |
| “Graph replay is the request latency.” | It excludes capture/instantiate and often warmup; include those where production performs them. |
| “An I/O error is just a slow sample.” | It is a failed initialization unless production successfully recovers; retain it separately and report retry behavior. |
| “An average across cold and hot runs is representative.” | It hides two distinct operating conditions; report them separately. |

## Brief case: LingBot-VA Thor Engine cold start

Two identical approximately 9.35 GiB Video/Action Engine files were compared on a USB HDD and an NVMe SSD. With Page Cache residency verified at 0% before each cold read, two-round medians were 122.12 s total Engine read and 127.63 s total Runner initialization on HDD, versus 12.32 s and 18.07 s on NVMe. TensorRT deserialization remained about 0.49 s total on both. One fixed recorded Chunk 0 replay measured first `infer_generate` at 128.36 s on HDD and 28.08 s on NVMe; generated Action and cache/KV hashes matched. This E2E pair was a single comparison, so it supports a strong startup diagnosis but does not establish E2E variance or a steady-state speedup. The storage change moved the delay; it did not make the GPU kernels faster.

## Minimum report fields

- Endpoint definition and lifecycle phase diagram / call path.
- GPU, OS, driver, CUDA, TensorRT, Engine hash/size, source and target device/filesystem.
- Cold/hot Page Cache method and residency evidence; concurrent GPU/storage work.
- Per-round read bytes/time/throughput, deserialize, context, I/O/buffer setup, warmup and graph setup times.
- Startup-to-Ready, Ready-to-End, and steady-state distributions with explicit inclusion/exclusion rules.
- CUDA Event scope, synchronization/stream, Nsight Systems kernel-busy evidence where GPU attribution is claimed.
- I/O errors/retries and failed-run handling.
- E2E output/cache correctness evidence and a statement of whether Engine files were byte-identical.
- Raw per-round JSON/CSV, command/configuration and interpretation limits.
