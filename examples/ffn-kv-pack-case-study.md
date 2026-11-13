# Brief case: combined FFN and KV preparation change

This anonymized case summarizes a reported multi-arm replay. It contains no
project identifiers, model details, paths, raw artifacts, or exact measurements.

## Evidence reported by the source session

- The cache-only arm preserved the tested outputs and reduced a measured KV
  preparation interval. The application-level replay gain was much smaller.
- A combined arm also changed FFN execution, KV binding, and graph behavior.
  Its result is a composite-candidate observation; it does not isolate FFN.
- Only a small number of paired blocks completed. One later block overlapped a
  confirmed external I/O fault and was retained as an excluded failure.
- An event interval changed substantially, but it included stream waits and was
  not a GPU kernel-busy measurement. No accepted Nsight attribution was
  available.

## Limits and next experiment

The report supports a bounded cache-path observation for its tested replay.
It does not establish a general speedup, an FFN-only effect, task-quality
acceptance, or a complete cache-correctness result for the composite arm. Keep
the raw report and inputs in their source project. Follow up with isolated
cache-only and FFN-only controls, more valid paired blocks, and matched output
and state checks.
