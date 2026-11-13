# Brief case: paged-KV Engine integration

An anonymized integration report compared dense-history and page-table-backed
attention. Its purpose here is to show how to label evidence and failure limits;
the source artifacts and exact workload remain in the originating project.

## Reported observations

- Direct Paged and Contiguous backend calls matched on the tested inputs, and a
  one-Plugin case ran. A later multi-Plugin candidate produced nonfinite output.
- An all-Contiguous multi-Plugin control also failed, so the observed failure
  did not isolate paged addressing as the cause.
- A separate staging-copy defect crossed an ownership boundary. A per-owner
  copy corrected that specific defect; this did not establish general memory
  safety.
- Small-pool state checks covered staging overwrite, commit/clear, eviction,
  reuse, isolation, and reset. A later recorded replay completed, but the
  reported numerical and task-quality gates were not all accepted.
- A latency spike seen in an earlier run did not recur in a later run. Its cause
  remained unresolved; non-reproduction did not establish a fix.

## What the evidence does not establish

A backend comparison does not establish Plugin or full-Engine equivalence. A
legal page table does not prove logical token order or matching KV values. A
completed replay does not imply task-quality acceptance. Keep metadata, page
legality, logical order, KV contents, model outputs, performance, and quality
as separate gates. Do not include private tensors, Engine files, paths, or raw
measurements in a portable example.
