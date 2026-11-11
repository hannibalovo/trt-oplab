# Candidate navigation and checklist additions

This proposal is intentionally separate from `SKILL.md`, `references/01-scope-baseline.md` and `references/08-benchmark-protocol.md`. It does not modify the formal Skill.

## `SKILL.md` — minimum changes

1. Add to the Reference index:

   `- `references/09-engine-lifecycle.md` — eager/lazy TensorRT initialization, cold/hot file cache, CUDA Graph setup, startup/first-request/steady-state timing`

2. Add one sentence to Quickstart step 1:

   `If Engine, context, warmup or CUDA Graph setup can be lazy, define whether it belongs to Startup-to-Ready or the first request; see `references/09-engine-lifecycle.md`.`

No new core rule or new workflow stage is needed.

## `references/01-scope-baseline.md` — minimum checklist addition

Under “Define the actual target before profilers,” add:

- runtime lifecycle state: serialized Engine only, deserialized Engine, context/profile/buffers ready, warmed, and/or graph captured;
- endpoint: Startup-to-Ready, Ready-to-End (first request), process-launch-to-first-result, or steady state; note which initialization is eager versus lazy;
- Engine storage device/path, hash and file-cache state when load time is in scope.

This prevents a cold first request from being silently compared with a warmed request.

## `references/08-benchmark-protocol.md` — minimum checklist addition

Under “Recording minimums,” add:

- Lifecycle: per-round file-read, deserialize, context, buffer/profile, warmup and CUDA Graph capture/instantiate/replay timings; declare where each phase belongs in the reported endpoint.
- Storage/cache: source/target device and filesystem verified with `findmnt`/`lsblk`/`df`, Engine SHA-256, cold/hot Page Cache method plus residency evidence; a new process alone does not establish cold cache.
- Attribution: keep host wall, CUDA Event and Nsight Systems GPU-kernel intervals separate; flag first-use host stalls and I/O failures/retries.

Under the sequencing section, add:

`When storage or lifecycle is the hypothesis, first compare byte-identical Engines and separate cold-cache initialization, hot-cache initialization and warmed inference. Do not report a storage-only result as a GPU-kernel speedup.`
