# Proposed incremental edits for references 04, 05, and 07

Proposal only. These snippets are not yet merged into the active Skill
references. See `10-paged-kv-contract.md` for the full candidate contract.

## `04-integration.md`

Add after “Distinct graph boundaries”:

> A paged-KV Engine has a different input ABI from a dense-history Engine.
> Specify physical pool layout, page-ID namespace, valid table prefix,
> per-batch sequence length, current-KV staging and maximum capacity before
> rewriting ONNX. Verify the serialized Engine has no dense `history_k/v`
> bindings or hidden history gather/pack/clone/contiguous path. The table must
> represent maximum committed history plus all current staging pages.

Add to the integration checklist:

> Define Plugin creator name/version/namespace and register it before ONNX
> parsing/build and Engine deserialization. Exercise cold-process runtime
> loading. Define handle, descriptor, execution-graph, workspace, binding and
> teardown ownership; use the TensorRT-provided stream or explicit stream
> dependencies. Check pointer lifetimes and input/output aliasing.

Add after the multi-Plugin NaN/Inf gate:

> Compare direct same-input Paged and Contiguous backend calls first. Then test
> one Plugin, the smallest failing multi-Plugin subset, and an all-Contiguous
> multi-Plugin control. Capture the first nonfinite plugin output and its next
> consumer input, addresses, stream, workspace and resource identity. A
> failing Contiguous control does not implicate paged addressing. Diagnostic
> taps may change TRT tactics; compare tapped and untapped Engines.

## `05-review-decisions.md`

Add to the review rubric:

> - Are metadata agreement, legal page IDs, logical token order, KV numerical
>   agreement, and task-quality acceptance reported as separate gates?
> - Were commit, predicted-clear, real commit, eviction, reuse and reset
>   validated against an independent logical KV reference?
> - Were cold pack, reused pack, page-table preparation, reused page table,
>   plugin events, Engine spans and complete Chunk wall measured separately?

Add to the failure decision tree:

> - **A spike disappears in a later run** → mark its cause unresolved. Do not
>   claim a memory or allocator fix without a matched trace and controlled
>   reproduction.
> - **Page table is legal but outputs differ** → verify logical order and KV
>   bytes, then compare backend, Attention, block, full Engine and task output
>   separately.
> - **All-Contiguous multi-Plugin control also becomes nonfinite** → investigate
>   Plugin resource lifetime, stream ordering, aliasing and TRT planning before
>   attributing the failure to paged indexing.

## `07-validation-harness.md`

Add to “Portable fixture contract”:

> For paged KV, record physical pool shape/strides and per-layer/batch
> ownership; page size and page-ID namespace; table shape, valid prefix and
> padding rule; sequence lengths; history/current ordering; staging capacity;
> maximum committed history; and commit/clear/eviction/reset semantics. Keep
> capture and raw data inside the source organization when they are
> proprietary.

Add to “Minimum candidate checks”:

> Verify metadata, page-index bounds/ownership, logical order and KV-content
> agreement separately. Test staging overwrite without commit; predicted
> commit; prediction clear; real commit; eviction and physical-page reuse; and
> reset. Run one layer, a small multi-Plugin Engine, a full Engine, real
> denoising steps, then recorded chunks. For repeated-state paths, report
> per-step/per-chunk error distributions and compare both fixed-state compute
> error and divergence after each candidate commits its own prediction.

Add to the performance checkpoint:

> Pair baseline pack-refresh and pack-reuse arms with candidate table-refresh
> and table-reuse arms. Keep Plugin events, TRT profiler spans, Engine timing
> and end-to-end Generate/Commit/Chunk wall as separate measurements; do not
> infer end-to-end gain from one Engine Forward.
