# 10. TensorRT integration contract for paged KV attention

This reference covers a TensorRT Engine whose attention layers read committed
K/V from a physical page pool and receive current K/V in separate staging
pages. It complements `02-capture-contract.md`, `04-integration.md`,
`05-review-decisions.md`, and `07-validation-harness.md`. It does not prescribe
a particular paged-attention library or claim that any one Engine rewrite is
numerically equivalent to native TensorRT MHA.

## 1. Dense-history and paged-KV are different Engine contracts

Write down the full input ABI before building either Engine.

| Property | Dense-history Engine | Paged-KV Engine |
|---|---|---|
| History input | Dense K/V in logical token order, commonly `[L,B,T,H,D]` or one per-layer tensor | Physical pool plus a page table; logical order is defined by the table |
| Current K/V | Often concatenated with history or passed to the attention subgraph | Separate current K/V inputs or Engine-produced current K/V, staged into reserved pages |
| Length | Dense tensor extent and/or mask | Per-batch effective sequence length, independently checked against table capacity |
| Storage guarantee | Usually contiguous at the Engine boundary | Pool may be physically fragmented; only each page's specified region is addressable |
| Engine graph | Native gather/concat/MHA path | Rewritten attention node(s) plus a page-aware Plugin; dead dense-history branches must be pruned |

Record, per tensor: semantic role, layer/batch ownership, shape, dtype, stride,
alignment, physical address space, mutability, producer/consumer, and lifetime.
Record whether page IDs are global or batch-local. Record the exact mask,
position, scale, RoPE, causal, and current-token ordering semantics. A page-table
input added to the binding list does not make a dense-history Engine paged.

## 2. Physical pool, table, and staging capacity

For page size `P`, committed capacity `C`, current tile length `Q`, and a
separate staging capacity `S`, define the contract explicitly:

```text
committed pages = ceil(C / P)
staging pages   = ceil(Q / P)       # or ceil(S / P) if the Engine reserves a maximum tile
table capacity  >= committed pages + staging pages
pool capacity   >= C + S tokens, with the staging region outside committed slots
```

If the last committed page can be partial, specify its valid-token count and
how the next current tile begins. Do not silently assume page-aligned history.
For variable history length, pass or update `seq_len` per batch and make every
padding table entry a valid in-bounds page that cannot contribute to attention.
Never use `-1` as padding unless the backend explicitly defines and validates
that sentinel.

Compute the bound from maximum live history and current-tile capacity for each
supported mode, including a partially occupied final page and the full-cache
case after eviction. Add small arithmetic tests for boundary lengths; do not
copy a production capacity into a reusable example.

Use separate K and V planes with documented layout, e.g. one per-layer
`[B,T,H,D]` BF16 allocation. If Q uses `[B,H,Q,D]`, document the conversion or
plugin descriptor; do not infer layout from dimension names. Prove that
historical reads use the page table directly. Check the rewritten ONNX and
serialized Engine for removed `history_k/history_v` bindings, dead gather,
pack, clone, transpose materialization, or hidden contiguous conversion.
Current-tile staging is allowed when explicit; it is not evidence that
historical KV is still being gathered.

## 3. Logical order and cache-state invariants

The table is an ordered mapping from **logical token time** to **physical
page**. Physical page ID order is not temporal order. For every call:

1. The valid table prefix names exactly the committed logical history in
   oldest-to-newest order, followed by the current staging page(s).
2. `seq_len` equals valid history tokens plus valid current tokens for that
   batch. Padding pages are excluded by length/mask.
3. No staging page is marked committed. Repeated denoising calls overwrite
   staging and do not append history.
4. Predicted and real commits follow the source Runtime's distinct rules. A
   prediction-clear operation invalidates metadata as specified by that Runtime;
   it need not zero the underlying bytes.
5. Eviction removes the oldest logical token/page according to Runtime
   metadata, then physical slots may be reused only after they are unreachable
   from all live tables and outstanding device work.
6. Reset/Clear restores the defined empty logical state, lengths, and metadata.
   Stale bytes may remain only if no valid length or page mapping can expose
   them.
7. Batch and layer ownership are isolated. A table for one batch/layer cannot
   expose another batch/layer's pages, including unused padding entries.
8. Commit, page-table update, staging copy, attention launch, eviction, and
   buffer reuse are ordered on known CUDA streams. Any cross-stream handoff
   uses explicit events/dependencies and keeps all referenced memory alive.

Test these invariants against an independently maintained logical contiguous
KV reference. At each operation, compare token identities/values and exact
metadata: append, predicted commit, prediction clear, real commit, eviction,
physical-page reuse, staging overwrite without commit, batch isolation,
layer isolation, and reset. A legal table is not necessarily in the right
logical order; matching metadata is not proof that KV bytes match.

## 4. ONNX rewrite and TensorRT Plugin integration

### Rewrite boundary

- Identify attention nodes by stable source node names/edges and layer mapping,
  not only by Inspector index or a repeated layer name.
- Replace the intended Self-Attention subgraph and preserve its output
  semantics, residual path, projections, masks, position handling, and current
  K/V outputs. Count replacements and fail the build if the count differs from
  expectation.
- Prune the now-dead dense history gather/concat path and assert no dense
  history bindings survive. Preserve weights and external ONNX data; hash both
  source and rewritten artifacts.
- Build a separate candidate Engine. Record profiles, precision, formats,
  workspace limits, TensorRT/cuDNN versions, plugin creator name/version/
  namespace, and build log.

### Registration and runtime resources

Register or load the Plugin creator before parsing/building an ONNX that uses
it, and before deserializing an Engine that contains it. Test build and cold
runtime in separate processes. Treat duplicate registration and import order as
explicit failure modes; do not silently register another creator with an
ambiguous namespace/version.

The Plugin must define supported shapes, dtypes, formats, dynamic dimensions,
serialization, workspace, output ownership, aliasing, and enqueue-stream
behavior. Use the TensorRT-provided stream for staging and backend execution,
or declare and enforce stream dependencies. Do not assume callback order,
default-stream synchronization, or that an Engine binding stays alive after
enqueue returns.

Backend handles, execution plans/graphs, descriptors, workspace, and bindings
must have explicit ownership and teardown. Key resources by stable Engine or
execution-context identity plus shape/device as required, not by a recyclable
raw pointer alone. On every invocation, bind current pointers and stream as
required by the backend. Ensure workspace and input/output allocations outlive
all queued work. Check input/output byte-range overlap and TensorRT memory
planning; avoid aliasing unless the Plugin contract explicitly supports it.
Exercise repeated enqueue, multiple contexts, stream changes, teardown, and
recreation. A Python prototype's module-global cache is not by itself a
production resource-lifecycle design.

## 5. Staged validation gates

Do not skip directly from one successful SDPA call to a full replay.

| Gate | Required evidence | What it does not prove |
|---|---|---|
| Direct SDPA | Same Q/K/V bytes, descriptor, lengths, mask, scale, dtype and output boundary; compare Paged with Contiguous and an independent math reference | TRT Plugin correctness or full Engine equivalence |
| One Plugin / one layer | Real layer binding, shape/stride, finite output, caller-owned output and guard checks; compare direct backend and Engine output | Multi-Plugin resource safety or full-Engine correctness |
| Two Plugins | Same Engine/context; inspect both plugin inputs/outputs, addresses, streams, workspace and next-layer input; test contiguous and paged controls | All-layer correctness |
| Full Engine | All intended layers replaced; no hidden dense history path; compare final outputs and current K/V; check NaN/Inf and memory safety | Repeated denoising or cache lifecycle correctness |
| Denoising loop | Every real scheduler step from the same initial state; compare each noise output and current K/V; verify staging overwrite and no repeated commit | Predicted-history trajectory behavior |
| Chunk replay | Each path owns its cache and commits its own prediction; exercise real commit, clear, eviction, reset, full window and page reuse over recorded chunks | Task acceptance or hardware safety unless those gates are separately defined |

Only advance after structural cache assertions pass. Record exact tested
history lengths, current lengths, boundary/full-cache conditions, modes,
layers, and chunk/step coverage.

## 6. Debugging nonfinite outputs in multi-Plugin Engines

NaN/Inf after a multi-Plugin rewrite is an integration failure to localize,
not immediate evidence that paged addressing is mathematically wrong. A
single-layer Paged-vs-Contiguous pass isolates only that tested backend call.

Use this sequence:

1. Freeze identical Q/K/V, page table, lengths, masks, and descriptors. Run
   direct Paged and Contiguous backend calls. Check page IDs, lengths, finite
   outputs, and byte-level input agreement.
2. Run one Plugin in an otherwise matched Engine. Check inputs, output range,
   stream, workspace size/address, backend handle/resource identity and
   lifetime.
3. Add one Plugin at a time, then the smallest failing pair. Include an
   all-Contiguous multi-Plugin control. If that control also fails, the failure
   is not specific to paged addressing.
4. On first nonfinite value, stop and capture the failing plugin's inputs,
   output, next layer input, layer index, effective lengths, page IDs, device
   addresses/ranges, stream handle, context, resource key, workspace and
   synchronization state. Find the first failing output, not just final model
   output.
5. Check input/output aliasing, batch offsets, staging-copy byte ranges,
   allocator lifetime/reuse, descriptor shapes/strides, graph capture and
   resource mutation. Use sanitizers where available.
6. Repeat with and without diagnostic outputs. Exposing intermediates can
   change TRT tactics or fusion. Compare each tapped Engine against its own
   untapped counterpart before treating tapped activations as production
   internals.

Keep a failure matrix by plugin count/order, layer pair, backend, context, and
stream. Do not expand replacement count until the first failure is explained.

## 7. Numerical acceptance has separate levels

Report these as distinct checks, never as one generic “cache passed” result:

1. **Metadata agreement:** valid lengths, IDs, predicted/real markers, eviction
   state and reset state match the reference contract.
2. **Page-index legality:** all valid and padding entries address allocated
   pages belonging to the intended batch/layer; lengths do not exceed backing
   capacity.
3. **Logical-order correctness:** resolving the table yields exactly the
   intended oldest-to-newest history followed by current tokens. Validate
   token IDs/values at page boundaries and after eviction/reuse.
4. **KV-content agreement:** compare current and committed K/V values with the
   chosen numerical reference. Hashes detect change but do not measure
   magnitude; report elementwise counts and error distributions.
5. **Attention/backend agreement:** same-input Paged-vs-Contiguous agreement is
   separate from either backend's agreement with TRT MHA or an FP32 reference.
6. **Model/task acceptance:** assess final model outputs and task-specific
   trajectory/quality criteria against predeclared, calibrated limits. A
   finite output or small average error alone is not task acceptance.

Apply the frozen numerical rule and report format in
`07-validation-harness.md`; do not create a paged-KV-specific tolerance. Use an
independent appropriate-precision reference where possible. Track fixed-state
compute error separately from state divergence after each path commits its own
predicted K/V. Compare final model outputs over the declared repeated steps and
chunk boundaries, then apply the separately declared task-quality gate.

## 8. Performance attribution and unresolved spikes

Use same-process paired arms on identical inputs and cache state, alternating
or randomizing AB/BA order. Measure separately:

- baseline pack refresh + Engine;
- baseline reused packed KV + Engine;
- candidate page-table preparation + Engine;
- candidate reused page table + Engine;
- plugin/backend event spans, TRT layer-profiler spans, Engine CUDA/host spans,
  and full Generate/Commit/Chunk wall time.

Include startup/prewarm policy and isolate fixture I/O. Report medians and all
valid paired samples. Do not subtract profiler sums from CUDA events or host
wall to manufacture an Attention-only delta; their boundaries and overhead
differ. A local Forward improvement can be diluted by other modes, application
preprocessing, commit work, synchronization, page preparation, and call count.

Treat an intermittent Chunk N latency spike as an observation until reproduced
under controlled paired runs and traced to a measured span. If a later run does
not reproduce it, say so; absence of the spike does not prove that a memory
problem was fixed. Keep cache pressure, page-table work, pack misses, allocator
activity, GPU clocks/contention, application work, Engine tactic/launch changes, and storage
as hypotheses until evidence separates them.
