# 11. Invariant subgraph precompute and cross-forward reuse

## Candidate, not proof

Look for `Y = f(C, W)` where condition `C` and weights `W` stay fixed while current latent/state `X_t` changes. Example: in a denoising Transformer, text-side **cross-attention K/V projections** may be invariant while self-attention Q/K/V, cross-attention Q, and FFN remain state-dependent.

Separate these caches: a historical/paged **self-attention KV cache** solves a different problem from precomputing **cross-attention condition K/V**.

Graph dependency proves independence from omitted dynamic inputs; it does **not** prove actual work-item inputs are identical. Verify the complete binding tensor's content, dtype, shape, batch/CFG order and preprocessing in an *unprofiled diagnostic run*. Do not perform per-forward GPU→CPU hashing in the timed path.

## Prototype boundary

```text
episode/task start + immutable condition
          ↓
   precompute subgraph → persistent GPU condition cache
          ↓
 current latent/timestep/history → main Engine consumes cache
          ↓
 output + unchanged paged/state-commit semantics
```

- Surgery must preserve graph output semantics, precision, all consumers and external weights. Preserve remaining branches of a shared condition embedder.
- Keep precompute/main Engine and model identities; pack cache bindings only when layout conversions do not erase savings. Many new inputs may disrupt TensorRT fusion/tactics.
- Explicit cache ownership and invalidation: episode/task, condition bytes/order, CFG layout, batch shape, dtype, quantization/precision, model/weight and Engine version. Never key solely by pointer address.
- Cache writes must finish before consumer streams read; keep buffers alive across all reads, avoid unintended in-place mutation and reset state on new episodes.
- Evaluate per-mode independent caches unless shared weights **and** input identity are proven.

## Cost model and validity

For a fixed task, the useful comparison is:

```text
baseline_total = sum(forward_baseline)
cached_total   = sum(cache_miss_precompute) + sum(forward_hit)
               + required cache maintenance/copies/synchronization
net_saved      = baseline_total - cached_total
```

Include one-time cache construction in a **task/Episode** result. A cache-hit Forward may improve even when cache-miss Forward regresses. Report break-even forwards and peak memory; logical cache tensor sizes are not peak process memory.

Validation tiers: individual projected tensors → isolated main Engine → repeated denoising → stateful multi-chunk replay (including commit/discard) → paired full workload. Check requested outputs, KV state and final outputs; if bitwise equality is observed, report its limited tested scope rather than assuming every mode is bitwise.

**Accept** only reproducible task-level net gain under the declared lifecycle; **reject** if changed tactics/reformats/memory invalidate the saving; **blocked** if invalidation or numerical contracts are incomplete.

## Transferable case pattern (no private workload data)

A fixed-condition two-mode inference pipeline may show a useful gain after caching many layer-specific K/V projections on GPU. The transferable lesson is **removing repeated work across forwards** and charging the miss cost once, not any particular model size, timing or proprietary trace.
