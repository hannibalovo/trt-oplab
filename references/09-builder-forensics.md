# 09. TensorRT Builder experiments: provenance before attribution

## Why this page exists

A faster rebuilt Engine is not proof that one Builder flag caused the gain. Builders search different tactic/fusion spaces; changing profiling verbosity, workspace, timing iterations or timing cache can yield a different binary. Treat each Engine as a distinct measured artifact.

## Immutable build fingerprint

Record together:

- ONNX graph SHA and external-data/weight hashes; plugin source and binary versions;
- GPU architecture, driver, CUDA, TensorRT, builder invocation and build log;
- optimization level, workspace **limit** (not actual allocation), tactic sources, timing iterations, timing cache identity/hit evidence, precision, strongly-typed setting, min/opt/max optimization profiles;
- profiling verbosity, Engine SHA, size, Inspector export, build time, peak build memory if actually measured;
- runtime ABI, input shapes and cache state for every performance comparison.

An Engine produced with `DETAILED` Inspector metadata is a **diagnostic rebuild**, not automatically the previously built production Engine. A diagnostic engine's tactics and numeric outputs may differ from the historical engine, even when ONNX and plugin hashes match.

## Experiments that answer the right question

1. Freeze original Engine and its input fixture. Benchmark it without rebuilding.
2. Change one Builder dimension at a time where practical; if several dimensions change, describe the trial as a **bundle**, not a causal test.
3. Compare the **old Engine against the new Engine** on the same replay workload, not different dates or inputs.
4. Use paired/reversed run order, multiple repetitions, and separate per-mode Forward, complete replay and startup cost.
5. Explain LayerInfo/ONNX fusion and tactic changes; Layer indices are not semantic IDs.
6. Keep negative and near-zero results. A larger workspace or longer tactic search is not a monotonic speedup guarantee.

If original flags were not saved, label them **unknown**. Do not fill a supposed default optimization level from recollection.

## Common traps

- Workspace cap ≠ actual consumed VRAM.
- Timing-cache file present ≠ measured cache hit; inspect build diagnostics.
- `LAYER_NAMES_ONLY` cannot retroactively reveal tactics in an old Engine.
- A dynamic `OPT` point affects only actual declared dynamic input dimensions; a fixed-shape paged KV pool with runtime lengths is not magically optimized for a chosen history value.
- Rebuilt DETAILED output differences do not, by themselves, implicate a particular flag.
- Comparable Engine sizes or Layer names do not prove identical kernels.

**Decision:** once additional Builder searches produce no reproducible net gain, stop tuning flags and return to model/dataflow cost analysis.

See also `10-module-cost-attribution.md` and `13-benchmark-lifecycle.md`.
