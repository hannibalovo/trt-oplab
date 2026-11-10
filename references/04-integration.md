# 04. Integration: choose the Plugin boundary, rebuild, compare

## Distinct graph boundaries

The selected study region might be **just SDPA**. Integrating into TensorRT
could mean replacing only the core MHA or absorbing adjacent layout/Norm/RoPE.
A larger Plugin can reduce extra copies, but can also destroy optimized native
GEMM fusion. Smaller is not automatically better. Decide using local cost and
pre-existing TRT fusion/tactic evidence.

## Integration checklist

1. Save baseline Engine + ONNX + LayerInfo, builder logs, profiles, custom plugin
   paths/version/creator/namespace, and source/control hashes.
2. Rewrite an **isolated candidate ONNX**. Assert target match count, exact
   input/output tensor semantics, preserve external data and remaining nodes.
3. Register/load Plugin implementation at **build and runtime** as required.
   Plugin is not merely an ONNX name: it needs compatible shape/type/format,
   execution and serialization contracts.
4. Build candidate Engine with matched precision, profiles, target device and
   as many identical build settings as feasible.
5. Perform same-input isolated **full-Engine numerical comparison**, then longer
   recorded-history replay (including KV commit/discard state).
6. Inspect new LayerInfo using `helpers/layerinfo.py diff BASE.json CAND.json`.
   Mark missing semantic anchors as **unresolved**, not automatic fusion changes.
7. Run Nsys/NCU on the affected region; compare actual launch pattern, reformat
   and synchronization costs. Inspector JSON alone has **no durations**.

## What to compare around a target

```text
baseline: A+B → native MHA → out projection
candidate: A' → Plugin → B' → out projection'
```

- target MHA node (removed/replaced?) and its source ONNX scope;
- immediate producer/consumer layer type, dtype, shape, format, tactics;
- reformat/copy/transpose materialization;
- globally changed grouped projections, cross-block fusions and memory planning;
- engine runtime, correctness, and baseline as a control.

**Important**: Inspector `Layers[i]` is not a stable identifier between builds.
Use semantically unique ONNX node IDs, graph edges, and source mappings. A full
fusion change can make one-to-one alignment impossible; report ambiguity.

## Exit gates

- Builder and runtime both succeed (including cold process plugin registration).
- Numeric gates pass on the fixed suite; no unexplained NaN/Inf.
- Structural deltas have a written explanation (or are documented unknowns).
- At least one production-like workload has an integration cost measurement.

If multi-plugin combinations cause nonfinite results, **block** final acceptance
and bisect layer count/order, shapes, lifecycle and memory aliasing. One-layer
correctness is not proof of 30-layer safety.
