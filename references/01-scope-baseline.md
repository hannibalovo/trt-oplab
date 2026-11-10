# 01. Scope, workload and baseline

## Question first

What *end-to-end* quantity are we improving? For a LingBot offline 5/5 replay,
there are two `model.infer` requests per chunk (`infer_generate` and
`infer_commit`), not one Transformer forward. The 5+1 Video, 5+1 Action,
and 1+1 real-commit forwards amount to **14 complete Transformer forwards** per
chunk in that setup. This is not a mechanical rollout metric.

Define the actual target before profilers:
- workload fingerprint: model/weights + ONNX graph + engine/build flags, min/opt/max
  profile, precision, GPU/TRT/CUDA versions, cache occupancy and layout;
- representative cases: first/partial/full KV, Action/Video, draft/pred/real;
- primary measure: `per-chunk generate+commit wall ms` **or** another declared metric;
- constraints: accuracy tolerance, warmup, memory ceiling, build/run overhead;
- negative control: original engine and identical replay inputs/commits.

## Measurement hierarchy and attribution

```text
application wall (data preprocessing + Python + synchronization)
  └─ Transformer engine execution (many TRT Layers)
       └─ kernels in Nsys (each launch duration)
            └─ single kernel metrics in NCU
```

CUDA Event elapsed, wall time and sum of GPU kernel durations measure different
things. Do not add nested intervals or compare unmatched scopes.

Use Engine Inspector to identify *structure* and Nsys/NCU for *actual launches*:

```bash
python helpers/layerinfo.py summary baseline.layerinfo.json
# If obtaining a report with installed trtexec, verify version/flags:
# trtexec --loadEngine=... --profilingVerbosity=detailed --dumpLayerInfo
# trtexec --loadEngine=... --dumpProfile  # adds profiler overhead
```

A layer's `TacticName` is a useful structural identifier, not proof that all
instances have identical execution latency. Inspector layer indices can change
between rebuilds. Prefer ONNX source scope / unique semantic anchors.

## Exit criteria

- a saved immutable baseline with hardware/shape/cache/input provenance;
- a mapped hotspot: PyTorch/module → ONNX node(s) → TRT layer(s) → Nsys kernel(s);
- a stated hypothesis explaining **why** the candidate should beat this path;
- one headline outcome and guardrails, predeclared before candidate tuning.

If the hotspot has not been identified, stop. No Plugin choice yet.
