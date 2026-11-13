# 01. Scope, workload and baseline

## Question first

What *end-to-end* quantity are we improving? Count the actual model/runtime
calls made by one user-visible request, including generation, commit, retries,
and other modes. A single Transformer forward may represent only part of a
request; state the call count and boundary for the workload being measured.

Define the actual target before profilers:
- workload fingerprint: model/weights + ONNX graph + engine/build flags, min/opt/max
  profile, precision, GPU/TRT/CUDA versions, cache occupancy and layout;
- representative cases: empty/partial/full KV and each supported mode or
  commit path;
- replay semantics: whether later chunks consume predicted, demonstrated, or other
  reference actions as history; initial cache state and prediction-clear, real-commit,
  and eviction behavior. These histories are different workloads and must not be
  paired as equivalent inputs;
- primary measure: `per-chunk generate+commit wall ms` **or** another declared metric;
- measurement horizon: first-use latency versus steady replay, chunk range/count,
  and whether Engine initialization, CUDA Graph construction, and warmup are included;
- runtime lifecycle state: serialized Engine, deserialized Engine, context/profile/
  buffers ready, warmed, or graph captured; state which steps are eager or lazy
  and whether the target is Startup-to-Ready, first-request, or steady state. See
  `09-engine-lifecycle.md` when initialization is in scope;
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
