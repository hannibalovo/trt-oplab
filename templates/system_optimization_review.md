# System optimization review

## Identity and lifecycle
- Experiment / repository commit / candidate commit:
- Source graph + external weight hashes:
- Baseline Engine SHA / candidate Engine SHA:
- GPU / driver / CUDA / TRT / plugins:
- Input shape, dtype, strides, mode, batch/CFG, timestep/steps:
- Cache state and episode/task identity:
- Measurement windows and inclusion rules:
- Ready condition; cold/setup/precompute/miss handling:
- Paired order and repeats:

## Hypothesis
- Target semantics and why work is avoidable / implementation is suboptimal:
- Model → ONNX → TRT Layer → actual GPU kernel evidence:
- Known / inferred / unknown:
- One change only? Any Builder/Tactic confounders?:

## Correctness
- Per-output numerical rule and artifact:
- State/cache/commit audit and multi-step scope:
- Nonfinite / bitwise / max-abs / rel-L2:
- Limits and untested invalidation cases:

## Cost (keep columns separate)
| Boundary | Baseline | Candidate | Unit | Method | Artifact |
|---|---:|---:|---|---|---|
| kernel K | unknown | unknown | ms | Nsys/event, exactly named | |
| full op O | unknown | unknown | ms | event/wall as specified | |
| engine Forward | unknown | unknown | ms | as specified | |
| ready replay | unknown | unknown | ms | wall and setup charging | |
| cold start | unknown | unknown | ms | wall | |
| cache memory/peak GPU | unknown | unknown | MiB | allocation evidence | |

- Precompute/graph-build cost and actual hit/miss counts:
- Per-chunk or per-step changes and unexpected regressions:
- Missing comparable measurements (explicitly **unavailable**):

## Decision
- accept / iterate / reject / blocked:
- Measured gain at which boundary:
- Key failure mode / negative control:
- Reproducibility command and artifacts:
- Deployment lifecycle/invalidation/rollback risks:
