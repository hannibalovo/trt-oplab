# trt-oplab

An evidence-led workbench for GPU operator experiments in TensorRT-based
inference systems. It guides one optimization hypothesis from workload
definition through capture, local validation, integration, and end-to-end
review. It does not provide a universal capture or benchmark runner.

## Contents

- `SKILL.md`: short workflow, gates, and navigation.
- `references/`: workload, capture, validation, integration, benchmarking,
  Engine lifecycle, and paged-KV contracts.
- `helpers/`: standard-library tools for Inspector JSON analysis, experiment
  state, and offline evidence-report consistency.
- `templates/`: capture, decision, and benchmark-report forms.
- `examples/`: brief anonymized cases that distinguish observations from
  unresolved claims.

Raw model inputs, weights, serialized Engines, credentials, private paths, and
project-specific reports belong in the source project, not in this repository.

## Quick start

```bash
python helpers/layerinfo.py summary /path/to/layerinfo.json \
  --out /tmp/summary.json
python helpers/layerinfo.py diff /path/to/base.json /path/to/candidate.json \
  --out /tmp/diff.json
python helpers/experiment.py init --out /tmp/opt-case \
  --hypothesis "Reduce a measured operator cost" --target "declared workload"
python helpers/experiment.py status /tmp/opt-case
```

These commands inspect supplied metadata and record evidence; they do not
capture tensors, load an Engine, run a GPU benchmark, or establish numerical
correctness. Adapt project-specific execution through the existing project
harness. `helpers/evidence_audit.py` checks consistency of reported JSON claims;
it does not recompute tensor values or prove performance.

## Working contract

| Stage | Evidence |
|---|---|
| Baseline | workload, environment, measurement boundary, and hotspot evidence |
| Capture | tensor semantics, provenance, and repeatable reference output |
| Local | numerical/behavior validation and correctly scoped K/O measurements |
| Integration | candidate Engine, structural diff, and matched numerical evidence |
| End to end | matched replay, state/quality guardrails, and application latency |

At every stage, label confirmed facts, reported claims, hypotheses, and unknowns
separately. A helper can check report structure; it cannot prove that the
underlying experiment was performed correctly.

## Further reading

See `SKILL.md` for the entrypoint. `references/06-governance.md` defines the
separation between navigation, references, helpers, project adapters, and
experiment evidence.
