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
- `templates/`: capture, decision, benchmark-report, and system-review forms.
- `examples/`: three brief anonymized cases that distinguish observations from
  unresolved claims; start at [examples/README.md](examples/README.md).

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

## Field lessons

The original 8-step workflow remains the entrypoint. These references collect
focused lessons while preserving their evidence limits:

| Reference | Reusable question |
|---|---|
| [09 Builder forensics](references/09-builder-forensics.md) | Does a rebuild isolate the claimed Builder change? |
| [10 Module-cost attribution](references/10-module-cost-attribution.md) | Where did GPU time go without double-counting fusion? |
| [11 Invariant subgraph cache](references/11-invariant-subgraph-cache.md) | Can immutable condition-side work be reused? |
| [12 GEMM negative results](references/12-gemm-negative-results.md) | Is the candidate faster at a matched boundary? |
| [13 Benchmark lifecycle](references/13-benchmark-lifecycle.md) | Are setup, replay and task costs accounted for? |
| [14 Conditioning/AdaLN](references/14-conditioning-modulation.md) | Is modulation reuse real, or only a logical-size claim? |
| [15 Paged KV contract](references/15-paged-kv-contract.md) | Are page ABI, order, staging, lifecycle and Plugin ownership correct? |
| [16 Engine startup lifecycle](references/16-engine-startup-lifecycle.md) | Which Engine initialization phases contribute to first result? |

See `SKILL.md` for the 8-step entrypoint and `references/06-governance.md` for
the separation between navigation, references, helpers, adapters and evidence.


## Repository layout

```text
references/{01..16}-*.md
helpers/{layerinfo.py,experiment.py,evidence_audit.py}
templates/{capture_contract.md,decision.md,benchmark_report.json,system_optimization_review.md}
examples/{README.md,engine-lifecycle-case-study.md,paged-kv-contract-case-study.md,ffn-kv-pack-case-study.md}
tests/{test_helpers.py,test_evidence_audit.py}
```
