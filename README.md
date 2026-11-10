# trt-oplab
# Operator Optimization Workbench — v0.3

An **evidence-led starter kit**, not a fully connected LingBot production pipeline.
Designed around the real workflow: recorded-history full-model inference →
operator extraction → local replay → kernel/full-call benchmark → TensorRT
integration → Engine Inspector diff → multi-chunk replay.

## What this delivers

1. **SKILL.md**: a short navigation and decision-oriented workflow for an agent/developer.
2. **references/**: specific contracts and failure modes, rather than a long one-shot prompt.
3. **helpers/**: runnable, standard-library-only scripts that work *without GPUs* for
   `layerinfo.json` analysis and experiment evidence state.
4. **templates/**: capture, review and JSON benchmark-report contracts.
5. **examples/**: no project-specific TensorRT files bundled by default; generate
   reports from your own files in your company workspace.
6. **project validator integration**: `references/07-validation-harness.md` describes
   how to use an existing `operator_validation/` fixture and negative tests;
   `helpers/evidence_audit.py` audits JSON reports, not raw tensors.
7. **benchmarking**: `references/08-benchmark-protocol.md` distinguishes a reduced
   correctness example from production-size performance comparison.

Do **not** install one skill per stage yet. One skill guides work; reference pages hold
knowledge; helpers perform deterministic, testable tasks; each experiment directory
holds the actual evidence. This separation prevents prompts from becoming the only
state store. Add a higher-level orchestrator only after multiple completed experiments
show a real coordination bottleneck.

## 3-minute example (no TensorRT environment required)

```bash
python helpers/layerinfo.py summary /path/to/action_engine_layerinfo.json \
    --out /tmp/action-summary.json
python helpers/layerinfo.py diff BASE.layerinfo.json CANDIDATE.layerinfo.json \
    --out /tmp/layer-diff.json
python helpers/experiment.py init --out /tmp/opt-case-001 \
    --hypothesis "Remove KV pack without hurting end-to-end latency" \
    --target "LingBot Action self-attention"
python helpers/experiment.py status /tmp/opt-case-001
```

Experiment files are structured *evidence indexes*. GPU-dependent capture and
performance runners must be **adapted to the actual project**, engine profiles,
plugin registration and model source; this starter kit deliberately does **not**
pretend to supply a generic working capture/benchmark of any `.engine`.

## Report audit (no raw safetensors required in this package)

```bash
python helpers/evidence_audit.py --cpu example_cpu.json --cuda example_cuda.json \
  --capture source_capture.json --out audit.json
```

The checker detects missing/failed reported checks and cross-report anomalies; it
**does not independently rerun the source Engine, load Q/K/V or prove GPU speed**.
Production fixtures stay in the company repository.

## Working contract

| Stage | Exit evidence | May proceed when |
|---|---|---|
| baseline | environment, real workload, layer/trace evidence | target cost identified and measurement boundary known |
| capture | captured inputs/reference output, metadata, provenance | tensors are at the same mathematical boundary; repeated replay is stable |
| local | numeric comparison, kernel-only and full-call costs | correctness and full-call benefit are credible *or* integration is an explicit hypothesis |
| integration | candidate Engine, structural diff, numerical evidence | parser/build/run succeed, adjacent graph changes explained |
| e2e | matched replay metrics and full-model outputs | accepted based on predeclared thresholds; otherwise iterate/reject |

At every stage, maintain explicit **known / inferred / unknown** distinctions.
No pass gate is an automated correctness proof.

## Example project wiring (manual to start)

```text
Existing LingBot replay harness
    ├── export/capture script             (project-specific)
    ├── standalone Attention/FFN harness  (project-specific)
    ├── CUDA/Triton/cuDNN candidates      (project-specific)
    ├── ONNX rewrite + Plugin build       (project-specific)
    └── 18-chunk offline replay           (project-specific)
                     │
                     ▼
         experiment.json + evidence/
                     │
             helper scripts
                     │
                     ▼
              decision.md
```

## Evidence and limits

The original supplied Action Engine Inspector file (not shipped in the Skill ZIP) includes TensorRT layers with names like
`_gemm_mha_v2` and `cutlass3x_sm100...`; that is **compiled structure**, not actual
kernel time and not accessible Q/K/V values. The provided offline replay harness
captures adapter/transformer-level snapshots, not automatic layer-14 internal Q/K/V.
The latter requires a dedicated graph debug/capture mechanism.

Official sources to consult for the **installed** TRT version:

- Engine Inspector: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/EngineInspector.html
- Custom Layers / Plugins: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html
- Performance benchmarking: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html
- TRT 10→11 inspector JSON schema migration: https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-IEngineInspector.html

This helper targets the supplied TRT 10-style `Layers` array; future JSON field
names may differ, so parser failures should not be silently masked.

## Files

```text
SKILL.md
README.md
AGENTS.md
references/{01..08}-*.md
templates/{capture_contract.md,decision.md,benchmark_report.json}
helpers/{layerinfo.py,experiment.py}
tests/{test_helpers.py,test_evidence_audit.py}
examples/README.md
```

## Companion task for company-side Codex

See `CODEX_COMPANY_TASK_CN.md`. It requests reproducible run commands and
**sanitized metadata/aggregate timings only**; never copy raw company samples,
weights or binaries into an installable Skill.

