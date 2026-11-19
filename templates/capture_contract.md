# Operator Capture Contract

Experiment ID: <...>
Source checkpoint/ONNX/production Engine hashes: <...>
Capture debug Engine hash and builder log: <...>
Device/TRT/CUDA: <...>

## Three boundaries
- Study: e.g. post-QK-Norm/RoPE Q,K and unnormalized V -> SDPA O
- Capture: exact ONNX node IDs / debug TRT tensor IDs
- Candidate Plugin: exact subgraph to replace (can differ from study)

## Runtime context
- Chunk / forward index / workload mode / draft, predicted, real:
- Cache history length, valid slot order, mask, actual K/V layout:
- Scale/causal/attention bias/position IDs:
- Tensor names and shapes/dtypes/strides/contiguity: Q / K / V / O:

## Verification
- Debug Engine vs production full-model output: metric, tolerance, result
- Independent replay vs saved O: metric, tolerance, NaN/Inf count
- Observed shapes/profile coverage and excluded cases
- Artifact files and hashes

## Paged KV contract (complete when applicable)
- Physical pool: shape / strides / dtype / device / owner / allocation lifetime:
- K/V planes and layer/batch layout:
- Page size / page-ID namespace / ownership:
- Page table: shape / valid prefix / padding rule / sequence length:
- Logical history order and physical slot order reference:
- Current-token staging: shape / capacity / reserved slots:
- History then current ordering and attention-mask semantics:
- Commit type (predicted or real) / clear-pred / eviction / page reuse / reset:
- State assertions and independent logical-KV reference artifact:
