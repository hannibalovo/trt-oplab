# 02. Extraction boundary and capture contract

## Crucial distinction

1. **Study boundary**: the mathematical operator you want to replay, e.g.
   Q/K/V after Q/K norm and RoPE → attention output *before* out projection.
2. **Capture boundary**: the actual tensors available from a debug ONNX/TRT graph.
3. **Integration boundary**: what the eventual Plugin or rewritten native subgraph
   will replace. It may be larger/smaller than the study boundary.

Do not conflate them. `history_k/v` at the engine input is **not** the same as
per-block Q/K/V after projections, norm, RoPE, concat and layout transforms.

## Capturing from a compiled 30-layer Engine

- **ONNX additional outputs**: expose the exact graph values at the desired
  boundary, rebuild an isolated debug Engine, capture on matched real inputs.
- **TRT Debug Tensors**: requires build-time marked tensors or corresponding
  debug support; runtime can activate a debug listener when supported by the
  installed version. Not a magic switch for any arbitrary production `.engine`.
- **PyTorch hooks**: useful mathematical reference, *not* direct evidence of
  actual production TRT intermediate numerical values.

Making a tensor observable may disrupt fusion/tactic selection or rounding.
Therefore store both a) debug Engine reference output and b) final production
Engine comparison under the same full-model inputs. Allow declared tolerance;
do not claim bitwise equivalence without proof.

## Minimum capture record

`templates/capture_contract.md` provides a fillable checklist. It must record:

- source engine/ONNX weights hashes, extraction source node(s), time-step/chunk,
  mode (action/video), branch (draft/pred/real), layer number;
- Q/K/V/O shapes, dtype, physical stride, on-device contiguity, and **layout
  meaning** (e.g. [B,H,S,D] vs [B,S,H,D]); scale, mask/bias, causal convention;
- whether RoPE/norm has already been applied, and whether history KV is included;
- cache capacity, valid slot order, dynamic lengths, position/grid IDs;
- reference output boundary, numeric tolerance, out-of-scope conditions.

Record full tensors when they fit, or documented deterministic subsets only
when a partial replay is explicitly intended. Never label partial capture as
full-model correctness evidence.

## Exit criteria

1. Standalone reference is deterministic enough under repeated replay.
2. The independent candidate consumes *exactly* the same semantic inputs.
3. Captured output has a clear boundary (after SDPA / before output projection).
4. Sample provenance, build flags, device, and cache mode are fixed.
