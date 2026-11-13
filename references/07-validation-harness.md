# 07. Real-data single-operator validation (project harness integration)

## Role and ownership

The Skill is a navigator, **not** the numerical oracle. Use the project's existing
validation harness and frozen contract/manifest/data (or equivalent). Do not
duplicate, rewrite, or vendor a project-owned implementation unless its owner
requests it. This skill provides an **offline JSON
report auditor**, not a substitute for actually running that validator.

## Truth hierarchy

1. **Captured input + matching original TRT boundary output** (recorded, not necessarily directly proven from production internals).
2. **Independent mathematical reference** (FP32 or appropriate precision, correct masking, scale, positional/cache semantics).
3. **Candidate output on exact captured inputs**.
4. **Separate integration and full-model outputs**; local validation does not prove them.

The debug Engine may have a distinct SHA256 from production. Same layer count,
Tactic signatures and bitwise-equal *external* outputs improve confidence but do
**not prove** each internal tensor matches the production Engine. Distinguish
reported source checks from independently rerun checks.

## Portable fixture contract

- Semantic operator boundary (e.g., post-RoPE Q/K/V to pre-output-projection O);
  input roles, tensor shapes, **strides**, dtype, physical layout and alignment;
  scale, mask, causal/dropout, quantization/rounding, state/cache semantics.
- Reproducible source run, model/engine/build config hashes, layer mapping and
  execution mode, real head/query/KV lengths and cache state.
- Caller-owned output allocation; allowed mutation, stream ownership, workspace,
  aliasing rules. A fixture that was sliced and made contiguous is **not** proof
  of original strided layout support.
- For paged KV, also record pool shape/strides and ownership, page size and ID
  namespace, table shape/valid prefix/padding rule, effective lengths, staging
  capacity, and commit/clear/eviction/reset semantics. Keep private raw fixtures
  in their source project.
- Numerical acceptance frozen **before** candidate evaluation:
  `abs(candidate - trt) <= atol + rtol * abs(trt)` per element, or a clearly
  specified different criterion. A scalar max-abs versus `atol` alone is NOT a
  valid replacement for a mixed absolute/relative check.
- Bitwise checksum of frozen data, contract and reports. Do not overwrite
  frozen sources/results; candidate code and compiler versions must be pinned.

## Minimum candidate checks

- shape/dtype, finite outputs, input immutability **when contracted**;
- caller-provided output written completely, and canary/guard regions unchanged;
- repeated-run check **according to chosen determinism policy** (not every
  legal reduction is bitwise deterministic);
- Q/K/V or applicable input perturbations, recomputing independent reference;
- original TRT output comparison AND independent mathematical reference
  diagnostics. Record separate MAE, RMSE, max abs, cosine; include counts
  outside the per-element criterion, not only aggregate metrics.
- negative tests deliberately producing wrong output, partial writes, input
  mutation, guard corruption and cached expected output, confirming the
  validator rejects them.

These finite checks are **not memory-safety proofs**. Compute Sanitizer and
multi-stream concurrency need separately scoped tests if relevant.

For paged KV, check metadata, page bounds/ownership, logical order, and KV value
agreement separately. Exercise staging overwrite without commit, prediction
commit/clear, real commit, eviction, physical-page reuse, batch/layer isolation,
and reset against an independent logical contiguous reference. Advance from a
single backend call to one Plugin, a small multi-Plugin Engine, full Engine,
real repeated steps, and recorded chunks; each gate has its own limits. See
`10-paged-kv-contract.md` for the Engine ABI and staged checks.

## Interpreting reduced fixtures

A reduced fixture can support a mathematical or caller-behavior check only when
its construction preserves the tested semantics. It is not evidence of
production-shape performance, original Engine internals, Plugin integration, or
full-model behavior. If independent rows/heads are selected, retain every K/V
value required by those rows and document the argument. Layer names alone do
not identify matching work across modes or Engine builds; compare workload,
source mapping, shapes, and Engine identity.

## Use the offline report auditor

```bash
python helpers/evidence_audit.py \
  --cpu /path/to/example_cpu.json \
  --cuda /path/to/example_cuda.json \
  --capture /path/to/source_capture.json \
  --manifest /path/to/manifest.json --contract /path/to/contract.json \
  --candidate /path/to/candidate_example.py \
  --out /tmp/operator-report-audit.json
```

`report_checks_passed=true` means **consistent reported JSON claims**, not
"candidate independently verified", "performance improved", or "safe to deploy".
Optional `--manifest` verifies available files; `--capture` and `--contract` can be supplied at separately mounted paths, and `--candidate` verifies its reported source hash.
Missing safetensors are flagged; never infer their integrity from JSON alone.

## Gates

- Source capture stage: raw tensors and actual provenance checked **in the source project environment**;
  external equivalence and any internal debug perturbation limitations documented.
- Local numerical checkpoint: `validate.py` report + code hash + frozen fixture,
  independent reference, behavior/perturbation checks. No speed claim yet.
- Local performance checkpoint: separate *matched* benchmark report. Do not
  mark the workbench's aggregate `local` stage PASS before required timing
  evidence exists; log numerical-only state as `running` with attached evidence.

Next: `references/08-benchmark-protocol.md`.
