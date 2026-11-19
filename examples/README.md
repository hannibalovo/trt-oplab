# Anonymized case studies

These short cases preserve reported observations and evidence limits without
shipping company paths, models, Engine files, tensors, raw traces or proprietary
performance totals. They are teaching examples, not general performance claims.

- [Engine lifecycle and first-request latency](engine-lifecycle-case-study.md) —
  distinguish storage/read/setup observations from warmed GPU throughput.
- [Paged KV Engine contract](paged-kv-contract-case-study.md) — staged correctness,
  multi-Plugin failure controls, cache lifecycle and unresolved latency spikes.
- [FFN plus KV Pack composite change](ffn-kv-pack-case-study.md) — avoid assigning
  a multi-factor replay result to one optimization.

See [Reference 15](../references/15-paged-kv-contract.md) for the detailed paged
contract and [Reference 16](../references/16-engine-startup-lifecycle.md) for
Engine initialization phases.
