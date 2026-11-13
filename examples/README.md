# Evidence examples

Raw Engine summaries, tensors, Engine binaries, hashes, and internal output
artifacts are **not bundled in this public workbench**. Run
`helpers/evidence_audit.py` against your own `example_cpu.json`,
`example_cuda.json`, `source_capture.json` and optional `manifest.json`
locally, writing each audit into the experiment's evidence folder.

The sample `templates/benchmark_report.json` is a placeholder, *not a measured
benchmark*. Project-specific Inspector summaries should remain in their source
project rather than being added to these portable examples.

The case studies show paired replay limits, startup attribution, and paged-KV
contract failures. They contain no raw tensors, proprietary parameters, paths,
or Engine artifacts; reported observations are not independent reproductions.

- [`ffn-kv-pack-case-study.md`](ffn-kv-pack-case-study.md)
- [`engine-lifecycle-case-study.md`](engine-lifecycle-case-study.md)
- [`paged-kv-contract-case-study.md`](paged-kv-contract-case-study.md)
