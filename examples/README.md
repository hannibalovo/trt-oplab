# Evidence examples

Raw company Engine summaries, tensors, Engine binaries, hashes and internal output
artifacts are **not bundled in a generally installable Skill**. Run
`helpers/evidence_audit.py` against your own `example_cpu.json`,
`example_cuda.json`, `source_capture.json` and optional `manifest.json`
locally, writing each audit into the experiment's evidence folder.

The sample `templates/benchmark_report.json` is a placeholder, *not a measured
benchmark*. The original v0.1 Action Engine summary was removed from the
installer to prevent accidental transmission of project-specific details.
