# Agent instructions for this workbench

1. Read `SKILL.md` first, then only the reference page needed for the current stage.
2. Work inside one experiment directory per candidate/shape/boundary. Do not silently
   overwrite baseline artifacts, change workload scope, or mix machines/builds.
3. Never mark a stage `pass` solely because code compiled or a microbenchmark improved.
4. Cite the exact artifact, command/config, device, profile and numeric comparison
   supporting a claim. Explicitly mark hypotheses and things not yet checked.
5. Use `helpers/layerinfo.py diff` as *structural evidence*, not proof of actual
   CUDA kernel launch identity or performance.
6. Avoid rewriting the production runtime during local extraction. Put debug
   instrumentation/candidate builds in isolated artifacts.
7. Respect hardware access windows and resource owners; no auto-execution of heavy
   GPU experiments or service restarts without user authorization.
8. Prefer references to existing NCU/KernelWiki skills over copied content.


## v0.3: company validation adapter

Use existing `operator_validation/` as the project-owned numerical executor.
Before a claim of local numerical PASS, inspect frozen contract, saved candidate
results and negative control tests; `helpers/evidence_audit.py` only audits
reported JSON claims. Never copy raw captured tensors into portable installers.
Before a speedup claim, require matching production-shaped benchmark evidence.
See references/07-validation-harness.md and references/08-benchmark-protocol.md.
