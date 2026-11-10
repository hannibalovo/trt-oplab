# Optimization Decision Record

- Experiment ID:
- Hypothesis and targeted workload:
- Status: accept / iterate / reject / blocked
- Design decision (small vs large Plugin boundary, or no Plugin):

## Claims and supporting evidence

| Claim | Measured scope & unit | Baseline | Candidate | Exact evidence path |
|---|---|---:|---:|---|
| Correctness | max abs / rel / other | | | |
| Kernel-only latency | ms, same shape/stream | | | |
| Full operator call | ms, including conversions | | | |
| Integrated engine | ms, same input | | | |
| End-to-end replay | ms/chunk generate+commit | | | |

## Build and graph changes

- TRT version / profile / device / precision:
- Plugin boundary and runtime registration:
- Target tactic/layer changes:
- Other fusion/tactic/reformat changes:
- Unexplained deltas:

## Guardrails and decision

- Real workload and KV states covered:
- Numeric tolerance predeclared?
- Multi-chunk state/commit passed?
- Reproduction commands and immutable artifacts:
- Open risks / next falsifiable experiment:
