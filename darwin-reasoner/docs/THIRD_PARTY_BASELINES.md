# Third-party SOTA baseline protocol

DarwinReasoner should be compared against the strongest *official* implementations available at submission time. Do not vendor or copy third-party source into this repository without checking its license.

## Current adjacent methods to track

| Method | Main overlap | Source |
|---|---|---|
| AutoMR | query-aware DAG meta-reasoning search | https://arxiv.org/abs/2510.04116 |
| BayesFlow | Bayesian workflow posterior sampling | https://arxiv.org/abs/2601.22305 |
| HyEvo | self-evolving hybrid LLM/code workflows | https://arxiv.org/abs/2603.19639 |
| MetaFlow | learned zero-shot workflow generator | https://arxiv.org/abs/2606.30704 |
| FlowEvo | workflow-to-skill compilation and persistent skill reuse | https://arxiv.org/abs/2607.21596 |
| UniScale | adaptive unified inference scaling | https://arxiv.org/abs/2605.30898 |

## Integration rule

For each external baseline, create `third_party/<name>/` as a git submodule only after:

1. official code is public;
2. license permits the intended use;
3. an exact commit is selected;
4. a wrapper converts DarwinReasoner JSONL input to the baseline format;
5. an output adapter returns final answer, tokens/calls, latency and search overhead;
6. the exact config is written to `configs/baselines/<name>.yaml`.

The paper should identify official/unofficial implementations explicitly.

## Fairness contract

Use the same base reasoner whenever the method permits it. If a baseline depends on a special trained checkpoint, report a separate checkpoint-native comparison rather than pretending the base models are matched. Count all planner/meta-agent/model calls toward compute. Report architecture-search cost separately and also include it in total-cost plots.
