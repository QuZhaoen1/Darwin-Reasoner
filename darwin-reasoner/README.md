# DarwinReasoner

**Counterfactual World Models for Self-Expanding Inference Architecture Search**

DarwinReasoner is a research harness for studying a stronger form of test-time scaling: instead of only deciding *how long* an LLM should think, it learns *which inference architecture should be executed under a compute budget*, predicts the consequences of alternative architectures before executing them, and expands its own typed reasoning grammar when reusable motifs survive matched validation.

> Status: research scaffold / experiment harness. **No SOTA claim is made by this repository.** The code is designed to make the full experimental logic reproducible before expensive GPU time is available.


## Zero-key quick start

The default pilot needs **no API keys and no pre-existing dataset**. It uses the public `Qwen/Qwen3-8B` model, downloads the public `HuggingFaceH4/MATH-500` benchmark automatically, and writes metrics locally instead of requiring W&B.

```bash
bash scripts/bootstrap_no_key.sh
bash scripts/smoke.sh
bash scripts/no_key_preflight.sh
```

Then launch the real-model pilot:

```bash
source .venv/bin/activate
python -m darwin_reasoner.cli collect \
  --config configs/pilot_8b.yaml \
  --run-dir runs/pilot_counterfactual
```

`HF_TOKEN` is only needed later for gated/private Hugging Face assets. `WANDB_API_KEY` is optional because tracking is disabled by default. See `docs/ZERO_KEY_START_CN.md`.

## Core research hypothesis

For the same intermediate reasoning state, different inference architectures can have sharply different marginal value. If we collect matched interventions, learn an uncertainty-aware world model over these architecture outcomes, and amortize successful patterns into an expanding grammar, then we should be able to improve the **accuracy-compute Pareto frontier** and transfer reasoning procedures to unseen tasks/models.

The intended loop is:

```text
same reasoning state
      |
      +--> architecture A --> reward / tokens / latency
      +--> architecture B --> reward / tokens / latency
      +--> architecture C --> reward / tokens / latency
                  |
                  v
      Counterfactual Architecture World Model
                  |
                  v
      Budget-conditioned architecture planner
                  |
                  v
      execute -> observe -> replan
                  |
                  v
      mine motifs -> validate macros -> expand grammar
                  |
                  v
      distill search experience into a cheap prior
```

## What is new in the intended paper

1. **Matched reasoning interventions.** From the same problem, reasoning prefix, model, budget family, and stochastic seed block, execute alternative inference architectures and measure paired outcome differences.
2. **Counterfactual architecture world model.** Predict success, token cost, latency, and epistemic uncertainty for a candidate reasoning DAG before paying its full execution cost.
3. **Active counterfactual design.** Spend GPU time on interventions with high predicted utility or high information value instead of uniformly evaluating every candidate.
4. **Self-expanding typed graph grammar.** Mine successful operator motifs and admit a new macro only after paired non-inferiority / cost validation. The search space itself therefore changes with experience.
5. **Budget-conditioned Pareto compilation.** Construct a query-specific reasoning architecture under token/latency constraints, with uncertainty-aware risk control and online replanning.
6. **Counterfactual-regret distillation.** Amortize expensive search into an architecture prior while retaining online search for OOD tasks.
7. **Experience-time scaling.** Test whether accumulated architecture experience reduces future test-time search cost at equal or better accuracy.

See `docs/IDEA_AND_NOVELTY_CN.md` and `docs/EXPERIMENT_PROTOCOL_CN.md` for the full Chinese research plan.

## Repository layout

```text
src/darwin_reasoner/
  backends/          mock + vLLM inference
  world_model/       uncertainty-aware architecture outcome model
  search/            candidate generation, Pareto planning, active design
  evolution/         motif mining, macro admission, architecture prior
  evaluation/        paired intervention estimators
  executor.py        typed DAG execution
  counterfactual.py  matched rollout collection
  experiment.py      experiment stages
  pipeline.py        pilot gate / resume logic
  sweep.py           full benchmark/model/budget sweep

configs/             smoke, pilot, 8xA100, pipeline and sweep configs
scripts/             setup, preflight, smoke, 8xA100, Slurm, result builder
data/                only tiny smoke data is committed
runs/                generated locally; gitignored
paper_results/       generated tables; gitignored except placeholder
```

## Zero-cost smoke test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
bash scripts/smoke.sh
```

The smoke test uses a mock backend and must finish without a GPU. It exercises baseline execution, matched rollouts, world-model training, motif mining, DarwinReasoner inference, resume-safe storage, and paper-table generation.

## GPU-server installation

On the borrowed A100 machine:

```bash
git clone git@github.com:YOUR_USERNAME/darwin-reasoner.git
cd darwin-reasoner
cp .env.example .env
# edit .env locally on the server; never commit it
DARWIN_GPU_INSTALL=1 bash scripts/setup.sh
source .venv/bin/activate
bash scripts/check_server.sh
```

Then run a tiny real-model pilot before the full job:

```bash
python -m darwin_reasoner.cli collect \
  --config configs/pilot_8b.yaml \
  --run-dir runs/pilot_counterfactual
```

After the pilot data are collected:

```bash
python -m darwin_reasoner.cli train-world-model \
  --config configs/pilot_8b.yaml \
  --data runs/pilot_counterfactual/raw_results.jsonl \
  --output runs/pilot_world_model/world_model.pt
```

The automated pipeline can then be launched with:

```bash
bash scripts/run_iclr.sh configs/pipeline_iclr.yaml configs/full_sweep.yaml
```

The pipeline is resume-safe. Re-running the same stage directory skips completed keys.

## 8xA100 strategy

`configs/a100_8gpu_32b.yaml` is a conservative single-process configuration using vLLM tensor parallelism across eight GPUs. For small/medium models, higher throughput can be obtained by sharding benchmark cells or seeds into independent processes with disjoint `CUDA_VISIBLE_DEVICES` groups.

vLLM currently supports tensor/pipeline parallel inference and data-parallel deployment; the repository deliberately keeps the orchestration layer backend-agnostic so server-specific parallel layouts can be changed without changing the scientific protocol.

## Dataset format

All benchmarks are converted to JSONL:

```json
{"problem_id":"...","prompt":"...","answer":"...","domain":"math","metadata":{}}
```

A generic Hugging Face converter is included:

```bash
python scripts/prepare_hf_dataset.py \
  --dataset DATASET_NAME \
  --split test \
  --prompt-field problem \
  --answer-field answer \
  --domain math \
  --output data/my_benchmark.jsonl
```

Field names vary by dataset, so inspect each dataset card before conversion.

## SOTA evaluation rules

Do **not** compare against a paper by re-implementing a weakened approximation and calling it the baseline. For AutoMR, MetaFlow, HyEvo, BayesFlow and other current methods, use official code/checkpoints when available, pin exact commits, and record their configs. `docs/THIRD_PARTY_BASELINES.md` defines the integration protocol.

All main comparisons should use, where technically possible:

- identical base reasoner;
- identical benchmark split;
- identical answer verifier;
- matched token/FLOP budgets;
- at least three seeds for stochastic methods;
- separate accuracy, token, latency and search-overhead reporting;
- paired bootstrap tests on the same problem set.

## Security

Never commit `.env`, Hugging Face tokens, W&B API keys, SSH private keys, checkpoints, model weights, or large rollout files. The `.gitignore` is already configured for these paths.

Generated code execution for code benchmarks should run inside a separate hardened sandbox/container. This repository does not enable arbitrary-code execution by default.

## Key adjacent work to reproduce/compare

The novelty audit should be refreshed immediately before submission. As of August 2026, particularly relevant work includes:

- AutoMR: query-aware DAG meta-reasoning search: https://arxiv.org/abs/2510.04116
- BayesFlow: Bayesian posterior sampling for workflow generation: https://arxiv.org/abs/2601.22305
- HyEvo: self-evolving hybrid LLM/code workflows: https://arxiv.org/abs/2603.19639
- MetaFlow: zero-shot workflow synthesis conditioned on a task/operator set: https://arxiv.org/abs/2606.30704
- FlowEvo: workflow-to-skill compilation and persistent skill reuse: https://arxiv.org/abs/2607.21596
- UniScale: adaptive unified inference scaling: https://arxiv.org/abs/2605.30898
- Test-Time Scaling in Reasoning LLMs: inference-regime taxonomy: https://arxiv.org/abs/2608.04001

## License

MIT. Third-party benchmark/model licenses remain their own.

## 8×A100 parameter training path

The repository now includes a conservative **Qwen3-8B BF16 LoRA + 8-process DDP** training path for the inference-architecture policy. It uses one A100 per process and launches with PyTorch `torchrun`.

```bash
bash scripts/setup_a100.sh
source .venv/bin/activate
bash scripts/check_8xa100.sh
bash scripts/train_policy_8xa100.sh
```

Before policy training, convert matched counterfactual outcomes into SFT examples:

```bash
python scripts/build_policy_dataset.py \
  --input runs/pilot_counterfactual/raw_results.jsonl
```

See `docs/8XA100_TRAINING_CN.md` and `docs/GITHUB_UPLOAD_8XA100_CN.md`.

For the complete 8-GPU pilot-to-training chain:

```bash
bash scripts/pilot_to_train_8xa100.sh
```

The rollout stage launches eight independent single-GPU vLLM workers on eight dataset shards; the policy-training stage switches to 8-process DDP via `torchrun`.
