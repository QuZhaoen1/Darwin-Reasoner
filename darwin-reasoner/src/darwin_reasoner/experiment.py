from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .backends import build_backend
from .baselines import run_self_consistency
from .config import ExperimentConfig
from .counterfactual import CounterfactualCollector
from .data import load_jsonl
from .dsl import chain_architecture, template_library
from .evolution.motif_miner import MotifMiner
from .executor import ArchitectureExecutor
from .schema import Budget, OperatorKind, Outcome, ReasoningArchitecture, ReasoningState
from .search import ActiveCounterfactualDesigner, ArchitectureGenerator, BudgetConditionedPlanner
from .storage import RunStore
from .tracking import Tracker
from .world_model import EnsembleWorldModel
from .world_model.model import load_examples_from_jsonl


def _outcome_row(
    outcome: Outcome,
    *,
    state: ReasoningState,
    architecture: ReasoningArchitecture,
    budget: Budget,
    method: str,
) -> dict:
    return {
        "problem_id": outcome.problem_id,
        "state_id": outcome.state_id,
        "state_text": f"{state.problem_prompt}\n{state.transcript}",
        "method": method,
        "architecture_id": outcome.architecture_id,
        "architecture_name": outcome.architecture_name,
        "architecture": architecture.canonical_dict(),
        "operator_sequence": architecture.operator_sequence,
        "budget": budget.model_dump(),
        "reward": outcome.reward,
        "correct": outcome.correct,
        "final_answer": outcome.final_answer,
        "tokens": outcome.tokens,
        "latency_s": outcome.latency_s,
        "calls": outcome.calls,
        "seed": outcome.seed,
        "trace": outcome.trace,
        "metadata": outcome.metadata,
    }


def _write_metrics(store: RunStore, rows: list[dict]) -> dict:
    if not rows:
        metrics = {
            "n": 0.0,
            "accuracy": 0.0,
            "avg_tokens": 0.0,
            "avg_latency_s": 0.0,
            "by_method": {},
            "best_method_accuracy": 0.0,
        }
    else:
        by_method: dict[str, dict[str, float]] = {}
        methods = sorted({str(r.get("method", "unknown")) for r in rows})
        for method in methods:
            subset = [r for r in rows if str(r.get("method", "unknown")) == method]
            by_method[method] = {
                "n": float(len(subset)),
                "accuracy": float(np.mean([float(r["reward"]) for r in subset])),
                "avg_tokens": float(np.mean([float(r.get("tokens", 0.0)) for r in subset])),
                "avg_latency_s": float(np.mean([float(r.get("latency_s", 0.0)) for r in subset])),
            }
        metrics = {
            "n": float(len(rows)),
            "accuracy": float(np.mean([float(r["reward"]) for r in rows])),
            "avg_tokens": float(np.mean([float(r.get("tokens", 0.0)) for r in rows])),
            "avg_latency_s": float(np.mean([float(r.get("latency_s", 0.0)) for r in rows])),
            "by_method": by_method,
            "best_method_accuracy": max(v["accuracy"] for v in by_method.values()),
        }
    store.write_json("metrics.json", metrics)
    return metrics


def _new_store(config: ExperimentConfig, run_dir: str | None) -> RunStore:
    if run_dir:
        path = Path(run_dir)
        return RunStore(path.parent, config.experiment_name, resume_dir=str(path))
    return RunStore(config.output_dir, config.experiment_name)


def _prepare(config: ExperimentConfig, run_dir: str | None):
    store = _new_store(config, run_dir)
    store.write_json("environment.json", RunStore.environment())
    store.write_text("config.resolved.json", json.dumps(config.model_dump(), indent=2))
    backend = build_backend(config.backend)
    executor = ArchitectureExecutor(backend, config.backend)
    problems = load_jsonl(config.dataset_path, limit=config.limit)
    return store, executor, problems


def run_baselines(config: ExperimentConfig, *, run_dir: str | None = None) -> Path:
    store, executor, problems = _prepare(config, run_dir)
    templates = template_library()
    allowed = set(config.extra.get("baseline_names", ["cot", "verify", "critical_repair"]))
    templates = [a for a in templates if a.name in allowed]
    sc_samples = int(config.extra.get("self_consistency_samples", 0))
    rows: list[dict] = []

    with Tracker(config.tracking, config.experiment_name, config.model_dump()) as tracker:
        for p_index, problem in enumerate(problems):
            state = ReasoningState(problem_id=problem.problem_id, problem_prompt=problem.prompt)
            for architecture in templates:
                key = f"baseline|{problem.problem_id}|{architecture.architecture_id}|{config.seed}"
                if store.is_completed(key):
                    continue
                outcome = executor.execute(
                    problem,
                    architecture,
                    config.budget,
                    seed=config.seed + p_index,
                    initial_state=state,
                )
                row = _outcome_row(
                    outcome,
                    state=state,
                    architecture=architecture,
                    budget=config.budget,
                    method=architecture.name,
                )
                store.append(key, row)
                rows.append(row)
                tracker.log({f"baseline/{architecture.name}/reward": outcome.reward})

            if sc_samples > 1:
                sc_arch = chain_architecture("self_consistency", [OperatorKind.SOLVE])
                key = f"baseline|{problem.problem_id}|self_consistency_{sc_samples}|{config.seed}"
                if not store.is_completed(key):
                    outcome = run_self_consistency(
                        executor,
                        problem,
                        state,
                        config.budget,
                        samples=sc_samples,
                        seed=config.seed + p_index,
                    )
                    row = _outcome_row(
                        outcome,
                        state=state,
                        architecture=sc_arch,
                        budget=config.budget,
                        method=f"self_consistency_{sc_samples}",
                    )
                    store.append(key, row)
                    rows.append(row)
                    tracker.log({f"baseline/self_consistency_{sc_samples}/reward": outcome.reward})

    # Include previous rows when resuming.
    all_rows = _read_jsonl_if_exists(store.results_path)
    _write_metrics(store, all_rows)
    return store.run_dir


def collect_counterfactuals(
    config: ExperimentConfig,
    *,
    run_dir: str | None = None,
    world_model_path: str | None = None,
) -> Path:
    store, executor, problems = _prepare(config, run_dir)
    generator = ArchitectureGenerator(seed=config.seed)
    world_model = _make_world_model(config, world_model_path)
    designer = ActiveCounterfactualDesigner(token_penalty=config.search.token_penalty)
    collector = CounterfactualCollector(executor, config.counterfactual)
    rows: list[dict] = []

    with Tracker(config.tracking, config.experiment_name, config.model_dump()) as tracker:
        for p_index, problem in enumerate(problems):
            state = ReasoningState(problem_id=problem.problem_id, problem_prompt=problem.prompt)
            candidates = generator.generate(
                state,
                num_candidates=config.search.num_candidates,
                mutation_rounds=config.search.mutation_rounds,
                max_nodes=config.budget.max_nodes,
            )
            predictions = world_model.predict_many(state, candidates, config.budget)
            if config.search.use_active_design:
                selected = [
                    c.architecture
                    for c in designer.select(
                        candidates,
                        predictions,
                        k=min(config.search.execute_top_k, len(candidates)),
                    )
                ]
            else:
                selected = candidates[: config.search.execute_top_k]

            batch = collector.collect(
                problem,
                state,
                selected,
                config.budget,
                base_seed=config.seed + p_index * 1_000_000,
            )
            architecture_by_id = {a.architecture_id: a for a in selected}
            for outcome in batch.outcomes:
                key = (
                    f"cf|{problem.problem_id}|{outcome.state_id}|{outcome.architecture_id}|"
                    f"{outcome.metadata['counterfactual_repeat']}|{outcome.seed}"
                )
                if store.is_completed(key):
                    continue
                architecture = architecture_by_id[outcome.architecture_id]
                row = _outcome_row(
                    outcome,
                    state=state,
                    architecture=architecture,
                    budget=config.budget,
                    method="counterfactual",
                )
                store.append(key, row)
                rows.append(row)
                tracker.log({"counterfactual/reward": outcome.reward, "counterfactual/tokens": outcome.tokens})

    all_rows = _read_jsonl_if_exists(store.results_path)
    _write_metrics(store, all_rows)
    return store.run_dir


def train_world_model(
    config: ExperimentConfig,
    *,
    data_path: str,
    output_path: str,
) -> dict[str, float]:
    examples = load_examples_from_jsonl(data_path, config.budget)
    model = EnsembleWorldModel(
        text_features=config.world_model.text_features,
        hidden_dim=config.world_model.hidden_dim,
        ensemble_size=config.world_model.ensemble_size,
        device=config.world_model.device,
    )
    metrics = model.fit(
        examples,
        epochs=config.world_model.epochs,
        batch_size=config.world_model.batch_size,
        learning_rate=config.world_model.learning_rate,
        seed=config.seed,
    )
    model.save(output_path)
    Path(output_path).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def run_darwin(
    config: ExperimentConfig,
    *,
    world_model_path: str,
    run_dir: str | None = None,
    macros_path: str | None = None,
) -> Path:
    store, executor, problems = _prepare(config, run_dir)
    generator = ArchitectureGenerator(seed=config.seed)
    if macros_path and Path(macros_path).exists():
        macros = json.loads(Path(macros_path).read_text())
        for macro in macros:
            generator.register_macro(macro["name"], macro["compiled_prompt"])

    world_model = _make_world_model(config, world_model_path)
    planner = BudgetConditionedPlanner(config.search)
    online_rounds = int(config.extra.get("online_rounds", 1))

    with Tracker(config.tracking, config.experiment_name, config.model_dump()) as tracker:
        for p_index, problem in enumerate(problems):
            key = f"darwin|{problem.problem_id}|{config.seed}|{config.budget.max_tokens}"
            if store.is_completed(key):
                continue
            state = ReasoningState(problem_id=problem.problem_id, problem_prompt=problem.prompt)
            remaining = config.budget.max_tokens
            all_trace: list[dict] = []
            total_tokens = 0
            total_latency = 0.0
            total_calls = 0
            chosen_architectures: list[str] = []
            final_outcome: Outcome | None = None

            for round_index in range(max(1, online_rounds)):
                candidates = generator.generate(
                    state,
                    num_candidates=config.search.num_candidates,
                    mutation_rounds=config.search.mutation_rounds,
                    max_nodes=config.budget.max_nodes,
                )
                # Split remaining compute across remaining replanning rounds.
                rounds_left = max(1, online_rounds - round_index)
                round_budget = config.budget.model_copy(
                    update={"max_tokens": max(128, remaining // rounds_left)}
                )
                predictions = world_model.predict_many(state, candidates, round_budget)
                ranked = planner.rank(candidates, predictions)
                selection_mode = str(config.extra.get("selection_mode", "planner"))
                if selection_mode == "random":
                    rng = np.random.default_rng(config.seed + p_index * 1000 + round_index)
                    choice = ranked[int(rng.integers(0, len(ranked)))]
                else:
                    choice = ranked[0]
                outcome = executor.execute(
                    problem,
                    choice.architecture,
                    round_budget,
                    seed=config.seed + p_index * 1000 + round_index,
                    initial_state=state,
                )
                chosen_architectures.append(choice.architecture.name)
                all_trace.extend(outcome.trace)
                total_tokens += outcome.tokens
                total_latency += outcome.latency_s
                total_calls += outcome.calls
                remaining = max(0, config.budget.max_tokens - total_tokens)
                final_outcome = outcome

                # Replanning receives the actual generated transcript, but never the ground-truth answer.
                transcript = "\n\n".join(item["text"] for item in outcome.trace)
                state = ReasoningState(
                    problem_id=problem.problem_id,
                    problem_prompt=problem.prompt,
                    transcript=(state.transcript + "\n\n" + transcript).strip(),
                    step=state.step + len(outcome.trace),
                    used_tokens=total_tokens,
                    used_calls=total_calls,
                    uncertainty=choice.prediction.reward_std,
                    metadata={"last_architecture": choice.architecture.name},
                )
                if remaining < 128 or outcome.calls == 0:
                    break
                # Stop early when the world model is confident and predicted gain from more search is low.
                if (
                    choice.prediction.reward_mean >= float(config.extra.get("stop_reward", 0.95))
                    and choice.prediction.reward_std <= config.search.uncertainty_threshold
                ):
                    break

            assert final_outcome is not None
            final_outcome.tokens = total_tokens
            final_outcome.latency_s = total_latency
            final_outcome.calls = total_calls
            final_outcome.trace = all_trace
            final_outcome.metadata["chosen_architectures"] = chosen_architectures
            # Reference answer is used only by executor's evaluator, not by architecture selection.
            row = _outcome_row(
                final_outcome,
                state=ReasoningState(problem_id=problem.problem_id, problem_prompt=problem.prompt),
                architecture=ReasoningArchitecture.model_validate(final_outcome.metadata["architecture"]),
                budget=config.budget,
                method="DarwinReasoner",
            )
            row["chosen_architectures"] = chosen_architectures
            store.append(key, row)
            tracker.log({"darwin/reward": final_outcome.reward, "darwin/tokens": total_tokens})

    all_rows = _read_jsonl_if_exists(store.results_path)
    _write_metrics(store, all_rows)
    return store.run_dir


def mine_macros(config: ExperimentConfig, *, data_path: str, output_path: str) -> list[dict]:
    rows = _read_jsonl_if_exists(Path(data_path))
    miner = MotifMiner(
        min_support=config.evolution.min_support,
        min_success_lift=config.evolution.min_success_lift,
        motif_min_len=config.evolution.motif_min_len,
        motif_max_len=config.evolution.motif_max_len,
        max_macros=config.evolution.max_macros,
    )
    macros = [m.__dict__ for m in miner.mine(rows)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(macros, indent=2), encoding="utf-8")
    return macros


def _make_world_model(config: ExperimentConfig, path: str | None) -> EnsembleWorldModel:
    if path and Path(path).exists():
        return EnsembleWorldModel.load(path, device=config.world_model.device)
    return EnsembleWorldModel(
        text_features=config.world_model.text_features,
        hidden_dim=config.world_model.hidden_dim,
        ensemble_size=config.world_model.ensemble_size,
        device=config.world_model.device,
    )


def _read_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_macros(
    config: ExperimentConfig,
    *,
    candidates_path: str,
    output_path: str,
    run_dir: str | None = None,
) -> list[dict]:
    """Run matched macro-vs-expanded interventions and admit only validated macros."""
    from .evolution.admission import evaluate_macro_admission
    from .schema import OperatorKind, ReasoningNode, ReasoningEdge

    store, executor, problems = _prepare(config, run_dir)
    candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    max_candidates = int(config.extra.get("macro_validation_candidates", 8))
    max_problems = int(config.extra.get("macro_validation_problems", min(50, len(problems))))
    noninferiority_margin = float(config.extra.get("macro_noninferiority_margin", 0.01))
    min_token_saving = float(config.extra.get("macro_min_token_saving", 0.05))
    admitted: list[dict] = []
    validation_rows: list[dict] = []

    for macro_index, macro in enumerate(candidates[:max_candidates]):
        operators = [OperatorKind(op) for op in macro["operators"]]
        expanded = chain_architecture(f"expanded::{macro['name']}", operators)
        macro_arch = ReasoningArchitecture(
            name=f"macro::{macro['name']}",
            nodes=[
                ReasoningNode(
                    node_id="n0",
                    operator=OperatorKind.MACRO,
                    params={"macro_name": macro["name"], "compiled_prompt": macro["compiled_prompt"]},
                )
            ],
            edges=[],
            metadata={"candidate_macro": macro["name"]},
        )

        macro_rewards: list[float] = []
        expanded_rewards: list[float] = []
        macro_tokens: list[float] = []
        expanded_tokens: list[float] = []
        for p_index, problem in enumerate(problems[:max_problems]):
            state = ReasoningState(problem_id=problem.problem_id, problem_prompt=problem.prompt)
            seed = config.seed + macro_index * 1_000_000 + p_index * 1000
            pair_key = f"macroval|{macro['name']}|{problem.problem_id}|{seed}"
            if store.is_completed(pair_key):
                # Resume reconstruction is handled from the result file after the loop.
                continue
            out_macro = executor.execute(problem, macro_arch, config.budget, seed=seed, initial_state=state)
            out_expanded = executor.execute(problem, expanded, config.budget, seed=seed, initial_state=state)
            row = {
                "macro_name": macro["name"],
                "problem_id": problem.problem_id,
                "state_id": state.state_id,
                "seed": seed,
                "macro_reward": out_macro.reward,
                "expanded_reward": out_expanded.reward,
                "macro_tokens": out_macro.tokens,
                "expanded_tokens": out_expanded.tokens,
                "macro_latency_s": out_macro.latency_s,
                "expanded_latency_s": out_expanded.latency_s,
            }
            store.append(pair_key, row)
            validation_rows.append(row)

        # Load all current/resumed pairs for this macro from disk.
        all_rows = _read_jsonl_if_exists(store.results_path)
        macro_rows = [r for r in all_rows if r.get("macro_name") == macro["name"]]
        macro_rewards = [float(r["macro_reward"]) for r in macro_rows]
        expanded_rewards = [float(r["expanded_reward"]) for r in macro_rows]
        macro_tokens = [float(r["macro_tokens"]) for r in macro_rows]
        expanded_tokens = [float(r["expanded_tokens"]) for r in macro_rows]
        result = evaluate_macro_admission(
            macro_rewards,
            expanded_rewards,
            macro_tokens,
            expanded_tokens,
            noninferiority_margin=noninferiority_margin,
            min_token_saving=min_token_saving,
            seed=config.seed + macro_index,
        )
        record = dict(macro)
        record["admission"] = result.__dict__
        if result.admitted:
            admitted.append(record)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(admitted, indent=2), encoding="utf-8")
    store.write_json(
        "macro_validation_summary.json",
        {"candidate_count": min(len(candidates), max_candidates), "admitted_count": len(admitted)},
    )
    return admitted
