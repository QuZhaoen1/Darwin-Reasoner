from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from .config import ExperimentConfig
from .experiment import (
    collect_counterfactuals,
    mine_macros,
    run_baselines,
    run_darwin,
    train_world_model,
    validate_macros,
)
from .pipeline import run_pipeline
from .sweep import run_sweep

app = typer.Typer(no_args_is_help=True, help="DarwinReasoner experiment CLI")


@app.command("baselines")
def baselines(
    config: str = typer.Option(..., "--config", "-c"),
    run_dir: str | None = typer.Option(None),
) -> None:
    cfg = ExperimentConfig.load(config)
    path = run_baselines(cfg, run_dir=run_dir)
    print(f"[green]Baselines complete:[/green] {path}")


@app.command("collect")
def collect(
    config: str = typer.Option(..., "--config", "-c"),
    run_dir: str | None = typer.Option(None),
    world_model: str | None = typer.Option(None),
) -> None:
    cfg = ExperimentConfig.load(config)
    path = collect_counterfactuals(cfg, run_dir=run_dir, world_model_path=world_model)
    print(f"[green]Counterfactual collection complete:[/green] {path}")


@app.command("train-world-model")
def train_world(
    config: str = typer.Option(..., "--config", "-c"),
    data: str = typer.Option(..., "--data"),
    output: str = typer.Option(..., "--output"),
) -> None:
    cfg = ExperimentConfig.load(config)
    metrics = train_world_model(cfg, data_path=data, output_path=output)
    print(json.dumps(metrics, indent=2))


@app.command("run")
def run(
    config: str = typer.Option(..., "--config", "-c"),
    world_model: str = typer.Option(..., "--world-model"),
    run_dir: str | None = typer.Option(None),
    macros: str | None = typer.Option(None),
) -> None:
    cfg = ExperimentConfig.load(config)
    path = run_darwin(
        cfg,
        world_model_path=world_model,
        run_dir=run_dir,
        macros_path=macros,
    )
    print(f"[green]DarwinReasoner run complete:[/green] {path}")


@app.command("mine-macros")
def mine(
    config: str = typer.Option(..., "--config", "-c"),
    data: str = typer.Option(..., "--data"),
    output: str = typer.Option(..., "--output"),
) -> None:
    cfg = ExperimentConfig.load(config)
    macros = mine_macros(cfg, data_path=data, output_path=output)
    print(f"[green]Mined {len(macros)} macros.[/green]")


@app.command("validate-macros")
def validate_macro_cmd(
    config: str = typer.Option(..., "--config", "-c"),
    candidates: str = typer.Option(..., "--candidates"),
    output: str = typer.Option(..., "--output"),
    run_dir: str | None = typer.Option(None),
) -> None:
    cfg = ExperimentConfig.load(config)
    admitted = validate_macros(
        cfg, candidates_path=candidates, output_path=output, run_dir=run_dir
    )
    print(f"[green]Admitted {len(admitted)} macros.[/green]")


@app.command("pipeline")
def pipeline(config: str = typer.Option(..., "--config", "-c")) -> None:
    status = run_pipeline(config)
    print(json.dumps(status, indent=2))


@app.command("sweep")
def sweep(config: str = typer.Option(..., "--config", "-c")) -> None:
    result = run_sweep(config)
    print(json.dumps(result, indent=2))


@app.command("inspect")
def inspect_run(run_dir: str) -> None:
    path = Path(run_dir)
    for name in ["environment.json", "metrics.json", "manifest.json", "pipeline_status.json"]:
        target = path / name
        if target.exists():
            print(f"\n[bold]{name}[/bold]\n{target.read_text()}")


if __name__ == "__main__":
    app()
