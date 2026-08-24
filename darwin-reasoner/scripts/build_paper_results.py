#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_rows(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("raw_results.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["run_path"] = str(path.parent)
                rows.append(row)
        except Exception as exc:
            print(f"WARNING: skipped {path}: {exc}")
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, seed: int = 0, draws: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ordered = df.sort_values(["avg_tokens", "accuracy"], ascending=[True, False])
    best = -float("inf")
    keep = []
    for idx, row in ordered.iterrows():
        if row["accuracy"] > best:
            keep.append(idx)
            best = row["accuracy"]
    return ordered.loc[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="paper_results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_rows(Path(args.runs))
    if df.empty:
        print("No raw_results.jsonl files found.")
        return

    for col in ["reward", "tokens", "latency_s"]:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "method" not in df:
        df["method"] = "unknown"

    groups = []
    for method, sub in df.groupby("method"):
        lo, hi = bootstrap_ci(sub["reward"].to_numpy())
        groups.append(
            {
                "method": method,
                "n": len(sub),
                "accuracy": sub["reward"].mean(),
                "accuracy_ci_low": lo,
                "accuracy_ci_high": hi,
                "avg_tokens": sub["tokens"].mean(),
                "avg_latency_s": sub["latency_s"].mean(),
            }
        )
    summary = pd.DataFrame(groups).sort_values("accuracy", ascending=False)
    summary.to_csv(out / "table_main.csv", index=False)
    pareto_frontier(summary).to_csv(out / "compute_frontier.csv", index=False)

    latex = summary.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
    (out / "table_main.tex").write_text(latex, encoding="utf-8")

    # Matched state/architecture diagnostics for world-model / counterfactual experiments.
    if {"state_id", "architecture_name"}.issubset(df.columns):
        matched = (
            df.groupby(["state_id", "architecture_name"], as_index=False)
            .agg(reward=("reward", "mean"), tokens=("tokens", "mean"), latency_s=("latency_s", "mean"))
        )
        matched.to_csv(out / "counterfactual_matched_means.csv", index=False)

    print(f"Wrote paper-ready summaries to {out}")


if __name__ == "__main__":
    main()
