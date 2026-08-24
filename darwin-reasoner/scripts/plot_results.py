#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="paper_results")
    parser.add_argument("--out", default="paper_results/figures")
    args = parser.parse_args()

    root = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    main_table = root / "table_main.csv"
    if main_table.exists():
        df = pd.read_csv(main_table).sort_values("avg_tokens")
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for _, row in df.iterrows():
            ax.scatter(row["avg_tokens"], row["accuracy"], s=70)
            ax.annotate(str(row["method"]), (row["avg_tokens"], row["accuracy"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Average tokens")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy–Compute Frontier")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "accuracy_compute_frontier.pdf")
        fig.savefig(out / "accuracy_compute_frontier.png", dpi=220)
        plt.close(fig)

    matched = root / "counterfactual_matched_means.csv"
    if matched.exists():
        df = pd.read_csv(matched)
        if not df.empty:
            counts = df.groupby("architecture_name")["reward"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8.0, 4.8))
            ax.bar(range(len(counts)), counts.values)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha="right")
            ax.set_ylabel("Mean matched reward")
            ax.set_title("Architecture Outcome Profile")
            fig.tight_layout()
            fig.savefig(out / "architecture_outcome_profile.pdf")
            fig.savefig(out / "architecture_outcome_profile.png", dpi=220)
            plt.close(fig)

    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
