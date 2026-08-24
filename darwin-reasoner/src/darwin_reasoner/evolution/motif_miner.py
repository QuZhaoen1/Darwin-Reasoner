from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class MinedMacro:
    name: str
    operators: tuple[str, ...]
    support: int
    success_rate: float
    baseline_rate: float
    success_lift: float
    compiled_prompt: str


class MotifMiner:
    """Mine reusable operator motifs from successful reasoning programs."""

    def __init__(
        self,
        *,
        min_support: int = 8,
        min_success_lift: float = 0.03,
        motif_min_len: int = 2,
        motif_max_len: int = 5,
        max_macros: int = 32,
    ) -> None:
        self.min_support = min_support
        self.min_success_lift = min_success_lift
        self.motif_min_len = motif_min_len
        self.motif_max_len = motif_max_len
        self.max_macros = max_macros

    def mine(self, rows: list[dict]) -> list[MinedMacro]:
        if not rows:
            return []
        global_rate = sum(float(r.get("reward", 0.0)) for r in rows) / len(rows)
        occurrence = Counter()
        reward_sum = defaultdict(float)

        for row in rows:
            seq = tuple(row.get("operator_sequence", []))
            reward = float(row.get("reward", 0.0))
            seen = set()
            for length in range(self.motif_min_len, self.motif_max_len + 1):
                for i in range(0, max(0, len(seq) - length + 1)):
                    motif = seq[i : i + length]
                    if motif in seen:
                        continue
                    seen.add(motif)
                    occurrence[motif] += 1
                    reward_sum[motif] += reward

        macros: list[MinedMacro] = []
        for motif, support in occurrence.items():
            if support < self.min_support:
                continue
            success_rate = reward_sum[motif] / support
            lift = success_rate - global_rate
            if lift < self.min_success_lift:
                continue
            name = "__".join(motif)
            prompt = (
                "Execute the following reusable reasoning routine in order: "
                + " -> ".join(motif)
                + ". Preserve only useful intermediate results and end with FINAL: <answer>."
            )
            macros.append(
                MinedMacro(
                    name=name,
                    operators=motif,
                    support=support,
                    success_rate=success_rate,
                    baseline_rate=global_rate,
                    success_lift=lift,
                    compiled_prompt=prompt,
                )
            )
        macros.sort(key=lambda m: (m.success_lift, m.support), reverse=True)
        return macros[: self.max_macros]
