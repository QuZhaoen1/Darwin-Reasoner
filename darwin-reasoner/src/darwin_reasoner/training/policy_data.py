from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


SYSTEM_PROMPT = (
    "You are the inference-architecture compiler in DarwinReasoner. "
    "Given a problem, a reasoning state, and a compute budget, output exactly one "
    "reasoning architecture as compact JSON. Do not solve the task itself."
)


@dataclass
class PolicyExample:
    prompt: str
    target: str
    metadata: dict[str, Any]


def load_policy_examples(path: str | Path) -> list[PolicyExample]:
    examples: list[PolicyExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                PolicyExample(
                    prompt=str(row["prompt"]),
                    target=str(row["target"]),
                    metadata=dict(row.get("metadata", {})),
                )
            )
    if not examples:
        raise ValueError(f"No policy examples found in {path}")
    return examples


class ArchitecturePolicyDataset(Dataset):
    """Causal-LM SFT data with the prompt portion masked from the loss."""

    def __init__(self, path: str | Path, tokenizer: Any, max_seq_len: int = 1024) -> None:
        self.examples = load_policy_examples(path)
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        if self.max_seq_len < 256:
            raise ValueError("max_seq_len must be >= 256")

    def __len__(self) -> int:
        return len(self.examples)

    def _chat_prompt(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"SYSTEM: {SYSTEM_PROMPT}\nUSER: {user_text}\nASSISTANT:"

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        prompt_text = self._chat_prompt(example.prompt)
        eos = self.tokenizer.eos_token or ""
        target_text = example.target + eos

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]

        # Preserve the target. Long problem/state context is clipped from the left.
        max_prompt = max(32, self.max_seq_len - min(len(target_ids), self.max_seq_len // 2))
        prompt_ids = prompt_ids[-max_prompt:]
        remaining = self.max_seq_len - len(prompt_ids)
        target_ids = target_ids[:remaining]
        if not target_ids:
            raise RuntimeError("Target was truncated completely; increase max_seq_len")

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids.copy()
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CausalSFTCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_masks, labels = [], [], []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention_masks.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
