from __future__ import annotations

import json
from pathlib import Path

from .schema import Problem


def load_jsonl(path: str | Path, *, limit: int | None = None) -> list[Problem]:
    problems: list[Problem] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            problems.append(Problem.model_validate(row))
            if limit is not None and len(problems) >= limit:
                break
    return problems


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
