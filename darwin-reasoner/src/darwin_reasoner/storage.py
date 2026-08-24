from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class RunStore:
    def __init__(self, root: str | Path, experiment_name: str, *, resume_dir: str | None = None) -> None:
        root = Path(root)
        if resume_dir:
            self.run_dir = Path(resume_dir)
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = root / f"{stamp}_{experiment_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.run_dir / "raw_results.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.completed_path = self.run_dir / "completed_keys.txt"
        self.completed = self._load_completed()

    def _load_completed(self) -> set[str]:
        if not self.completed_path.exists():
            return set()
        return {line.strip() for line in self.completed_path.read_text().splitlines() if line.strip()}

    def is_completed(self, key: str) -> bool:
        return key in self.completed

    def append(self, key: str, row: dict[str, Any]) -> None:
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        with self.completed_path.open("a", encoding="utf-8") as handle:
            handle.write(key + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.completed.add(key)

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        (self.run_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def write_text(self, name: str, text: str) -> None:
        (self.run_dir / name).write_text(text, encoding="utf-8")

    @staticmethod
    def git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unknown"

    @staticmethod
    def environment() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": RunStore.git_commit(),
        }
        try:
            import torch

            payload.update(
                {
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "gpus": [
                        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                    ],
                }
            )
        except Exception:
            pass
        return payload
