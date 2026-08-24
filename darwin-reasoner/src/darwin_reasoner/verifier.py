from __future__ import annotations

import math
import re
from dataclasses import dataclass


_FINAL_PATTERNS = [
    re.compile(r"FINAL\s*:\s*(.+)", re.IGNORECASE),
    re.compile(r"\\boxed\{([^{}]+)\}"),
]


def extract_final_answer(text: str) -> str:
    for pattern in _FINAL_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).strip().splitlines()[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace("$", "").replace(",", "")
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.strip(" .")
    return answer.lower()


@dataclass
class VerificationResult:
    correct: bool | None
    reward: float
    predicted_answer: str
    reason: str


def verify_exact(generation: str, reference: str | None) -> VerificationResult:
    predicted = extract_final_answer(generation)
    if reference is None:
        return VerificationResult(None, 0.0, predicted, "no_reference")

    pred_norm = normalize_answer(predicted)
    ref_norm = normalize_answer(reference)
    if pred_norm == ref_norm:
        return VerificationResult(True, 1.0, predicted, "exact_match")

    # Lightweight numeric tolerance for decimal answers.
    try:
        pred_float = float(pred_norm)
        ref_float = float(ref_norm)
        if math.isclose(pred_float, ref_float, rel_tol=1e-6, abs_tol=1e-8):
            return VerificationResult(True, 1.0, predicted, "numeric_match")
    except ValueError:
        pass

    return VerificationResult(False, 0.0, predicted, "mismatch")
