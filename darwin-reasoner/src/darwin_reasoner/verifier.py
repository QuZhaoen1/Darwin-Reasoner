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
        # A truncated generation can leave an empty capture ("FINAL:" or
        # \boxed{ } with nothing after it), so skip blank matches instead of
        # indexing into an empty splitlines() result.
        for match in reversed(list(pattern.finditer(text))):
            captured = [ln.strip() for ln in match.group(1).strip().splitlines() if ln.strip()]
            if captured:
                return captured[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


# Wrappers that carry presentation, not mathematical content. Applied to the
# prediction and the reference alike, so no comparison is loosened asymmetrically.
# Measured on 8 MATH-500 problems with Qwen3-8B under a chat template: the model
# answered 8/8 correctly and the un-normalized comparison scored 0/8, every miss
# caused by markdown emphasis or a \text{}/\boxed{}/\frac{} wrapper.
_STRIP_SPACING = re.compile(r"\\[!,;:]|\\quad|\\qquad")
_WRAPPERS = re.compile(r"\\(?:left|right|displaystyle|mathrm|mathbf)\b")
_TEXTISH = re.compile(r"\\(?:text|textbf|textit|mbox|boxed)\s*\{([^{}]*)\}")
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def normalize_answer(answer: str) -> str:
    answer = str(answer or "").strip()
    answer = _STRIP_SPACING.sub("", answer)
    answer = _WRAPPERS.sub("", answer)
    # \frac before \boxed: the brace-free capture in _TEXTISH cannot see through
    # a nested \frac{}{}, so \boxed{\frac{14}{3}} only unwraps once the fraction
    # has already collapsed to 14/3.
    for _ in range(3):
        new = _FRAC.sub(r"\1/\2", answer)
        new = _TEXTISH.sub(r"\1", new)
        if new == answer:
            break
        answer = new
    answer = answer.replace("\\pi", "pi").replace("\u03c0", "pi")
    answer = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|\\circ|\u00b0", "", answer)  # 90 == 90 degrees
    answer = re.sub(r"\*{1,2}|__", "", answer)  # markdown emphasis
    # Drop thousands separators only. Stripping every comma would fuse the
    # elements of a list answer ("-2, 1" -> "-21"), inventing mismatches; list
    # answers are compared elementwise in verify_exact instead.
    answer = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", answer)
    answer = answer.replace("$", "").replace("\\%", "%")
    # Whitespace is never load-bearing in these answers, and dropping it rather
    # than collapsing it makes "( 3, \frac{\pi}{2} )" and "(3, pi/2)" agree.
    answer = re.sub(r"\s+", "", answer)
    return answer.strip(".").lower()


def _is_bare_list(s: str) -> bool:
    """True for "1,-2" but not "(3,pi/2)" or "{1,2}" -- a comma inside brackets
    denotes position in a tuple or interval, where order carries meaning."""
    return "," in s and not re.search(r"[(){}\[\]]", s)


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

    # "the solutions are -2, 1" and "1,-2" are the same answer. Only applied when
    # both sides are bare top-level lists of equal length, so ordered structures
    # like (3,pi/2) -- where the comma sits inside brackets -- are untouched.
    if _is_bare_list(pred_norm) and _is_bare_list(ref_norm):
        pred_items = sorted(pred_norm.split(","))
        ref_items = sorted(ref_norm.split(","))
        if pred_items == ref_items:
            return VerificationResult(True, 1.0, predicted, "set_match")

    # Lightweight numeric tolerance for decimal answers.
    try:
        pred_float = float(pred_norm)
        ref_float = float(ref_norm)
        if math.isclose(pred_float, ref_float, rel_tol=1e-6, abs_tol=1e-8):
            return VerificationResult(True, 1.0, predicted, "numeric_match")
    except ValueError:
        pass

    return VerificationResult(False, 0.0, predicted, "mismatch")
