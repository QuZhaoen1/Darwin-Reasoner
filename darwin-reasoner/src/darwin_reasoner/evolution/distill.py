from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression


class ArchitecturePrior:
    """Amortize expensive architecture search into a cheap family prior.

    Labels are architecture families (e.g. cot, verify, critical_repair). At test time the prior can
    seed the candidate generator; it does not replace online search, so OOD tasks can still invent
    new programs.
    """

    def __init__(self, n_features: int = 4096) -> None:
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.fitted = False

    def fit(self, state_texts: list[str], labels: list[str]) -> None:
        if len(set(labels)) < 2:
            raise ValueError("ArchitecturePrior needs at least two architecture families")
        x = self.vectorizer.transform(state_texts)
        self.model.fit(x, labels)
        self.fitted = True

    def predict_proba(self, state_text: str) -> dict[str, float]:
        if not self.fitted:
            return {}
        x = self.vectorizer.transform([state_text])
        probs = self.model.predict_proba(x)[0]
        return {str(label): float(prob) for label, prob in zip(self.model.classes_, probs, strict=True)}

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "ArchitecturePrior":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError("Invalid ArchitecturePrior file")
        return obj
