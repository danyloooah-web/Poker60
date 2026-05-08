"""Calibrated classifiers saved alongside chunk training bundles (joblib unpickle support)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibratedClassifier:
    """
    Sigmoid (Platt) scaling on top of a fitted binary classifier.

    Lives in ``poker44`` so ``joblib`` artifacts unpickle from ``neurons/miner.py`` reliably
    (pickles must reference importable module paths, not ``__main__``).
    """

    def __init__(self, base_clf: Any, calibrator: LogisticRegression):
        self.base_clf = base_clf
        self.calibrator = calibrator
        self.classes_ = np.asarray(getattr(base_clf, "classes_", [0, 1]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.base_clf.decision_function(X)
        if scores.ndim > 1:
            scores = scores[:, 1]
        return self.calibrator.predict_proba(scores.reshape(-1, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
