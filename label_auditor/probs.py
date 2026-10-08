"""Out-of-fold predicted probabilities: every example is scored by models that never saw its label."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10)),
    "forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
}


def out_of_fold(X: np.ndarray, y: np.ndarray, model: str, folds: int = 5, seed: int = 0) -> np.ndarray:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(MODELS[model](), X, y, cv=cv, method="predict_proba")


def all_models(X: np.ndarray, y: np.ndarray, models=tuple(MODELS), folds: int = 5) -> dict[str, np.ndarray]:
    return {m: out_of_fold(X, y, m, folds) for m in models}
