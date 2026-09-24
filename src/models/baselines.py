"""
Classical and Feature Baseline Models for Software Supply Chain Compromise Detection.

Implements:
1. PrevalenceBaseline: Trivial constant predictor based on training positive prevalence.
2. LogisticRegressionBaseline: L2-regularized logistic regression with balanced class weights and train-fitted scaler.
3. RandomForestBaseline: Balanced Random Forest ensemble.
4. XGBoostBaseline: Gradient boosted decision trees with scale_pos_weight = N_neg / N_pos.

Adheres strictly to the single primary imbalance mechanism rule per model (no sample_weight combination).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)


class PrevalenceBaseline:
    """Trivial baseline predicting constant training positive prevalence."""

    def __init__(self) -> None:
        self.prevalence: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrevalenceBaseline":
        y_arr = np.asarray(y, dtype=int)
        pos = int(np.sum(y_arr == 1))
        total = int(len(y_arr))
        self.prevalence = float(pos / total) if total > 0 else 0.0
        self.is_fitted = True
        logger.info(f"PrevalenceBaseline fitted with training prevalence: {self.prevalence:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        probs = np.full((n_samples, 2), [1.0 - self.prevalence, self.prevalence])
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= threshold).astype(int)

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"prevalence": self.prevalence, "model_type": "PrevalenceBaseline"}, f, indent=2)

    def load(self, file_path: Union[str, Path]) -> "PrevalenceBaseline":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.prevalence = float(data["prevalence"])
        self.is_fitted = True
        return self


class LogisticRegressionBaseline:
    """Logistic Regression baseline with train-only StandardScaler and class_weight='balanced'."""

    def __init__(self, random_state: int = 42, max_iter: int = 1000, C: float = 1.0) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self.pipeline: Pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=max_iter,
                    C=C,
                    solver="lbfgs",
                ),
            ),
        ])
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        logger.info("Fitting LogisticRegressionBaseline with class_weight='balanced'...")
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= threshold).astype(int)

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, file_path: Union[str, Path]) -> "LogisticRegressionBaseline":
        self.pipeline = joblib.load(file_path)
        self.is_fitted = True
        return self


class RandomForestBaseline:
    """Random Forest baseline with class_weight='balanced'."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 12,
        min_samples_split: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clf: RandomForestClassifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        logger.info("Fitting RandomForestBaseline with class_weight='balanced'...")
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= threshold).astype(int)

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)

    def load(self, file_path: Union[str, Path]) -> "RandomForestBaseline":
        self.clf = joblib.load(file_path)
        self.is_fitted = True
        return self


class XGBoostBaseline:
    """XGBoost baseline with scale_pos_weight = N_neg / N_pos."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed in the environment.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scale_pos_weight: float = 1.0
        self.clf: Optional[XGBClassifier] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        y_arr = np.asarray(y, dtype=int)
        pos = int(np.sum(y_arr == 1))
        neg = int(np.sum(y_arr == 0))
        self.scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

        logger.info(
            f"Fitting XGBoostBaseline with scale_pos_weight={self.scale_pos_weight:.2f} "
            f"(neg={neg}, pos={pos})..."
        )

        self.clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=self.n_jobs,
        )
        self.clf.fit(X, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model is not fitted.")
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= threshold).astype(int)

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.clf is not None:
            self.clf.save_model(str(path))

    def load(self, file_path: Union[str, Path]) -> "XGBoostBaseline":
        self.clf = XGBClassifier()
        self.clf.load_model(str(file_path))
        self.is_fitted = True
        return self
