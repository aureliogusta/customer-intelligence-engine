"""
decision_service/training.py
=============================
Treina ChurnModel (GradientBoosting) e ExpansionModel (RandomForest).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from .models import ChurnModel, ExpansionModel


@dataclass
class TrainingReport:
    model_name:  str
    target:      str
    auc:         float
    avg_precision: float
    train_rows:  int
    test_rows:   int
    artifact_path: str


class ChurnTrainer:
    """
    Treina um GradientBoostingClassifier para predição de churn.

    Uso:
        config  = ChurnModel()
        trainer = ChurnTrainer(config)
        model, report = trainer.train(X, y, feature_names)
        trainer.save("ml/models/churn_model.pkl")
    """

    def __init__(self, config: ChurnModel):
        self.config = config
        self.model  = GradientBoostingClassifier(
            n_estimators  = config.n_estimators,
            max_depth     = config.max_depth,
            learning_rate = config.learning_rate,
            random_state  = config.random_state,
            subsample     = 0.85,
        )
        self._feature_names: list[str] = []

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> tuple[Any, TrainingReport]:
        self._feature_names = feature_names or []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = self.config.test_size,
            random_state = self.config.random_state,
            stratify     = y,
        )

        self.model.fit(X_train, y_train)

        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred  = self.model.predict(X_test)

        auc      = roc_auc_score(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)

        print(f"\n{'='*50}")
        print(f"  ChurnModel — {self.config.target}")
        print(f"{'='*50}")
        print(f"  AUC-ROC       : {auc:.4f}")
        print(f"  Avg Precision : {avg_prec:.4f}")
        print(f"  Train rows    : {len(X_train)}")
        print(f"  Test rows     : {len(X_test)}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['churned','renewed'])}")

        if self._feature_names:
            importance = sorted(
                zip(self._feature_names, self.model.feature_importances_),
                key=lambda t: t[1],
                reverse=True,
            )
            print("  Feature Importance (top 8):")
            for name, imp in importance[:8]:
                bar = "#" * int(imp * 50)
                print(f"    {name:<25} {imp:.4f}  {bar}")

        report = TrainingReport(
            model_name    = "churn_model",
            target        = self.config.target,
            auc           = round(auc, 4),
            avg_precision = round(avg_prec, 4),
            train_rows    = len(X_train),
            test_rows     = len(X_test),
            artifact_path = "",
        )
        return self.model, report

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"ChurnModel salvo: {path}")


class ExpansionTrainer:
    """
    Treina um RandomForestClassifier para predição de upsell/expansão.
    """

    def __init__(self, config: ExpansionModel):
        self.config = config
        self.model  = RandomForestClassifier(
            n_estimators = config.n_estimators,
            max_depth    = config.max_depth,
            random_state = config.random_state,
            class_weight = "balanced",
            n_jobs       = -1,
        )
        self._feature_names: list[str] = []

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> tuple[Any, TrainingReport]:
        self._feature_names = feature_names or []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = self.config.test_size,
            random_state = self.config.random_state,
            stratify     = y,
        )

        self.model.fit(X_train, y_train)

        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred  = self.model.predict(X_test)

        auc      = roc_auc_score(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)

        print(f"\n{'='*50}")
        print(f"  ExpansionModel — {self.config.target}")
        print(f"{'='*50}")
        print(f"  AUC-ROC       : {auc:.4f}")
        print(f"  Avg Precision : {avg_prec:.4f}")
        print(f"  Train rows    : {len(X_train)}")
        print(f"  Test rows     : {len(X_test)}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['no_upsell','upsell'])}")

        report = TrainingReport(
            model_name    = "expansion_model",
            target        = self.config.target,
            auc           = round(auc, 4),
            avg_precision = round(avg_prec, 4),
            train_rows    = len(X_train),
            test_rows     = len(X_test),
            artifact_path = "",
        )
        return self.model, report

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"ExpansionModel salvo: {path}")
