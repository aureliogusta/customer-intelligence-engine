from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from decision_service.dataset import build_ex_ante_training_frame, walk_forward_split
from feature_store.contracts import EX_ANTE_SCHEMA_VERSION
from mlops.registry import dataset_fingerprint, save_dataset_snapshot, save_manifest, save_model_artifact


MODEL_NAME = "ex_ante_policy"


@dataclass
class TrainingReport:
    model_name: str
    model_version: str
    dataset_version: str
    schema_version: str
    train_rows: int
    val_rows: int
    test_rows: int
    baseline_accuracy: float
    test_accuracy: float
    test_f1_macro: float
    test_log_loss: float
    confidence: str
    artifact_path: str


def _build_pipeline() -> Pipeline:
    numeric_features = ["stack_bb", "pot_bb_before", "num_players", "limpers", "open_size_bb", "ante_bb", "bb_chips", "effective_stack_bb", "spr", "open_to_stack_ratio", "board_cards_count", "is_3bet_spot"]
    categorical_features = ["hand", "position", "street", "board_texture"]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto")),
        ]
    )


def train_ex_ante_policy(df_hands: pd.DataFrame, artifact_root: str | Path | None = None) -> TrainingReport:
    training_frame = build_ex_ante_training_frame(df_hands)
    if training_frame.empty:
        raise ValueError("training_frame_empty")

    splits = walk_forward_split(training_frame.sort_values("date_utc", kind="stable"))
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("insufficient_temporal_split")

    feature_cols = [col for col in training_frame.columns if col not in {"label_action", "date_utc", "session_id"}]
    train_session_ids = sorted(train_df["session_id"].dropna().astype(str).unique().tolist())
    val_session_ids = sorted(val_df["session_id"].dropna().astype(str).unique().tolist())
    test_session_ids = sorted(test_df["session_id"].dropna().astype(str).unique().tolist())

    x_train = train_df[feature_cols]
    y_train = train_df["label_action"].astype(str)
    x_val = val_df[feature_cols]
    y_val = val_df["label_action"].astype(str)
    x_test = test_df[feature_cols]
    y_test = test_df["label_action"].astype(str)

    baseline_label = y_train.mode().iloc[0]
    baseline_accuracy = float((y_test == baseline_label).mean()) if len(y_test) else 0.0

    pipeline = _build_pipeline()
    pipeline.fit(x_train, y_train)

    _ = pipeline.predict(x_val)
    test_pred = pipeline.predict(x_test)
    test_proba = pipeline.predict_proba(x_test)

    test_accuracy = float(accuracy_score(y_test, test_pred))
    test_f1_macro = float(f1_score(y_test, test_pred, average="macro"))
    try:
        test_log_loss = float(log_loss(y_test, test_proba, labels=list(pipeline.classes_)))
    except Exception:
        test_log_loss = 0.0

    dataset_version = dataset_fingerprint(training_frame, schema_version=EX_ANTE_SCHEMA_VERSION)
    model_version = f"{MODEL_NAME}-{dataset_version[:12]}"
    confidence = "high" if len(training_frame) >= 2000 else ("medium" if len(training_frame) >= 500 else "low")

    saved_dataset = save_dataset_snapshot(training_frame, dataset_version=dataset_version, schema_version=EX_ANTE_SCHEMA_VERSION, name=MODEL_NAME, artifact_root=artifact_root)
    artifact_path = save_model_artifact(pipeline, model_name=MODEL_NAME, model_version=model_version, metadata={
        "schema_version": EX_ANTE_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "feature_columns": feature_cols,
        "classes": list(pipeline.classes_),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_session_ids": train_session_ids,
        "val_session_ids": val_session_ids,
        "test_session_ids": test_session_ids,
        "baseline_accuracy": baseline_accuracy,
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "test_log_loss": test_log_loss,
        "confidence": confidence,
        "dataset_path": str(saved_dataset),
    }, artifact_root=artifact_root)

    save_manifest(
        model_name=MODEL_NAME,
        payload={
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "model_version": model_version,
            "dataset_version": dataset_version,
            "schema_version": EX_ANTE_SCHEMA_VERSION,
            "feature_columns": feature_cols,
            "classes": list(pipeline.classes_),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_session_ids": train_session_ids,
            "val_session_ids": val_session_ids,
            "test_session_ids": test_session_ids,
            "baseline_accuracy": baseline_accuracy,
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1_macro,
            "test_log_loss": test_log_loss,
            "confidence": confidence,
            "artifact_path": artifact_path,
        },
        artifact_root=artifact_root,
    )

    return TrainingReport(
        model_name=MODEL_NAME,
        model_version=model_version,
        dataset_version=dataset_version,
        schema_version=EX_ANTE_SCHEMA_VERSION,
        train_rows=int(len(train_df)),
        val_rows=int(len(val_df)),
        test_rows=int(len(test_df)),
        baseline_accuracy=baseline_accuracy,
        test_accuracy=test_accuracy,
        test_f1_macro=test_f1_macro,
        test_log_loss=test_log_loss,
        confidence=confidence,
        artifact_path=artifact_path,
    )
