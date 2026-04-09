"""
decision_service/inference.py
==============================
Carrega modelos treinados e aplica a novas contas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from .models import ChurnModel, ExpansionModel


class ChurnPredictor:
    """
    Predição de churn para uma conta.

    Uso:
        predictor = ChurnPredictor(
            model_path="ml/models/churn_model.pkl",
            scaler_path="ml/models/churn_scaler.pkl",
        )
        result = predictor.predict({
            "engajamento_pct": 35.0,
            "nps_score": 4.0,
            ...
        })
    """

    FEATURES = ChurnModel().features

    def __init__(
        self,
        model_path:  str | Path,
        scaler_path: str | Path,
        encoder_path: str | Path | None = None,
    ):
        self.model   = joblib.load(model_path)
        self.scaler  = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path) if encoder_path and Path(encoder_path).exists() else None

    def _encode_segment(self, features: dict) -> dict:
        f = dict(features)
        if "segment" in f:
            if self.encoder is not None:
                try:
                    f["segment_encoded"] = int(self.encoder.transform([str(f["segment"])])[0])
                except Exception:
                    f["segment_encoded"] = 0
            else:
                seg_map = {"SMB": 0, "MID_MARKET": 1, "ENTERPRISE": 2}
                f["segment_encoded"] = seg_map.get(str(f.get("segment", "")), 1)
        return f

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Retorna churn_risk (0–1), risk_level e retention_prob."""
        f = self._encode_segment(features_dict)

        # Montar DataFrame com as features do modelo
        row = {feat: f.get(feat, 0) for feat in self.FEATURES}
        df  = pd.DataFrame([row])

        X = self.scaler.transform(df)

        proba = self.model.predict_proba(X)[0]
        # classes: [0=churned, 1=renewed]  →  churn_risk = P(churned)
        churn_risk = float(proba[0])
        retention  = float(proba[1])

        if churn_risk > 0.70:
            level = "HIGH"
        elif churn_risk > 0.40:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "churn_risk":      round(churn_risk, 4),
            "retention_prob":  round(retention,  4),
            "risk_level":      level,
        }


class ExpansionPredictor:
    """
    Predição de probabilidade de upsell/expansão.
    """

    FEATURES = ExpansionModel().features

    def __init__(
        self,
        model_path:   str | Path,
        scaler_path:  str | Path,
        encoder_path: str | Path | None = None,
    ):
        self.model   = joblib.load(model_path)
        self.scaler  = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path) if encoder_path and Path(encoder_path).exists() else None

    def _encode_segment(self, features: dict) -> dict:
        f = dict(features)
        if "segment" in f:
            if self.encoder is not None:
                try:
                    f["segment_encoded"] = int(self.encoder.transform([str(f["segment"])])[0])
                except Exception:
                    f["segment_encoded"] = 0
            else:
                seg_map = {"SMB": 0, "MID_MARKET": 1, "ENTERPRISE": 2}
                f["segment_encoded"] = seg_map.get(str(f.get("segment", "")), 1)
        return f

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        f   = self._encode_segment(features_dict)
        row = {feat: f.get(feat, 0) for feat in self.FEATURES}
        df  = pd.DataFrame([row])

        X = self.scaler.transform(df)

        proba      = self.model.predict_proba(X)[0]
        upsell_prob = float(proba[1])

        return {
            "upsell_probability": round(upsell_prob, 4),
            "upsell_signal":      "HIGH" if upsell_prob > 0.5 else "LOW",
        }


def batch_predict(
    csv_path:     str | Path,
    churn_pred:   ChurnPredictor,
    expansion_pred: ExpansionPredictor | None = None,
    output_path:  str | Path | None = None,
) -> pd.DataFrame:
    """Aplica os modelos a um CSV inteiro. Útil para testes em lote."""
    df = pd.read_csv(csv_path)

    results: List[Dict] = []
    for _, row in df.iterrows():
        features = row.to_dict()
        churn    = churn_pred.predict(features)
        record   = {"account_id": features.get("account_id", ""), **churn}
        if expansion_pred:
            expansion = expansion_pred.predict(features)
            record.update(expansion)
        results.append(record)

    out_df = pd.DataFrame(results)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        print(f"Predições salvas: {output_path}")

    return out_df
