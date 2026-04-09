"""
decision_service/dataset.py
============================
Carrega train_dataset.csv, codifica categorias e normaliza features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .models import ChurnModel, ExpansionModel


class ChurnDataset:
    """
    Prepara o dataset de churn para treinamento.

    Uso:
        config  = ChurnModel()
        ds      = ChurnDataset("data/train_dataset.csv", config)
        X, y, scaler, le = ds.prepare()
    """

    def __init__(self, csv_path: str | Path, model_config: ChurnModel | ExpansionModel):
        self.path   = Path(csv_path)
        self.config = model_config
        self.df     = pd.read_csv(self.path)
        self._le: LabelEncoder | None = None

    # ── Encode + scale ────────────────────────────────────────────────────────

    def _encode_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encodes 'segment' → 'segment_encoded'."""
        df = df.copy()
        if "segment" in df.columns:
            self._le = LabelEncoder()
            df["segment_encoded"] = self._le.fit_transform(df["segment"].astype(str))
        else:
            df["segment_encoded"] = 0
        return df

    def prepare(self) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder | None]:
        """
        Retorna (X_scaled, y, scaler, label_encoder).
        X_scaled e y prontos para sklearn.
        """
        df = self._encode_segment(self.df)

        # Apenas as features configuradas
        available = [f for f in self.config.features if f in df.columns]
        missing   = set(self.config.features) - set(available)
        if missing:
            print(f"[ChurnDataset] Features ausentes (serão ignoradas): {missing}")

        X = df[available].copy().fillna(0)
        y = df[self.config.target].astype(int)

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y.values, scaler, self._le

    def feature_names(self) -> list[str]:
        """Retorna as features que realmente existem no CSV."""
        df_cols = set(self.df.columns) | {"segment_encoded"}
        return [f for f in self.config.features if f in df_cols]

    @staticmethod
    def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
        joblib.dump(scaler, path)
        print(f"Scaler salvo: {path}")

    @staticmethod
    def save_encoder(le: LabelEncoder, path: str | Path) -> None:
        joblib.dump(le, path)
        print(f"LabelEncoder salvo: {path}")
