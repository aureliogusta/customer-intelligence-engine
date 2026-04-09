"""
analytics/drift_monitor.py
===========================
Detecção de data drift e decisão de retreino automático.

Compara distribuição das features de produção vs o baseline de treino
usando o teste KS (Kolmogorov-Smirnov) e PSI (Population Stability Index).

Uso:
    baseline = pd.read_csv("data/train_dataset.csv")
    detector = DriftDetector(baseline)

    new_data  = fetch_recent_accounts_from_db()
    drifts    = detector.check_drift(new_data)

    if detector.should_retrain(drifts):
        trigger_retraining()
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureDrift:
    feature_name:    str
    baseline_mean:   float
    baseline_std:    float
    current_mean:    float
    current_std:     float
    ks_statistic:    float    # 0–1, maior = mais drift
    ks_pvalue:       float    # < 0.05 = drift estatisticamente significativo
    psi:             float    # > 0.25 = drift severo
    drift_detected:  bool


@dataclass
class DriftReport:
    timestamp:        str
    n_baseline:       int
    n_current:        int
    drifts:           List[FeatureDrift]
    should_retrain:   bool
    critical_features: List[str]   # features críticas com drift

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["drifts"] = [asdict(f) for f in self.drifts]
        return d

    def summary_line(self) -> str:
        drifted = [d.feature_name for d in self.drifts if d.drift_detected]
        return (
            f"[DriftReport {self.timestamp[:10]}] "
            f"{len(drifted)}/{len(self.drifts)} features com drift. "
            f"Retreinar: {self.should_retrain}. "
            f"Críticas: {drifted or 'nenhuma'}"
        )


# ── Detector ──────────────────────────────────────────────────────────────────

# Features mais importantes para o modelo de churn (impactam mais a decisão)
CRITICAL_FEATURES = {
    "engajamento_pct",
    "nps_score",
    "dias_sem_interacao",
    "engagement_trend",
    "nps_trend",
}

PSI_WARNING  = 0.10   # PSI moderado
PSI_CRITICAL = 0.25   # PSI severo — retreino recomendado


class DriftDetector:
    """
    Monitora distribuição de features em produção vs baseline de treino.

    Métricas usadas:
      - KS test: detecta mudanças na distribuição (p < 0.05 = drift)
      - PSI: mede magnitude do shift (> 0.25 = drift severo)
    """

    def __init__(
        self,
        baseline_df:       pd.DataFrame,
        critical_features: set[str] | None = None,
        psi_threshold:     float = PSI_CRITICAL,
        ks_alpha:          float = 0.05,
    ):
        self.baseline_df       = baseline_df
        self.critical_features = critical_features or CRITICAL_FEATURES
        self.psi_threshold     = psi_threshold
        self.ks_alpha          = ks_alpha
        self._numeric_cols     = self._get_numeric_cols(baseline_df)
        self._baseline_stats   = self._compute_stats(baseline_df)

    # ── Public API ────────────────────────────────────────────────────────────

    def check_drift(self, current_df: pd.DataFrame) -> DriftReport:
        """
        Compara current_df contra baseline e retorna DriftReport.
        current_df deve ter as mesmas colunas numéricas que o baseline.
        """
        drifts: List[FeatureDrift] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        for col in self._numeric_cols:
            if col not in current_df.columns:
                continue

            baseline_vals = self.baseline_df[col].dropna().values
            current_vals  = current_df[col].dropna().values

            if len(current_vals) < 10:
                continue

            ks_stat, ks_pval = self._ks_test(baseline_vals, current_vals)
            psi              = self._compute_psi(baseline_vals, current_vals)
            drift_detected   = (ks_pval < self.ks_alpha) or (psi > self.psi_threshold)

            drifts.append(FeatureDrift(
                feature_name   = col,
                baseline_mean  = float(np.mean(baseline_vals)),
                baseline_std   = float(np.std(baseline_vals)),
                current_mean   = float(np.mean(current_vals)),
                current_std    = float(np.std(current_vals)),
                ks_statistic   = round(float(ks_stat), 4),
                ks_pvalue      = round(float(ks_pval), 6),
                psi            = round(float(psi), 4),
                drift_detected = drift_detected,
            ))

        critical_drifted = [
            d.feature_name for d in drifts
            if d.drift_detected and d.feature_name in self.critical_features
        ]

        return DriftReport(
            timestamp        = timestamp,
            n_baseline       = len(self.baseline_df),
            n_current        = len(current_df),
            drifts           = drifts,
            should_retrain   = len(critical_drifted) >= 2,
            critical_features= critical_drifted,
        )

    def should_retrain(self, report: DriftReport) -> bool:
        return report.should_retrain

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _ks_test(self, a: np.ndarray, b: np.ndarray):
        try:
            from scipy.stats import ks_2samp
            return ks_2samp(a, b)
        except ImportError:
            # Fallback sem scipy: compara médias (menos preciso)
            diff  = abs(float(np.mean(a)) - float(np.mean(b)))
            scale = float(np.std(a)) + 1e-9
            stat  = min(diff / scale, 1.0)
            pval  = 0.01 if stat > 0.3 else 0.99
            return stat, pval

    def _compute_psi(self, baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """Population Stability Index."""
        try:
            breaks   = np.percentile(baseline, np.linspace(0, 100, buckets + 1))
            breaks   = np.unique(breaks)
            if len(breaks) < 2:
                return 0.0

            base_counts = np.histogram(baseline, bins=breaks)[0].astype(float)
            curr_counts = np.histogram(current,  bins=breaks)[0].astype(float)

            base_pct = (base_counts + 0.5) / (len(baseline) + buckets * 0.5)
            curr_pct = (curr_counts + 0.5) / (len(current)  + buckets * 0.5)

            psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
            return max(0.0, psi)
        except Exception:
            return 0.0

    def _get_numeric_cols(self, df: pd.DataFrame) -> List[str]:
        skip = {"account_id", "segment", "profile", "renovado", "fez_upsell",
                "contract_start", "contract_end", "csm_name", "name", "industry"}
        return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        return {
            col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
            for col in self._numeric_cols
            if col in df.columns
        }
