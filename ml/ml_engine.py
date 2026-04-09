"""
ml_engine.py
============
Scikit-Learn Pipelines — Roadmap Block: Predictive Modeling.

Responsabilidades
-----------------
  1. FeatureEngineer   → transforma mãos brutas em features para ML
  2. OpponentProfiler  → classifica oponentes por VPIP/PFR/tendências
  3. DecisionAdjuster  → ajusta decisões GTO baseado no histórico do hero
  4. ModelTrainer      → treina, valida e serializa os modelos
  5. MLEngine          → interface unificada para o decision_engine.py

Arquitetura de dados
--------------------
  ENTRADA  : DataFrame do PostgreSQL (hands) ou CSV (hands.csv)
  SAÍDA    : fator de ajuste [0.5–1.5] que modifica o EV calculado pelo MC
             + label de confiança baseado no volume de amostras similares

Integração com o sistema
------------------------
  decision_engine.py chama MLEngine.adjust(context) antes de retornar
  a decisão final. Se não houver modelo treinado, retorna 1.0 (neutro).

  Fluxo:
    Monte Carlo → EV matemático
    MLEngine    → fator de ajuste baseado no seu histórico
    EV final    = EV_matematico × fator_ajuste
    Decisão     = engine.decidir(EV_final)

Run:
  python ml_engine.py --train          # treina com dados existentes
  python ml_engine.py --evaluate       # métricas do modelo atual
  python ml_engine.py --predict        # modo interativo de teste
  python ml_engine.py --report         # relatório de leaks + perfil
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy  as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Scikit-Learn
try:
    from sklearn.pipeline           import Pipeline
    from sklearn.preprocessing      import StandardScaler, LabelEncoder
    from sklearn.ensemble           import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.calibration        import CalibratedClassifierCV
    from sklearn.linear_model       import LogisticRegression
    from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics            import (mean_absolute_error, r2_score,
                                            classification_report, confusion_matrix)
    from sklearn.impute             import SimpleImputer
    _SKLEARN_AVAILABLE = True
    _SKLEARN_IMPORT_ERROR = ""
except Exception as _sk_err:
    _SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = str(_sk_err)
try:
    import joblib
    _JOBLIB_AVAILABLE = True
    _JOBLIB_IMPORT_ERROR = ""
except Exception as _jb_err:
    joblib = None  # type: ignore[assignment]
    _JOBLIB_AVAILABLE = False
    _JOBLIB_IMPORT_ERROR = str(_jb_err)

# ── Path setup ────────────────────────────────────────────────────────────────

_BASE_DIR  = Path(__file__).parent.resolve()
_MODEL_DIR = _BASE_DIR / "models"
_MODEL_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(_BASE_DIR))

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml_engine")

# =============================================================================
# Configuração
# =============================================================================

# Volume mínimo de amostras para treinar com confiança
MIN_SAMPLES_TRAIN    = 500    # mínimo para treinar o modelo
MIN_SAMPLES_RELIABLE = 2000   # volume para confiança alta
MIN_SAMPLES_CONTEXT  = 10     # mínimo para ajuste contextual

# Paths dos modelos serializados
MODEL_EV_ADJUSTER    = _MODEL_DIR / "ev_adjuster.joblib"
MODEL_RESULT_CLF     = _MODEL_DIR / "result_classifier.joblib"
MODEL_METADATA       = _MODEL_DIR / "model_metadata.json"

STACK_BUCKETS = {
    "LOW":  (0.0, 20.0),
    "MID":  (20.0, 50.0),
    "HIGH": (50.0, 10_000.0),
}

MODEL_EV_SEGMENT = {
    bucket: _MODEL_DIR / f"ev_adjuster_{bucket.lower()}.joblib"
    for bucket in STACK_BUCKETS
}
MODEL_CLF_SEGMENT = {
    bucket: _MODEL_DIR / f"result_classifier_{bucket.lower()}.joblib"
    for bucket in STACK_BUCKETS
}

# Benchmarks GTO para detecção de desvios
GTO_BENCHMARKS = {
    "UTG":   {"vpip": 0.14, "pfr": 0.13},
    "MP":    {"vpip": 0.17, "pfr": 0.15},
    "HJ":    {"vpip": 0.20, "pfr": 0.18},
    "CO":    {"vpip": 0.26, "pfr": 0.23},
    "BTN":   {"vpip": 0.45, "pfr": 0.38},
    "SB":    {"vpip": 0.35, "pfr": 0.28},
    "BB":    {"vpip": 0.40, "pfr": 0.10},
}

# =============================================================================
# 1. Feature Engineering
# =============================================================================

class FeatureEngineer:
    """
    Transforma o DataFrame bruto de mãos em features numéricas para ML.

    Features geradas
    ----------------
    Posição (one-hot)     : BTN, CO, HJ, MP, UTG, SB, BB
    Stack zone            : deep (>50bb), medium (20-50bb), short (<20bb)
    M-ratio zone          : green, yellow, orange, red
    Ação pré-flop         : fold=0, call=1, raise=2
    Board texture         : sem board=0, dry=1, wet=2, paired=3, monotone=4
    Multiway              : 0/1
    Tier da mão           : 1-6 (mapeado de hero_cards)
    Resultado numérico    : hero_amount_won (target para regressão)
    Resultado binário     : win=1, não-win=0 (target para classificação)
    """

    POSITIONS = ["BTN", "CO", "HJ", "MP", "UTG", "UTG+1", "LJ", "SB", "BB"]

    POSITION_ORDER = {p: i for i, p in enumerate(POSITIONS)}

    def __init__(self) -> None:
        # Importa knowledge_base se disponível
        try:
            from knowledge_base import HAND_KEY_BY_LABEL, HAND_RANKINGS
            from decision_engine import normalize_hand
            self._kb_available  = True
            self._hand_key      = HAND_KEY_BY_LABEL
            self._hand_rankings = HAND_RANKINGS
            self._normalize     = normalize_hand
        except ImportError:
            self._kb_available = False
            log.warning("knowledge_base não disponível — tier das mãos não será calculado.")

    def _get_tier(self, cards: str) -> int:
        """Retorna o tier (1-6) da mão. 0 se não encontrado."""
        if not self._kb_available or not cards:
            return 0
        try:
            canonical = self._normalize(cards.replace(" ", ""))
            key       = self._hand_key.get(canonical)
            if key:
                return self._hand_rankings[key]["tier"]
        except Exception:
            pass
        return 0

    def _action_to_int(self, action: str) -> int:
        """Converte ação pré-flop em inteiro: fold=0, call=1, raise=2."""
        if not action:
            return 0
        a = action.lower()
        if "raise" in a or "3bet" in a or "all-in" in a:
            return 2
        if "call" in a or "flat" in a:
            return 1
        return 0

    def _stack_zone(self, stack: float, bb: float) -> int:
        """Zona de stack: 0=short, 1=medium, 2=deep."""
        if bb <= 0:
            return 1
        ratio = stack / bb
        if ratio < 20:
            return 0
        if ratio < 50:
            return 1
        return 2

    def _m_zone(self, m_ratio: float) -> int:
        """Zona de M-ratio: 0=red, 1=orange, 2=yellow, 3=green."""
        if m_ratio < 6:   return 0
        if m_ratio < 10:  return 1
        if m_ratio < 20:  return 2
        return 3

    def _board_texture(self, flop: str) -> int:
        """Textura do board: 0=sem board, 1=dry, 2=wet, 3=paired, 4=monotone."""
        if not flop or flop.strip() == "":
            return 0
        cards = flop.strip().split()
        if len(cards) < 3:
            return 1
        suits = [c[-1].lower() for c in cards if len(c) >= 2]
        ranks = [c[:-1].upper() for c in cards if len(c) >= 2]
        if len(set(suits)) == 1:
            return 4   # monotone
        rank_freq = {r: ranks.count(r) for r in set(ranks)}
        if max(rank_freq.values()) >= 2:
            return 3   # paired
        if len(set(suits)) == 2:
            return 2   # wet (two-tone)
        return 1       # dry

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma o DataFrame bruto em features para ML.

        Parâmetros
        ----------
        df : DataFrame com colunas do HandRecord / PostgreSQL

        Retorna
        -------
        DataFrame com features numéricas prontas para Scikit-Learn
        """
        feat = pd.DataFrame()

        # ── Posição (ordinal) ──
        feat["position_ord"] = df["hero_position"].map(
            self.POSITION_ORDER
        ).fillna(4).astype(int)

        # ── Posição (one-hot) ──
        for pos in ["BTN", "CO", "HJ", "MP", "SB", "BB", "UTG"]:
            feat[f"pos_{pos}"] = (df["hero_position"] == pos).astype(int)

        # ── Stack zone ──
        feat["stack_zone"] = df.apply(
            lambda r: self._stack_zone(
                r.get("hero_stack_start", 0),
                r.get("big_blind", 1)
            ), axis=1
        )

        # ── M-ratio zone ──
        feat["m_zone"] = df["m_ratio"].fillna(10).apply(self._m_zone)
        feat["m_ratio"] = df["m_ratio"].fillna(10).clip(0, 50)

        # ── Ação pré-flop ──
        feat["action_preflop"] = df["hero_action_preflop"].fillna("").apply(
            self._action_to_int
        )

        # ── Flags ──
        feat["vpip"]       = df["hero_vpip"].fillna(0).astype(int)
        feat["pfr"]        = df["hero_pfr"].fillna(0).astype(int)
        feat["aggressor"]  = df["hero_aggressor"].fillna(0).astype(int)
        feat["went_allin"] = df["hero_went_allin"].fillna(0).astype(int)

        # ── Board texture ──
        feat["board_texture"] = df["board_flop"].fillna("").apply(
            self._board_texture
        )
        feat["has_board"] = (feat["board_texture"] > 0).astype(int)

        # ── Número de jogadores ──
        num_players = pd.to_numeric(df["num_players"], errors="coerce").fillna(6)
        feat["num_players"] = num_players.clip(2, 9).astype(int)
        feat["multiway"]    = (feat["num_players"] > 2).astype(int)

        # ── Tier da mão ──
        feat["hand_tier"] = df["hero_cards"].fillna("").apply(self._get_tier)

        # ── Nível do torneio ──
        level_num = pd.to_numeric(df["level"], errors="coerce").fillna(1)
        feat["level"] = level_num.clip(1, 50).astype(int)

        # ── Went to showdown ──
        feat["wtsd"] = df["went_to_showdown"].fillna(0).astype(int)

        # ── Features do módulo de estudos (quando disponíveis) ──
        feat["study_severity_score"] = pd.to_numeric(
            df.get("study_severity_score", 0), errors="coerce"
        ).fillna(0.0).clip(0.0, 100.0)
        feat["study_priority_rank"] = pd.to_numeric(
            df.get("study_priority_rank", 0), errors="coerce"
        ).fillna(0.0).clip(0.0, 100.0)
        feat["study_confidence"] = pd.to_numeric(
            df.get("study_confidence", 0), errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        feat["study_is_critical"] = (
            df.get("study_severity_label", "")
            .fillna("")
            .astype(str)
            .str.upper()
            .eq("CRÍTICO")
            .astype(int)
        )

        # ── Targets ──
        # No modo de inferencia online, essas colunas nao existem.
        if "hero_amount_won" in df.columns:
            feat["target_amount"] = df["hero_amount_won"].fillna(0)
        else:
            feat["target_amount"] = 0

        if "hero_result" in df.columns:
            feat["target_win"] = (df["hero_result"] == "win").astype(int)
        else:
            feat["target_win"] = 0

        return feat.reset_index(drop=True)


# =============================================================================
# 2. Model Trainer
# =============================================================================

class ModelTrainer:
    """
    Treina dois modelos complementares:

    1. EV Adjuster (GradientBoostingRegressor)
       Prevê hero_amount_won dado o contexto.
       Usado para calibrar o fator de ajuste do EV matemático.

    2. Result Classifier (RandomForestClassifier)
       Prevê probabilidade de vitória dado o contexto.
       Usado para calcular confiança da decisão.
    """

    def __init__(self) -> None:
        self.fe = FeatureEngineer()
        self._feature_cols = [
            "position_ord", "pos_BTN", "pos_CO", "pos_HJ", "pos_MP",
            "pos_SB", "pos_BB", "pos_UTG",
            "stack_zone", "m_zone", "m_ratio",
            "action_preflop", "vpip", "pfr", "aggressor", "went_allin",
            "board_texture", "has_board", "num_players", "multiway",
            "hand_tier", "level", "wtsd",
            "study_severity_score", "study_priority_rank", "study_confidence", "study_is_critical",
        ]

    def _build_pipelines(self):
        """Constrói os pipelines de ML."""
        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()

        ev_pipeline = Pipeline([
            ("imputer", imputer),
            ("scaler",  scaler),
            ("model",   GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            )),
        ])

        clf_pipeline = Pipeline([
            ("imputer", imputer),
            ("scaler",  scaler),
            ("model",   CalibratedClassifierCV(
                estimator=RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
                cv=3,
                method="sigmoid",
            )),
        ])

        return ev_pipeline, clf_pipeline

    def _stack_bucket_name(self, stack_bb: float) -> str:
        s = float(stack_bb)
        for bucket, (lo, hi) in STACK_BUCKETS.items():
            if lo <= s < hi:
                return bucket
        return "HIGH"

    def _canonical_action(self, action: str) -> str:
        a = (action or "").upper()
        if "ALL" in a and "IN" in a:
            return "ALL-IN"
        if "FOLD" in a:
            return "FOLD"
        if "CALL" in a or "FLAT" in a:
            return "CALL"
        if any(k in a for k in ("RAISE", "RFI", "3BET", "3-BET", "4BET", "4-BET", "BET")):
            return "BET"
        return "FOLD"

    def _load_gto_ranges(self) -> dict:
        p = _BASE_DIR / "gto_ranges.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _nearest_depth_key(self, keys: list[str], stack_bb: float) -> str | None:
        parsed: list[tuple[float, str]] = []
        for k in keys:
            try:
                parsed.append((float(str(k).replace("bb", "")), k))
            except Exception:
                continue
        if not parsed:
            return None
        return min(parsed, key=lambda x: abs(x[0] - float(stack_bb)))[1]

    def _expand_range_token(self, token: str) -> set[str]:
        # Usa o mesmo expansor do motor para manter coerência de auditoria.
        try:
            import decision_engine as de
            return set(de._expand_range(token))
        except Exception:
            return set()

    def _build_push_set(self, raw_depth: dict[str, float], threshold: float = 0.5) -> set[str]:
        out: set[str] = set()
        for token, weight in raw_depth.items():
            if str(token).startswith("_"):
                continue
            try:
                w = float(weight)
            except Exception:
                continue
            if w >= threshold:
                out.update(self._expand_range_token(token))
        return out

    def _gto_expected_action(self, gto: dict, hand: str, position: str, stack_bb: float) -> str | None:
        pos = (position or "").upper()
        if not pos:
            return None

        # LOW: push/fold
        if float(stack_bb) <= 20.0:
            push_fold = gto.get("push_fold", {})
            pos_map = push_fold.get(pos, {}) if isinstance(push_fold, dict) else {}
            if not isinstance(pos_map, dict) or not pos_map:
                return None
            depth_key = self._nearest_depth_key([k for k in pos_map.keys() if not str(k).startswith("_")], stack_bb)
            if not depth_key:
                return None
            raw = pos_map.get(depth_key, {})
            if not isinstance(raw, dict):
                return None
            return "ALL-IN" if hand in self._build_push_set(raw) else "FOLD"

        # MID/HIGH: RFI dominante
        rfi = gto.get("RFI", {})
        pos_map = rfi.get(pos, {}) if isinstance(rfi, dict) else {}
        if not isinstance(pos_map, dict) or not pos_map:
            return None
        depth_key = self._nearest_depth_key([k for k in pos_map.keys() if not str(k).startswith("_")], stack_bb)
        if not depth_key:
            return None
        table = pos_map.get(depth_key, {})
        if not isinstance(table, dict):
            return None
        freq = table.get(hand)
        if not isinstance(freq, dict):
            return "FOLD"
        r = float(freq.get("r", 0.0) or 0.0)
        c = float(freq.get("c", 0.0) or 0.0)
        f = float(freq.get("f", 0.0) or 0.0)
        return max(("BET", r), ("CALL", c), ("FOLD", f), key=lambda x: x[1])[0]

    def _scenario_concordance(self, df: pd.DataFrame) -> dict:
        """Mede concordância da ação histórica pré-flop vs chart GTO por bucket de stack."""
        required = {"hero_cards", "hero_position", "hero_action_preflop", "hero_stack_start", "big_blind"}
        if not required.issubset(set(df.columns)):
            return {"LOW": None, "MID": None, "HIGH": None, "overall": None, "n": 0}

        try:
            from decision_engine import normalize_hand
        except Exception:
            normalize_hand = lambda x: str(x or "")

        gto = self._load_gto_ranges()
        if not gto:
            return {"LOW": None, "MID": None, "HIGH": None, "overall": None, "n": 0}

        stats = {
            "LOW": {"ok": 0, "n": 0},
            "MID": {"ok": 0, "n": 0},
            "HIGH": {"ok": 0, "n": 0},
        }

        for _, row in df.iterrows():
            try:
                bb = float(row.get("big_blind", 0) or 0)
                stack = float(row.get("hero_stack_start", 0) or 0)
            except Exception:
                continue
            if bb <= 0:
                continue

            stack_bb = stack / bb
            bucket = self._stack_bucket_name(stack_bb)
            hand = normalize_hand(str(row.get("hero_cards", "") or "").replace(" ", ""))
            if not hand:
                continue

            expected = self._gto_expected_action(gto, hand, str(row.get("hero_position", "")), stack_bb)
            if not expected:
                continue

            actual = self._canonical_action(str(row.get("hero_action_preflop", "")))
            # No recorte atual, em MID/HIGH tratamos ALL-IN como BET no comparativo pré-flop unopened.
            if expected in ("BET", "CALL", "FOLD") and actual == "ALL-IN":
                actual = "BET"

            stats[bucket]["n"] += 1
            if actual == expected:
                stats[bucket]["ok"] += 1

        out: dict[str, float | int | None] = {"n": 0}
        total_ok = 0
        total_n = 0
        for bucket in ("LOW", "MID", "HIGH"):
            n = stats[bucket]["n"]
            ok = stats[bucket]["ok"]
            out[bucket] = round(ok / n, 3) if n else None
            out[f"{bucket}_n"] = n
            total_ok += ok
            total_n += n

        out["overall"] = round(total_ok / total_n, 3) if total_n else None
        out["n"] = total_n
        return out

    def train(self, df: pd.DataFrame) -> dict:
        """
        Treina os modelos com o DataFrame fornecido.

        Parâmetros
        ----------
        df : DataFrame bruto do PostgreSQL ou CSV

        Retorna
        -------
        dict com métricas de performance
        """
        raise RuntimeError(
            "legacy_ml_engine_disabled_use_decision_service_training_or_analysis_service"
        )
        n = len(df)
        log.info("Iniciando treinamento com %d amostras...", n)

        if n < MIN_SAMPLES_TRAIN:
            log.warning(
                "Volume insuficiente (%d amostras). Mínimo recomendado: %d. "
                "Modelo será treinado mas com baixa confiança.",
                n, MIN_SAMPLES_TRAIN
            )

        # Feature engineering
        feat = self.fe.transform(df)
        X    = feat[self._feature_cols].values
        y_ev  = feat["target_amount"].values
        y_clf = feat["target_win"].values

        # Split
        X_train, X_test, y_ev_train, y_ev_test, y_clf_train, y_clf_test = \
            train_test_split(X, y_ev, y_clf, test_size=0.2, random_state=42)

        ev_pipe, clf_pipe = self._build_pipelines()

        # ── Treina EV Adjuster ──
        log.info("Treinando EV Adjuster (GradientBoosting)...")
        ev_pipe.fit(X_train, y_ev_train)
        y_ev_pred = ev_pipe.predict(X_test)
        mae = mean_absolute_error(y_ev_test, y_ev_pred)
        r2  = r2_score(y_ev_test, y_ev_pred)
        log.info("  MAE=%.2f  R²=%.3f", mae, r2)

        # ── Treina Result Classifier ──
        log.info("Treinando Result Classifier (RandomForest)...")
        clf_pipe.fit(X_train, y_clf_train)
        y_clf_pred = clf_pipe.predict(X_test)
        clf_report = classification_report(
            y_clf_test, y_clf_pred,
            target_names=["não-win", "win"],
            output_dict=True,
        )
        win_f1 = clf_report.get("win", {}).get("f1-score", 0)
        log.info("  Win F1=%.3f", win_f1)

        # ── Salva modelos ──
        joblib.dump(ev_pipe,  MODEL_EV_ADJUSTER)
        joblib.dump(clf_pipe, MODEL_RESULT_CLF)

        # ── Treino segmentado por stack bucket ──
        bucket_metrics: dict[str, dict] = {}
        for bucket, zone_code in (("LOW", 0), ("MID", 1), ("HIGH", 2)):
            mask = feat["stack_zone"] == zone_code
            n_bucket = int(mask.sum())
            bucket_metrics[bucket] = {
                "n_samples": n_bucket,
                "trained": False,
                "ev_mae": None,
                "ev_r2": None,
                "win_f1": None,
                "confidence": "low",
            }

            # Evita overfit extremo e erro de treino em buckets muito pequenos.
            if n_bucket < 120:
                continue

            Xb = feat.loc[mask, self._feature_cols].values
            yb_ev = feat.loc[mask, "target_amount"].values
            yb_clf = feat.loc[mask, "target_win"].values
            Xb_train, Xb_test, yb_ev_train, yb_ev_test, yb_clf_train, yb_clf_test = train_test_split(
                Xb, yb_ev, yb_clf, test_size=0.2, random_state=42
            )
            ev_b, clf_b = self._build_pipelines()
            ev_b.fit(Xb_train, yb_ev_train)
            clf_b.fit(Xb_train, yb_clf_train)

            yb_ev_pred = ev_b.predict(Xb_test)
            yb_clf_pred = clf_b.predict(Xb_test)
            b_mae = float(mean_absolute_error(yb_ev_test, yb_ev_pred))
            b_r2 = float(r2_score(yb_ev_test, yb_ev_pred))
            b_rep = classification_report(
                yb_clf_test,
                yb_clf_pred,
                target_names=["não-win", "win"],
                output_dict=True,
            )
            b_f1 = float(b_rep.get("win", {}).get("f1-score", 0.0))

            joblib.dump(ev_b, MODEL_EV_SEGMENT[bucket])
            joblib.dump(clf_b, MODEL_CLF_SEGMENT[bucket])

            bucket_metrics[bucket].update({
                "trained": True,
                "ev_mae": round(b_mae, 3),
                "ev_r2": round(b_r2, 3),
                "win_f1": round(b_f1, 3),
                "confidence": "high" if n_bucket >= MIN_SAMPLES_RELIABLE else "medium",
            })

        concordance = self._scenario_concordance(df)

        # ── Salva metadados ──
        metadata = {
            "trained_at":    pd.Timestamp.now().isoformat(),
            "n_samples":     int(n),
            "features":      self._feature_cols,
            "ev_mae":        round(float(mae), 3),
            "ev_r2":         round(float(r2), 3),
            "win_f1":        round(float(win_f1), 3),
            "confidence":    "high" if n >= MIN_SAMPLES_RELIABLE else
                             "medium" if n >= MIN_SAMPLES_TRAIN else "low",
            "segmented_training": bucket_metrics,
            "scenario_concordance": concordance,
            "ml_policy_version": "2026.03-segmented-1",
        }
        MODEL_METADATA.write_text(json.dumps(metadata, indent=2))

        log.info("Modelos salvos em %s", _MODEL_DIR)
        log.info("Confiança: %s (%d amostras)", metadata["confidence"], n)

        return metadata

    def evaluate(self, df: pd.DataFrame) -> None:
        """Avalia os modelos em um conjunto de dados."""
        raise RuntimeError(
            "legacy_ml_engine_disabled_use_decision_service_training_or_analysis_service"
        )
        if not MODEL_EV_ADJUSTER.exists():
            log.error("Modelos não encontrados. Execute --train primeiro.")
            return

        feat = self.fe.transform(df)
        X    = feat[self._feature_cols].values
        y_ev  = feat["target_amount"].values
        y_clf = feat["target_win"].values

        ev_pipe  = joblib.load(MODEL_EV_ADJUSTER)
        clf_pipe = joblib.load(MODEL_RESULT_CLF)

        # Cross-validation
        ev_cv = cross_val_score(ev_pipe, X, y_ev, cv=5,
                                scoring="neg_mean_absolute_error")
        log.info("EV Adjuster CV MAE: %.2f ± %.2f",
                 -ev_cv.mean(), ev_cv.std())

        clf_cv = cross_val_score(clf_pipe, X, y_clf, cv=5,
                                 scoring="f1")
        log.info("Result Classifier CV F1: %.3f ± %.3f",
                 clf_cv.mean(), clf_cv.std())

        # Feature importance
        clf_model = clf_pipe.named_steps["model"]
        base_model = getattr(clf_model, "estimator", None)
        if base_model is None and hasattr(clf_model, "calibrated_classifiers_"):
            calibrated = getattr(clf_model, "calibrated_classifiers_", [])
            if calibrated:
                base_model = getattr(calibrated[0], "estimator", None)

        if base_model is not None and hasattr(base_model, "feature_importances_"):
            importances = sorted(
                zip(self._feature_cols, base_model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )
            log.info("Top 10 features mais importantes:")
            for feat_name, imp in importances[:10]:
                log.info("  %-20s %.4f", feat_name, imp)
        else:
            log.info("Feature importance indisponível para o classificador calibrado atual.")


# =============================================================================
# 3. Decision Adjuster — integração com o decision_engine
# =============================================================================

@dataclass
class AdjustmentResult:
    """Resultado do ajuste ML para uma situação específica."""
    factor:        float   # multiplicador do EV [0.5–1.5]
    win_prob:      float   # probabilidade de vitória [0.0–1.0]
    confidence:    str     # "high" | "medium" | "low" | "no_model"
    samples_used:  int     # amostras similares no histórico
    insight:       str     # explicação em texto


class DecisionAdjuster:
    """
    Interface entre o ML e o decision_engine.

    Uso no decision_engine.py:
        adjuster = DecisionAdjuster()
        result   = adjuster.adjust(context_dict)
        ev_final = ev_matematico * result.factor
    """

    def __init__(self) -> None:
        self._fe      = FeatureEngineer()
        self._ev_pipe  = None
        self._clf_pipe = None
        self._ev_pipe_by_bucket: dict[str, object] = {}
        self._clf_pipe_by_bucket: dict[str, object] = {}
        self._metadata = {}
        self._loaded   = False
        self._load_models()

    def _bucket_from_context(self, context: dict) -> str:
        try:
            m_ratio = float(context.get("m_ratio", 0) or 0)
        except Exception:
            m_ratio = 0.0
        if m_ratio < 20:
            return "LOW"
        if m_ratio < 50:
            return "MID"
        return "HIGH"

    def _load_models(self) -> None:
        """Carrega modelos serializados se disponíveis."""
        if MODEL_EV_ADJUSTER.exists() and MODEL_RESULT_CLF.exists():
            try:
                self._ev_pipe  = joblib.load(MODEL_EV_ADJUSTER)
                self._clf_pipe = joblib.load(MODEL_RESULT_CLF)
                for bucket in STACK_BUCKETS:
                    ev_path = MODEL_EV_SEGMENT[bucket]
                    clf_path = MODEL_CLF_SEGMENT[bucket]
                    if ev_path.exists() and clf_path.exists():
                        self._ev_pipe_by_bucket[bucket] = joblib.load(ev_path)
                        self._clf_pipe_by_bucket[bucket] = joblib.load(clf_path)
                if MODEL_METADATA.exists():
                    self._metadata = json.loads(MODEL_METADATA.read_text())
                self._loaded = True
                log.info("Modelos ML carregados. Confiança: %s (%d amostras)",
                         self._metadata.get("confidence", "?"),
                         self._metadata.get("n_samples", 0))
            except Exception as e:
                log.warning("Erro ao carregar modelos ML: %s", e)
                self._loaded = False
        else:
            log.info("Modelos ML não encontrados — rodando sem ajuste adaptativo.")

    def adjust(self, context: dict) -> AdjustmentResult:
        """
        Calcula o fator de ajuste para uma situação de jogo.

        Parâmetros
        ----------
        context : dict com as chaves:
            hero_position, hero_stack_start, big_blind, m_ratio,
            hero_vpip, hero_pfr, hero_aggressor, hero_went_allin,
            board_flop, num_players, hero_cards, level,
            went_to_showdown, hero_action_preflop

        Retorna
        -------
        AdjustmentResult com fator, confiança e insight
        """
        if not self._loaded:
            return AdjustmentResult(
                factor=1.0,
                win_prob=0.5,
                confidence="no_model",
                samples_used=0,
                insight="Modelo não treinado — usando GTO puro.",
            )

        # Constrói DataFrame de uma linha
        row = pd.DataFrame([context])
        feat = self._fe.transform(row)

        feature_cols = self._metadata.get("features", [])
        if not feature_cols:
            return AdjustmentResult(1.0, 0.5, "low", 0, "Metadados ausentes.")

        # Garante que todas as colunas existem
        for col in feature_cols:
            if col not in feat.columns:
                feat[col] = 0

        X = feat[feature_cols].values

        bucket = self._bucket_from_context(context)
        ev_model = self._ev_pipe_by_bucket.get(bucket, self._ev_pipe)
        clf_model = self._clf_pipe_by_bucket.get(bucket, self._clf_pipe)

        # Predição
        try:
            ev_pred  = float(ev_model.predict(X)[0])
            win_prob = float(clf_model.predict_proba(X)[0][1])
        except Exception as e:
            log.debug("Erro na predição ML: %s", e)
            return AdjustmentResult(1.0, 0.5, "low", 0, f"Erro na predição: {e}")

        # Calcula fator de ajuste
        # EV previsto positivo → fator > 1.0 (reforça a decisão)
        # EV previsto negativo → fator < 1.0 (desconta a decisão)
        factor = self._ev_to_factor(ev_pred, context.get("big_blind", 1))

        # Confiança
        seg_meta = self._metadata.get("segmented_training", {}).get(bucket, {})
        n_samples  = int(seg_meta.get("n_samples", self._metadata.get("n_samples", 0)) or 0)
        confidence = str(seg_meta.get("confidence", self._metadata.get("confidence", "low")))

        # Abstencao: baixa confianca ou sinal estatisticamente neutro.
        if confidence == "low" or n_samples < 120:
            return AdjustmentResult(
                factor=1.0,
                win_prob=round(win_prob, 3),
                confidence="abstain",
                samples_used=n_samples,
                insight=f"ML abstain no bucket {bucket}: baixa confianca ({n_samples} amostras).",
            )

        if abs(win_prob - 0.5) <= 0.06 and 0.95 <= factor <= 1.05:
            return AdjustmentResult(
                factor=1.0,
                win_prob=round(win_prob, 3),
                confidence="abstain",
                samples_used=n_samples,
                insight=f"ML abstain no bucket {bucket}: sinal neutro.",
            )

        # Insight contextual
        insight = self._build_insight(
            factor, win_prob, context, confidence, n_samples
        )

        return AdjustmentResult(
            factor=round(factor, 3),
            win_prob=round(win_prob, 3),
            confidence=confidence,
            samples_used=n_samples,
            insight=f"[{bucket}] {insight}",
        )

    def _ev_to_factor(self, ev_pred: float, big_blind: float) -> float:
        """
        Converte EV previsto em fator multiplicativo [0.5–1.5].

        Neutro (fator=1.0) quando EV previsto ≈ 0.
        Positivo (fator>1.0) quando modelo prevê lucro.
        Negativo (fator<1.0) quando modelo prevê perda.
        """
        bb = max(big_blind, 1)
        # Normaliza: ±5bb → ±0.5 de fator
        normalized = ev_pred / (bb * 10)
        factor = 1.0 + max(-0.5, min(0.5, normalized))
        return factor

    def _build_insight(
        self,
        factor: float,
        win_prob: float,
        context: dict,
        confidence: str,
        n_samples: int,
    ) -> str:
        """Gera insight textual sobre o ajuste."""
        pos = context.get("hero_position", "?")
        parts = []

        if factor > 1.15:
            parts.append(f"Situação historicamente lucrativa no {pos}")
        elif factor < 0.85:
            parts.append(f"Situação historicamente deficitária no {pos}")
        else:
            parts.append(f"Situação neutra no {pos}")

        if win_prob > 0.60:
            parts.append(f"probabilidade de vitória alta ({win_prob:.0%})")
        elif win_prob < 0.40:
            parts.append(f"probabilidade de vitória baixa ({win_prob:.0%})")

        if confidence == "low":
            parts.append(f"baixa confiança ({n_samples} amostras)")
        elif confidence == "high":
            parts.append(f"alta confiança ({n_samples} amostras)")

        return " | ".join(parts)


# =============================================================================
# 4. Opponent Profiler
# =============================================================================

class OpponentProfiler:
    """
    Classifica oponentes por estilo de jogo baseado em VPIP/PFR.

    Perfis:
      Nit       : VPIP < 15%, PFR < 10%
      Tight-AG  : VPIP 15-25%, PFR 12-20%
      TAG       : VPIP 20-30%, PFR 15-25%  ← padrão vencedor
      LAG       : VPIP 30-40%, PFR 25-35%
      Loose-P   : VPIP > 35%, PFR < 15%   ← call station
      Maniac    : VPIP > 45%, PFR > 35%
    """

    PROFILES = {
        "Nit":      {"vpip": (0.00, 0.15), "pfr": (0.00, 0.10)},
        "Tight-AG": {"vpip": (0.15, 0.25), "pfr": (0.12, 0.20)},
        "TAG":      {"vpip": (0.20, 0.30), "pfr": (0.15, 0.25)},
        "LAG":      {"vpip": (0.30, 0.42), "pfr": (0.25, 0.38)},
        "Loose-P":  {"vpip": (0.35, 1.00), "pfr": (0.00, 0.15)},
        "Maniac":   {"vpip": (0.45, 1.00), "pfr": (0.35, 1.00)},
    }

    def classify(self, vpip: float, pfr: float) -> str:
        """Retorna o perfil mais próximo dado VPIP e PFR."""
        best_profile = "TAG"
        best_dist    = float("inf")

        for name, bounds in self.PROFILES.items():
            vpip_mid = (bounds["vpip"][0] + bounds["vpip"][1]) / 2
            pfr_mid  = (bounds["pfr"][0]  + bounds["pfr"][1])  / 2
            dist = ((vpip - vpip_mid) ** 2 + (pfr - pfr_mid) ** 2) ** 0.5
            if dist < best_dist:
                best_dist    = dist
                best_profile = name

        return best_profile

    def profile_hero(self, df: pd.DataFrame) -> dict:
        """Gera perfil completo do hero baseado no histórico."""
        if df.empty:
            return {}

        vpip = df["hero_vpip"].mean()
        pfr  = df["hero_pfr"].mean()

        profile = self.classify(vpip, pfr)

        # Desvios em relação ao GTO por posição
        desvios = {}
        for pos, benchmarks in GTO_BENCHMARKS.items():
            pos_df = df[df["hero_position"] == pos]
            if len(pos_df) < 10:
                continue
            vpip_real = pos_df["hero_vpip"].mean()
            pfr_real  = pos_df["hero_pfr"].mean()
            desvios[pos] = {
                "vpip_real": round(vpip_real * 100, 1),
                "vpip_gto":  round(benchmarks["vpip"] * 100, 1),
                "vpip_diff": round((vpip_real - benchmarks["vpip"]) * 100, 1),
                "pfr_real":  round(pfr_real * 100, 1),
                "pfr_gto":   round(benchmarks["pfr"] * 100, 1),
                "pfr_diff":  round((pfr_real - benchmarks["pfr"]) * 100, 1),
            }

        return {
            "perfil":     profile,
            "vpip_geral": round(vpip * 100, 1),
            "pfr_geral":  round(pfr * 100, 1),
            "total_maos": len(df),
            "desvios_gto": desvios,
        }


# =============================================================================
# 5. MLEngine — interface unificada
# =============================================================================

class MLEngine:
    """
    Interface principal — usada pelo decision_engine.py.

    Uso:
        engine = MLEngine()
        result = engine.adjust(context)
        ev_final = ev_matematico * result.factor
    """

    _instance: Optional["MLEngine"] = None

    def __new__(cls) -> "MLEngine":
        """Singleton — carrega modelos uma única vez."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._adjuster  = DecisionAdjuster()
            cls._instance._profiler  = OpponentProfiler()
        return cls._instance

    def adjust(self, context: dict) -> AdjustmentResult:
        """Ajusta a decisão baseado no histórico do hero."""
        return self._adjuster.adjust(context)

    def profile_hero(self, df: pd.DataFrame) -> dict:
        """Gera perfil do hero."""
        return self._profiler.profile_hero(df)

    @property
    def is_trained(self) -> bool:
        return self._adjuster._loaded

    @property
    def confidence(self) -> str:
        return self._adjuster._metadata.get("confidence", "no_model")

    @property
    def n_samples(self) -> int:
        return self._adjuster._metadata.get("n_samples", 0)


# =============================================================================
# 6. Carregador de dados
# =============================================================================

def load_data(source: str) -> pd.DataFrame:
    """
    Carrega dados de treinamento.
    Aceita: CSV (hands.csv) ou conexão PostgreSQL.
    """
    path = Path(source)

    if path.exists() and path.suffix == ".csv":
        log.info("Carregando dados do CSV: %s", source)
        return pd.read_csv(source)

    # Tenta PostgreSQL
    try:
        import psycopg2
        import json as _json
        cfg_path = _BASE_DIR / "db_config.json"
        if cfg_path.exists():
            cfg = _json.loads(cfg_path.read_text())
        else:
            cfg = {
                "host": "localhost", "port": 5432,
                "database": "poker_dss", "user": "postgres",
                "password": "sua_senha_aqui"
            }

        log.info("Carregando dados do PostgreSQL...")
        conn = psycopg2.connect(**cfg)
        df   = pd.read_sql("SELECT * FROM hands ORDER BY date_utc", conn)
        conn.close()
        log.info("  %d mãos carregadas do banco.", len(df))
        return df

    except Exception as e:
        log.error("Não foi possível carregar dados: %s", e)
        return pd.DataFrame()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poker DSS ML Engine")
    parser.add_argument("--train",    action="store_true", help="Treina os modelos")
    parser.add_argument("--evaluate", action="store_true", help="Avalia os modelos")
    parser.add_argument("--report",   action="store_true", help="Relatório do perfil do hero")
    parser.add_argument("--predict",  action="store_true", help="Modo interativo de teste")
    parser.add_argument(
        "--data", type=str,
        default=str(_BASE_DIR / "hands.csv"),
        help="Fonte de dados: caminho do CSV ou 'postgres'"
    )
    args = parser.parse_args()

    if (args.train or args.evaluate or args.predict) and (not _SKLEARN_AVAILABLE or not _JOBLIB_AVAILABLE):
        missing = []
        if not _SKLEARN_AVAILABLE:
            missing.append(f"scikit-learn: {_SKLEARN_IMPORT_ERROR or 'módulo não encontrado'}")
        if not _JOBLIB_AVAILABLE:
            missing.append(f"joblib: {_JOBLIB_IMPORT_ERROR or 'módulo não encontrado'}")
        log.error(
            "Dependências de ML indisponíveis neste ambiente: %s. "
            "Instale com: python -m pip install scikit-learn joblib",
            "; ".join(missing),
        )
        sys.exit(1)

    df = load_data(args.data)

    if df.empty:
        log.error("Nenhum dado disponível.")
        sys.exit(1)

    log.info("Dados carregados: %d mãos", len(df))

    if args.train:
        trainer = ModelTrainer()
        metadata = trainer.train(df)
        print("\n=== Resultado do Treinamento ===")
        print(f"  Amostras:   {metadata['n_samples']}")
        print(f"  EV MAE:     {metadata['ev_mae']}")
        print(f"  EV R²:      {metadata['ev_r2']}")
        print(f"  Win F1:     {metadata['win_f1']}")
        print(f"  Confiança:  {metadata['confidence']}")

    elif args.evaluate:
        trainer = ModelTrainer()
        trainer.evaluate(df)

    elif args.report:
        engine  = MLEngine()
        profile = engine.profile_hero(df)
        print("\n=== Perfil do Hero ===")
        print(f"  Perfil:     {profile.get('perfil', '?')}")
        print(f"  VPIP geral: {profile.get('vpip_geral', '?')}%")
        print(f"  PFR geral:  {profile.get('pfr_geral', '?')}%")
        print(f"  Total mãos: {profile.get('total_maos', 0)}")
        print()
        print("  Desvios vs GTO por posição:")
        for pos, d in profile.get("desvios_gto", {}).items():
            vpip_arrow = "↑" if d["vpip_diff"] > 1 else ("↓" if d["vpip_diff"] < -1 else "≈")
            pfr_arrow  = "↑" if d["pfr_diff"]  > 1 else ("↓" if d["pfr_diff"]  < -1 else "≈")
            print(f"  {pos:<6} VPIP {d['vpip_real']}% {vpip_arrow} (GTO {d['vpip_gto']}%)  "
                  f"PFR {d['pfr_real']}% {pfr_arrow} (GTO {d['pfr_gto']}%)")

    elif args.predict:
        if not MODEL_EV_ADJUSTER.exists():
            print("Modelo não treinado. Execute --train primeiro.")
            sys.exit(1)

        engine = MLEngine()
        print("=== Modo Predict — Teste de Ajuste ===")
        print("Exemplo de contexto:")
        ctx = {
            "hero_position":    "BTN",
            "hero_stack_start": 5000,
            "big_blind":        100,
            "m_ratio":          25.0,
            "hero_vpip":        1,
            "hero_pfr":         1,
            "hero_aggressor":   0,
            "hero_went_allin":  0,
            "board_flop":       "",
            "num_players":      6,
            "hero_cards":       "AKs",
            "level":            5,
            "went_to_showdown": 0,
            "hero_action_preflop": "raises",
        }
        result = engine.adjust(ctx)
        print(f"  Fator de ajuste: ×{result.factor}")
        print(f"  Prob. vitória:   {result.win_prob:.1%}")
        print(f"  Confiança:       {result.confidence}")
        print(f"  Insight:         {result.insight}")

    else:
        parser.print_help()
