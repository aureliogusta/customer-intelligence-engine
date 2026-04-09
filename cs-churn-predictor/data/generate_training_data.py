"""
generate_training_data.py
=========================
Lê (ou gera) contas sintéticas e produz um dataset com labels de churn/upsell.

Pipeline:
  1. Tenta carregar cs-health-dashboard/data/contas_sinteticas.csv
  2. Se não existir, gera os dados sintéticos inline
  3. Agrega por conta (últimos 30 dias vs 31–60 dias)
  4. Calcula features derivadas (trends) e labels realistas
  5. Salva data/train_dataset.csv  (500 linhas × ~16 colunas)

Run:
    python data/generate_training_data.py
"""

from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DATA = ROOT.parent / "cs-health-dashboard" / "data"
OUTPUT_PATH    = ROOT / "data" / "train_dataset.csv"

N_ACCOUNTS = 500
DAYS       = 60

# ── Perfis internos (usados para label realista) ──────────────────────────────
PROFILES = {
    "healthy":  {"weight": 0.40, "base_login": 0.85, "base_nps": 8.0, "ticket": 0.05, "renewal_p": 0.95, "upsell_p": 0.30},
    "at_risk":  {"weight": 0.25, "base_login": 0.35, "base_nps": 5.0, "ticket": 0.20, "renewal_p": 0.35, "upsell_p": 0.05},
    "churning": {"weight": 0.15, "base_login": 0.10, "base_nps": 2.5, "ticket": 0.40, "renewal_p": 0.08, "upsell_p": 0.01},
    "improving":{"weight": 0.10, "base_login": 0.50, "base_nps": 6.5, "ticket": 0.10, "renewal_p": 0.80, "upsell_p": 0.15},
    "new":      {"weight": 0.10, "base_login": 0.65, "base_nps": 7.0, "ticket": 0.15, "renewal_p": 0.80, "upsell_p": 0.10},
}

SEGMENTS = {
    "ENTERPRISE": {"pct": 0.20, "mrr_range": (5000, 50000), "users_range": (20, 200)},
    "MID_MARKET": {"pct": 0.50, "mrr_range": (1000,  5000), "users_range": (5,   50)},
    "SMB":        {"pct": 0.30, "mrr_range": (200,   1000), "users_range": (1,   10)},
}


# ── Gerador inline (fallback) ─────────────────────────────────────────────────

def _pick(d: dict) -> str:
    keys    = list(d.keys())
    weights = [d[k].get("weight", d[k].get("pct", 1)) for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def _simulate_daily(profile: str, max_users: int, start_date: date) -> list[dict]:
    p = PROFILES[profile]
    rows: list[dict] = []
    for day_idx in range(DAYS):
        ev_date  = start_date + timedelta(days=day_idx)
        progress = day_idx / DAYS

        if profile == "improving":
            login_rate = p["base_login"] * (0.5 + progress * 0.8)
            nps_base   = p["base_nps"]  * (0.7 + progress * 0.5)
            ticket_m   = p["ticket"]    * (1.5 - progress)
        elif profile == "churning":
            login_rate = p["base_login"] * (1.0 - progress * 0.7)
            nps_base   = p["base_nps"]   * (1.0 - progress * 0.4)
            ticket_m   = p["ticket"]     * (1.0 + progress * 2.0)
        else:
            login_rate = p["base_login"]
            nps_base   = p["base_nps"]
            ticket_m   = p["ticket"]

        login_rate = float(np.clip(login_rate + np.random.normal(0, 0.08), 0.0, 1.0))
        nps_base   = float(np.clip(nps_base   + np.random.normal(0, 0.50), 1.0, 10.0))

        logins       = max(1, int(np.random.poisson(max_users * login_rate * 0.6))) if random.random() < login_rate else 0
        active_users = min(logins, max_users)
        tickets_open = int(np.random.poisson(max(0.1, max_users * ticket_m * 0.1)))
        nps          = round(float(np.clip(nps_base + np.random.normal(0, 0.3), 1.0, 10.0)), 1) if random.random() < 0.20 else None

        rows.append({
            "event_date":   ev_date.isoformat(),
            "logins_count": logins,
            "active_users": active_users,
            "tickets_open": max(0, tickets_open),
            "nps_score":    nps,
        })
    return rows


def _generate_synthetic() -> pd.DataFrame:
    """Gera contas sintéticas do zero (independente do BLOCO 1)."""
    start = date.today() - timedelta(days=DAYS - 1)
    records: list[dict] = []

    for i in range(1, N_ACCOUNTS + 1):
        account_id = f"ACC_{i:06d}"
        profile    = _pick(PROFILES)
        segment    = _pick(SEGMENTS)
        seg        = SEGMENTS[segment]
        max_users  = random.randint(*seg["users_range"])
        mrr        = round(random.uniform(*seg["mrr_range"]), 2)
        contract_start = start - timedelta(days=random.randint(30, 730))

        daily = _simulate_daily(profile, max_users, start)
        for row in daily:
            records.append({
                "account_id":     account_id,
                "segment":        segment,
                "mrr":            mrr,
                "max_users":      max_users,
                "contract_start": contract_start.isoformat(),
                "profile":        profile,   # label helper
                **row,
            })

    return pd.DataFrame(records)


# ── Feature engineering ───────────────────────────────────────────────────────

def _aggregate(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """Agrega interações diárias por conta → 1 linha / conta com features."""
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    today_ts         = pd.Timestamp(today)

    cutoff_30 = today_ts - pd.Timedelta(days=30)
    cutoff_60 = today_ts - pd.Timedelta(days=61)

    recent  = df[df["event_date"] > cutoff_30]
    prior   = df[(df["event_date"] > cutoff_60) & (df["event_date"] <= cutoff_30)]

    def agg_period(sub: pd.DataFrame) -> pd.DataFrame:
        return sub.groupby("account_id").agg(
            login_days   = ("logins_count",  lambda x: (x > 0).sum()),
            total_logins = ("logins_count",  "sum"),
            avg_users    = ("active_users",  "mean"),
            tickets_sum  = ("tickets_open",  "sum"),
            nps_mean     = ("nps_score",     "mean"),
            last_login   = ("event_date",    "max"),
        ).reset_index()

    r = agg_period(recent).add_suffix("_30").rename(columns={"account_id_30": "account_id"})
    p = agg_period(prior).add_suffix("_60").rename(columns={"account_id_60": "account_id"})

    meta_cols = ["account_id", "segment", "mrr", "max_users", "contract_start"]
    if "profile" in df.columns:
        meta_cols.append("profile")
    meta = df[meta_cols].drop_duplicates("account_id")

    merged = meta.merge(r, on="account_id", how="left").merge(p, on="account_id", how="left")
    merged = merged.fillna({
        "login_days_30": 0, "total_logins_30": 0, "avg_users_30": 0,
        "tickets_sum_30": 0, "nps_mean_30": 5.0,
        "login_days_60": 0, "total_logins_60": 0, "avg_users_60": 0,
        "tickets_sum_60": 0, "nps_mean_60": 5.0,
    })

    # Features principais
    merged["engajamento_pct"]   = (merged["login_days_30"] / 30 * 100).round(1)
    merged["nps_score"]         = merged["nps_mean_30"].round(1)
    merged["tickets_abertos"]   = merged["tickets_sum_30"].astype(int)
    merged["dias_no_contrato"]  = (today_ts - pd.to_datetime(merged["contract_start"])).dt.days

    # Trends  (diferença normalizada entre período recente e anterior)
    eps = 1e-6
    merged["engagement_trend"] = (
        (merged["login_days_30"] - merged["login_days_60"]) /
        (merged["login_days_60"] + eps)
    ).clip(-2, 2).round(3)

    merged["tickets_trend"] = (
        (merged["tickets_sum_30"] - merged["tickets_sum_60"]) /
        (merged["tickets_sum_60"] + eps)
    ).clip(-2, 2).round(3)

    merged["nps_trend"] = (merged["nps_mean_30"] - merged["nps_mean_60"]).round(3)

    # Dias sem interação
    merged["last_login_30"] = pd.to_datetime(merged["last_login_30"])
    merged["dias_sem_interacao"] = (today_ts - merged["last_login_30"]).dt.days.fillna(30).astype(int)

    return merged


# ── Label generation ──────────────────────────────────────────────────────────

_RENEWAL_BY_PROFILE = {
    "healthy": 0.95, "improving": 0.80, "new": 0.75,
    "at_risk": 0.35, "churning": 0.08,
}

_UPSELL_BY_PROFILE = {
    "healthy": 0.30, "improving": 0.15, "new": 0.12,
    "at_risk": 0.04, "churning": 0.01,
}


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(SEED)

    if "profile" in df.columns:
        renewal_p = df["profile"].map(_RENEWAL_BY_PROFILE).fillna(0.5)
        upsell_p  = df["profile"].map(_UPSELL_BY_PROFILE).fillna(0.05)
    else:
        # Labels derivadas das features quando não há profile
        renewal_p = (
            0.10 * (df["engajamento_pct"] / 100) +
            0.10 * (df["nps_score"] / 10) +
            -0.05 * (df["tickets_abertos"] / (df["tickets_abertos"].max() + 1)) +
            0.10 * df["engagement_trend"].clip(-1, 1) +
            0.10 * df["nps_trend"].clip(-2, 2) / 2
        ).clip(0.05, 0.95)
        upsell_p = (renewal_p * 0.3).clip(0.01, 0.40)

    # Ajuste baseado nas features (realismo)
    engagement_bonus = ((df["engajamento_pct"] - 50) / 100).clip(-0.20, 0.20)
    nps_bonus        = ((df["nps_score"] - 5) / 20).clip(-0.15, 0.15)

    renewal_p = (renewal_p + engagement_bonus + nps_bonus).clip(0.02, 0.98)

    df["renovado"]  = rng.uniform(0, 1, len(df)) < renewal_p
    df["fez_upsell"] = rng.uniform(0, 1, len(df)) < upsell_p

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def gerar_dataset_com_labels(source_csv: Path | None = None) -> pd.DataFrame:
    """
    Gera ou carrega dados brutos, agrega por conta e adiciona labels.
    Retorna DataFrame pronto para ML (500 linhas × ~16 colunas).
    """
    # 1. Carregar ou gerar dados brutos
    if source_csv and source_csv.exists():
        print(f"Carregando interações de {source_csv} ...")
        raw = pd.read_csv(source_csv, parse_dates=["event_date"])
        # Garantir coluna profile se existir em accounts_meta.csv
        meta_path = source_csv.parent / "accounts_meta.csv"
        if meta_path.exists() and "profile" not in raw.columns:
            meta = pd.read_csv(meta_path, usecols=["account_id", "profile"])
            raw  = raw.merge(meta, on="account_id", how="left")
    else:
        print("contas_sinteticas.csv não encontrado — gerando dados sintéticos inline ...")
        raw = _generate_synthetic()

    # 2. Agregar
    today = date.today()
    agg   = _aggregate(raw, today)

    # 3. Labels
    df = _add_labels(agg)

    # 4. Selecionar colunas finais
    keep = [
        "account_id", "segment", "mrr", "max_users",
        "engajamento_pct", "nps_score", "tickets_abertos", "dias_no_contrato",
        "engagement_trend", "tickets_trend", "nps_trend", "dias_sem_interacao",
        "renovado", "fez_upsell",
    ]
    if "profile" in df.columns:
        keep.insert(2, "profile")

    final = df[[c for c in keep if c in df.columns]].copy()
    return final


if __name__ == "__main__":
    source = DASHBOARD_DATA / "contas_sinteticas.csv"
    df     = gerar_dataset_com_labels(source_csv=source if source.exists() else None)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"  Dataset de treino gerado com sucesso!")
    print(f"{'='*60}")
    print(f"  Contas     : {len(df)}")
    print(f"  Colunas    : {list(df.columns)}")
    print(f"\n  Labels:")
    print(f"    renovado    : {df['renovado'].sum()} ({df['renovado'].mean():.1%})")
    print(f"    fez_upsell  : {df['fez_upsell'].sum()} ({df['fez_upsell'].mean():.1%})")
    if "profile" in df.columns:
        print(f"\n  Distribuição de perfis:")
        print(df["profile"].value_counts().to_string())
    print(f"\n  Arquivo salvo: {OUTPUT_PATH}")
    print(f"{'='*60}")
