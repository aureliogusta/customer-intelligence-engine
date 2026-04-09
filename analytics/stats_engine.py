"""
stats_engine.py
===============
Core Data Manipulation — Roadmap Block 1 (parte 2).

Consome o DataFrame do hand_history_parser e produz:
  1. Estatísticas globais do hero (VPIP, PFR, AF, WTSD, W$SD, ROI)
  2. Breakdown por posição
  3. Breakdown por M-ratio (push/fold zone analysis)
  4. Leaks identificados automaticamente (comparação vs benchmarks GTO)
  5. Relatório exportado em CSV + resumo no terminal

Run:  python stats_engine.py [hands.csv]
      Sem argumento: usa /mnt/user-data/outputs/hands.csv
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stats")

_BASE_DIR = Path(__file__).parent.resolve()
_DEFAULT_INPUT_CSV = os.getenv("POKER_STATS_CSV", str(_BASE_DIR / "hands_combined_acr_ps.csv"))
_DEFAULT_OUTPUT_DIR = os.getenv("POKER_STATS_OUTPUT_DIR", str(_BASE_DIR / "outputs"))

# ── Benchmarks GTO para 6-max MTT (referência para detecção de leaks) ────────

GTO_BENCHMARKS: dict[str, dict] = {
    "UTG":   {"vpip": 0.14, "pfr": 0.13, "vpip_pfr_gap": 0.02},
    "UTG+1": {"vpip": 0.15, "pfr": 0.14, "vpip_pfr_gap": 0.02},
    "MP":    {"vpip": 0.17, "pfr": 0.15, "vpip_pfr_gap": 0.03},
    "HJ":    {"vpip": 0.20, "pfr": 0.18, "vpip_pfr_gap": 0.03},
    "CO":    {"vpip": 0.26, "pfr": 0.23, "vpip_pfr_gap": 0.04},
    "BTN":   {"vpip": 0.45, "pfr": 0.38, "vpip_pfr_gap": 0.07},
    "SB":    {"vpip": 0.35, "pfr": 0.28, "vpip_pfr_gap": 0.08},
    "BB":    {"vpip": 0.40, "pfr": 0.10, "vpip_pfr_gap": 0.30},
}

# Limiares para alertas de leak
LEAK_THRESHOLDS = {
    "vpip_too_low":   0.70,   # % do benchmark — abaixo = muito tight
    "vpip_too_high":  1.40,   # % do benchmark — acima = muito loose
    "pfr_too_low":    0.65,   # % do benchmark
    "pfr_vpip_gap":   0.15,   # gap > 15pp = call station
    "fold_pct_high":  0.92,   # > 92% de fold geral = nit extremo
    "allin_loss_pct": 0.50,   # > 50% all-ins resultam em derrota
}

M_RATIO_ZONES = {
    "green":  (20, 999),   # stack profundo — jogo normal
    "yellow": (10, 20),    # pressão crescente
    "orange": (6,  10),    # push/fold parcial
    "red":    (0,   6),    # push/fold total
}


class StatsEngine:
    """
    Calcula estatísticas completas de desempenho do hero.

    Uso:
        engine = StatsEngine()
        engine.load("hands.csv")
        report = engine.compute()
        engine.print_report(report)
        engine.export(report)
    """

    def __init__(self) -> None:
        self.df: pd.DataFrame = pd.DataFrame()

    def load(self, csv_path: str) -> None:
        """Carrega DataFrame do CSV gerado pelo ParserPipeline."""
        self.df = pd.read_csv(csv_path, parse_dates=["date_utc"], low_memory=False)
        log.info("Carregadas %d mãos de %s", len(self.df), csv_path)

    def load_df(self, df: pd.DataFrame) -> None:
        """Carrega diretamente de um DataFrame em memória."""
        self.df = df.copy()

    def compute(self) -> dict:
        """
        Computa todas as estatísticas e retorna um dict estruturado.
        """
        df = self.df
        if df.empty:
            return {}

        report = {
            "summary":        self._global_stats(df),
            "by_position":    self._by_position(df),
            "by_m_ratio":     self._by_m_ratio(df),
            "by_tournament":  self._by_tournament(df),
            "leaks":          self._detect_leaks(df),
            "session_trend":  self._session_trend(df),
        }
        return report

    # ── Estatísticas globais ─────────────────────────────────────────────────

    def _global_stats(self, df: pd.DataFrame) -> dict:
        total       = len(df)
        vpip_hands  = df["hero_vpip"].sum()
        pfr_hands   = df["hero_pfr"].sum()
        agg_hands   = df["hero_aggressor"].sum()
        fold_hands  = (df["hero_result"] == "fold").sum()
        win_hands   = (df["hero_result"] == "win").sum()
        sd_hands    = df["went_to_showdown"].sum()
        won_at_sd   = ((df["went_to_showdown"] == 1) & (df["hero_result"] == "win")).sum()
        allin_hands = df["hero_went_allin"].sum()
        allin_loss  = (df["hero_went_allin"] == 1) & (df["hero_result"] == "allin_loss")

        vpip_pct = vpip_hands / total if total else 0
        pfr_pct  = pfr_hands  / total if total else 0

        # Aggression Factor = (bets + raises) / calls
        # Aproximado via flags — refinado na v2 com parsing de amount
        af = agg_hands / max((vpip_hands - pfr_hands), 1)

        # WTSD: Went To Showdown dado que viu o flop
        flop_seen = (~df["board_flop"].isna() & (df["board_flop"] != "") &
                     (df["hero_action_preflop"].str.contains("calls|raises", na=False) |
                      (df["hero_position"] == "BB"))).sum()
        wtsd = sd_hands / max(flop_seen, 1)
        w_sd = won_at_sd / max(sd_hands, 1)

        return {
            "total_hands":      total,
            "vpip_pct":         round(vpip_pct * 100, 1),
            "pfr_pct":          round(pfr_pct * 100, 1),
            "vpip_pfr_gap":     round((vpip_pct - pfr_pct) * 100, 1),
            "aggression_factor": round(af, 2),
            "fold_pct":         round(fold_hands / total * 100, 1),
            "win_pct":          round(win_hands / total * 100, 1),
            "wtsd_pct":         round(wtsd * 100, 1),
            "w_at_sd_pct":      round(w_sd * 100, 1),
            "total_wins":       int(win_hands),
            "total_folds":      int(fold_hands),
            "went_to_showdown": int(sd_hands),
            "allin_total":      int(allin_hands),
            "allin_loss_pct":   round(allin_loss.sum() / max(allin_hands, 1) * 100, 1),
            "avg_m_ratio":      round(df["m_ratio"].mean(), 2),
        }

    # ── Por posição ──────────────────────────────────────────────────────────

    def _by_position(self, df: pd.DataFrame) -> pd.DataFrame:
        pos_order = ["UTG", "UTG+1", "MP", "HJ", "CO", "BTN", "SB", "BB"]

        def pos_stats(g: pd.DataFrame) -> pd.Series:
            n    = len(g)
            vpip = g["hero_vpip"].mean() * 100
            pfr  = g["hero_pfr"].mean() * 100
            agg  = g["hero_aggressor"].mean() * 100
            folds = (g["hero_result"] == "fold").mean() * 100
            wins  = (g["hero_result"] == "win").mean() * 100
            sd    = g["went_to_showdown"].mean() * 100

            # Benchmarks GTO para esta posição
            pos  = g.name
            bm   = GTO_BENCHMARKS.get(pos, {})
            bm_vpip = bm.get("vpip", 0) * 100
            bm_pfr  = bm.get("pfr",  0) * 100

            return pd.Series({
                "hands":        n,
                "vpip%":        round(vpip, 1),
                "pfr%":         round(pfr, 1),
                "gap_vp_pfr":   round(vpip - pfr, 1),
                "aggr%":        round(agg, 1),
                "fold%":        round(folds, 1),
                "win%":         round(wins, 1),
                "wtsd%":        round(sd, 1),
                "gto_vpip%":    round(bm_vpip, 1),
                "gto_pfr%":     round(bm_pfr, 1),
                "vpip_delta":   round(vpip - bm_vpip, 1),
                "pfr_delta":    round(pfr - bm_pfr, 1),
            })

        by_pos = df.groupby("hero_position", observed=True).apply(pos_stats)

        # Reordena por posição lógica
        existing = [p for p in pos_order if p in by_pos.index]
        return by_pos.reindex(existing)

    # ── Por M-ratio ──────────────────────────────────────────────────────────

    def _by_m_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        def get_zone(m: float) -> str:
            for zone, (lo, hi) in M_RATIO_ZONES.items():
                if lo <= m < hi:
                    return zone
            return "green"

        df2 = df.copy()
        df2["m_zone"] = df2["m_ratio"].apply(get_zone)

        def zone_stats(g: pd.DataFrame) -> pd.Series:
            n     = len(g)
            vpip  = g["hero_vpip"].mean() * 100
            pfr   = g["hero_pfr"].mean() * 100
            allin = g["hero_went_allin"].mean() * 100
            wins  = (g["hero_result"] == "win").mean() * 100
            return pd.Series({
                "hands":   n,
                "vpip%":   round(vpip, 1),
                "pfr%":    round(pfr, 1),
                "allin%":  round(allin, 1),
                "win%":    round(wins, 1),
            })

        zone_order = ["green", "yellow", "orange", "red"]
        by_zone = df2.groupby("m_zone").apply(zone_stats)
        existing = [z for z in zone_order if z in by_zone.index]
        return by_zone.reindex(existing)

    # ── Por torneio ──────────────────────────────────────────────────────────

    def _by_tournament(self, df: pd.DataFrame) -> pd.DataFrame:
        def tourn_stats(g: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "hands":     len(g),
                "vpip%":     round(g["hero_vpip"].mean() * 100, 1),
                "pfr%":      round(g["hero_pfr"].mean() * 100, 1),
                "wins":      int((g["hero_result"] == "win").sum()),
                "allins":    int(g["hero_went_allin"].sum()),
                "avg_m":     round(g["m_ratio"].mean(), 1),
            })
        return df.groupby("tournament_name", observed=True).apply(tourn_stats)

    # ── Detecção automática de leaks ─────────────────────────────────────────

    def _detect_leaks(self, df: pd.DataFrame) -> list[dict]:
        leaks = []
        total = len(df)

        by_pos = self._by_position(df)

        for pos, row in by_pos.iterrows():
            bm = GTO_BENCHMARKS.get(pos, {})
            if not bm or row["hands"] < 10:   # mínimo de 10 mãos para ser relevante
                continue

            bm_vpip = bm["vpip"] * 100
            bm_pfr  = bm["pfr"] * 100

            # VPIP muito baixo (muito tight)
            if row["vpip%"] < bm_vpip * LEAK_THRESHOLDS["vpip_too_low"]:
                leaks.append({
                    "tipo":     "VPIP Baixo",
                    "posição":  pos,
                    "valor":    f"{row['vpip%']}%",
                    "benchmark": f"{bm_vpip}%",
                    "severidade": "ALTA" if row["vpip%"] < bm_vpip * 0.5 else "MÉDIA",
                    "sugestão": f"Você está foldando muito em {pos}. "
                                f"GTO sugere {bm_vpip}% de VPIP."
                })

            # VPIP muito alto (muito loose)
            if row["vpip%"] > bm_vpip * LEAK_THRESHOLDS["vpip_too_high"]:
                leaks.append({
                    "tipo":     "VPIP Alto",
                    "posição":  pos,
                    "valor":    f"{row['vpip%']}%",
                    "benchmark": f"{bm_vpip}%",
                    "severidade": "ALTA",
                    "sugestão": f"Você está entrando em muitos potes em {pos}. "
                                f"Reduza para ~{bm_vpip}%."
                })

            # Gap VPIP-PFR alto (call station)
            if row["gap_vp_pfr"] > LEAK_THRESHOLDS["pfr_vpip_gap"] * 100:
                leaks.append({
                    "tipo":     "Call Station",
                    "posição":  pos,
                    "valor":    f"gap {row['gap_vp_pfr']}pp",
                    "benchmark": f"≤ {LEAK_THRESHOLDS['pfr_vpip_gap']*100:.0f}pp",
                    "severidade": "MÉDIA",
                    "sugestão": f"Em {pos} você chama mais do que aposta. "
                                f"Converta calls em 3-bets ou folds."
                })

            # PFR muito baixo
            if row["pfr%"] < bm_pfr * LEAK_THRESHOLDS["pfr_too_low"] and row["vpip%"] > 0:
                leaks.append({
                    "tipo":     "PFR Baixo",
                    "posição":  pos,
                    "valor":    f"{row['pfr%']}%",
                    "benchmark": f"{bm_pfr}%",
                    "severidade": "MÉDIA",
                    "sugestão": f"Você raramente abre/3-beta em {pos}. "
                                f"Seja mais agressivo no pré-flop."
                })

        # Fold global muito alto
        fold_pct = (df["hero_result"] == "fold").mean()
        if fold_pct > LEAK_THRESHOLDS["fold_pct_high"]:
            leaks.append({
                "tipo":     "Fold Rate Global",
                "posição":  "GERAL",
                "valor":    f"{fold_pct*100:.1f}%",
                "benchmark": "< 90%",
                "severidade": "ALTA",
                "sugestão": "Você está foldando demais no geral. "
                            "Revise sua range de defesa no BB e steals."
            })

        # All-in loss rate
        allin = df["hero_went_allin"].sum()
        if allin >= 3:
            allin_loss_pct = (
                (df["hero_went_allin"] == 1) & (df["hero_result"] == "allin_loss")
            ).sum() / allin
            if allin_loss_pct > LEAK_THRESHOLDS["allin_loss_pct"]:
                leaks.append({
                    "tipo":     "All-In Bad Spots",
                    "posição":  "GERAL",
                    "valor":    f"{allin_loss_pct*100:.0f}% perda nos all-ins",
                    "benchmark": "< 50%",
                    "severidade": "ALTA",
                    "sugestão": "Você está indo all-in em spots ruins. "
                                "Revise suas situações de shove/call-off."
                })

        return leaks

    # ── Tendência por sessão ─────────────────────────────────────────────────

    def _session_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrupa por dia e calcula tendência de VPIP/PFR ao longo do tempo."""
        if "date_utc" not in df.columns:
            return pd.DataFrame()
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date_utc"]).dt.date
        return df2.groupby("date").agg(
            hands    = ("hand_id",    "count"),
            vpip_pct = ("hero_vpip",  lambda x: round(x.mean() * 100, 1)),
            pfr_pct  = ("hero_pfr",   lambda x: round(x.mean() * 100, 1)),
            wins     = ("hero_result", lambda x: (x == "win").sum()),
            allins   = ("hero_went_allin", "sum"),
        ).reset_index()

    # ── Print formatado ──────────────────────────────────────────────────────

    def print_report(self, report: dict) -> None:
        SEP  = "=" * 65
        SEP2 = "─" * 65

        print(f"\n{SEP}")
        print(f"{'  POKER DSS — STATS ENGINE':^65}")
        print(SEP)

        # Global
        s = report.get("summary", {})
        print(f"\n  {'RESUMO GLOBAL':}")
        print(SEP2)
        print(f"  Total de mãos analisadas : {s.get('total_hands', 0)}")
        print(f"  VPIP                     : {s.get('vpip_pct', 0)}%")
        print(f"  PFR                      : {s.get('pfr_pct', 0)}%")
        print(f"  Gap VPIP-PFR             : {s.get('vpip_pfr_gap', 0)}pp")
        print(f"  Aggression Factor        : {s.get('aggression_factor', 0)}")
        print(f"  Fold Rate                : {s.get('fold_pct', 0)}%")
        print(f"  Win Rate (mãos)          : {s.get('win_pct', 0)}%")
        print(f"  WTSD                     : {s.get('wtsd_pct', 0)}%")
        print(f"  W@SD                     : {s.get('w_at_sd_pct', 0)}%")
        print(f"  All-ins                  : {s.get('allin_total', 0)}"
              f"  (perda: {s.get('allin_loss_pct', 0)}%)")
        print(f"  M-ratio médio            : {s.get('avg_m_ratio', 0)}")

        # Por posição
        by_pos = report.get("by_position", pd.DataFrame())
        if not by_pos.empty:
            print(f"\n  {'POR POSIÇÃO':}")
            print(SEP2)
            cols = ["hands", "vpip%", "pfr%", "gap_vp_pfr",
                    "fold%", "win%", "gto_vpip%", "vpip_delta", "pfr_delta"]
            cols = [c for c in cols if c in by_pos.columns]
            print(by_pos[cols].to_string())

        # Por M-ratio
        by_m = report.get("by_m_ratio", pd.DataFrame())
        if not by_m.empty:
            print(f"\n  {'POR M-RATIO (ZONA DE STACK)':}")
            print(SEP2)
            print(f"  green=20+  yellow=10-20  orange=6-10  red=<6")
            print(by_m.to_string())

        # Tendência
        trend = report.get("session_trend", pd.DataFrame())
        if not trend.empty:
            print(f"\n  {'TENDÊNCIA POR SESSÃO':}")
            print(SEP2)
            print(trend.to_string(index=False))

        # Leaks
        leaks = report.get("leaks", [])
        print(f"\n  {'LEAKS DETECTADOS':}  ({len(leaks)} encontrados)")
        print(SEP2)
        if not leaks:
            print("  ✓  Nenhum leak crítico detectado com o volume atual.")
        else:
            for i, leak in enumerate(leaks, 1):
                sev_icon = "🔴" if leak["severidade"] == "ALTA" else "🟡"
                print(f"\n  {i}. {sev_icon} [{leak['severidade']}] {leak['tipo']} — {leak['posição']}")
                print(f"     Seu valor  : {leak['valor']}")
                print(f"     Benchmark  : {leak['benchmark']}")
                print(f"     Sugestão   : {leak['sugestão']}")

        print(f"\n{SEP}\n")

    # ── Export ───────────────────────────────────────────────────────────────

    def export(self, report: dict, output_dir: str = _DEFAULT_OUTPUT_DIR) -> None:
        """Exporta cada seção do relatório como CSV separado."""
        os.makedirs(output_dir, exist_ok=True)
        for section, data in report.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                path = os.path.join(output_dir, f"stats_{section}.csv")
                data.to_csv(path)
                log.info("Exportado: %s", path)
            elif isinstance(data, dict):
                path = os.path.join(output_dir, f"stats_{section}.csv")
                pd.DataFrame([data]).to_csv(path, index=False)
                log.info("Exportado: %s", path)
            elif isinstance(data, list) and data:
                path = os.path.join(output_dir, f"stats_{section}.csv")
                pd.DataFrame(data).to_csv(path, index=False)
                log.info("Exportado: %s", path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_INPUT_CSV

    if not os.path.exists(csv_path):
        # Roda o parser primeiro se não houver CSV
        log.info("CSV não encontrado. Rodando o parser primeiro...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from hand_history_parser import ParserPipeline

        candidate_inputs = [
            os.getenv("POKER_HANDS_INPUT_DIR", ""),
            r"C:\Program Files (x86)\ACR Poker\handHistory",
            str(Path.home() / "AppData" / "Local" / "PokerStars.ES" / "HandHistory"),
            "/mnt/user-data/uploads",
        ]
        input_dir = next((p for p in candidate_inputs if p and os.path.exists(p)), "")
        if not input_dir:
            raise FileNotFoundError(
                "Nenhum diretório de hand history encontrado. "
                "Defina POKER_HANDS_INPUT_DIR ou informe CSV manualmente."
            )

        pipeline = ParserPipeline()
        df = pipeline.run(input_dir)
        pipeline.export(df, csv_path)

    engine = StatsEngine()
    engine.load(csv_path)
    report = engine.compute()
    engine.print_report(report)
    engine.export(report)
