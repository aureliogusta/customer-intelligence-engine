"""
latency_manager.py
==================
Terminal assíncrono unificado — pré-flop e pós-flop.

Formatos de input aceitos
--------------------------
  Pré-flop:  MÃO POSIÇÃO OPEN_BB STACK_BB MULTIWAY
             ex: 66 CO 2.5 35 N

  Pós-flop:  MÃO POSIÇÃO CALL_BB POT_BB MULTIWAY CARTA1 CARTA2 CARTA3 [CARTA4] [CARTA5]
             ex: AKs BTN 10 30 N Jh Td 3c
             ex: QJs HJ 8 25 N Th 9d 2c 8h

Roteamento automático: len(tokens) > 5 → pós-flop com board.

Comandos especiais
------------------
  help / ajuda   Exibe sintaxe
  quit / exit    Encerra sessão
"""

from __future__ import annotations

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision_engine import evaluate_action  # noqa: E402


# ---------------------------------------------------------------------------
# ANSI — desativado automaticamente fora de TTY
# ---------------------------------------------------------------------------

_TTY = sys.stdout.isatty()

class _C:
    if _TTY:
        RESET   = "\033[0m";  BOLD    = "\033[1m";  DIM     = "\033[2m"
        CYAN    = "\033[96m"; GREEN   = "\033[92m"; YELLOW  = "\033[93m"
        RED     = "\033[91m"; MAGENTA = "\033[95m"; WHITE   = "\033[97m"
        GREY    = "\033[90m"; BLUE    = "\033[94m"
    else:
        RESET=BOLD=DIM=CYAN=GREEN=YELLOW=RED=MAGENTA=WHITE=GREY=BLUE=""

_SEP  = f"{_C.GREY}{'─' * 60}{_C.RESET}"
_SEP2 = f"{_C.CYAN}{'═' * 60}{_C.RESET}"


# ---------------------------------------------------------------------------
# Renderização — pré-flop
# ---------------------------------------------------------------------------

def _render_preflop(res: dict, elapsed_us: float) -> None:
    is_multiway = bool(res.get("is_multiway", False))
    mway  = f"{_C.YELLOW}Multiway{_C.RESET}" if is_multiway else f"{_C.GREY}Heads-up{_C.RESET}"
    eq    = res.get("preflop_equity")
    eq_str = f"  Eq≈{_C.WHITE}{eq}%{_C.RESET}" if eq else ""
    open_size = res.get("open_size_bb", res.get("sizing_bb", "?"))
    stack_bb = res.get("eff_stack_bb", res.get("stack_bb", "?"))

    decision      = res["decision"]
    decision_col  = _colorize_decision(decision)

    print(_SEP)
    print(
        f"  {_C.BOLD}Mão:{_C.RESET} {_C.WHITE}{res['hand']:<6}{_C.RESET}"
        f"  {_C.BOLD}Pos:{_C.RESET} {_C.WHITE}{res['position']:<4}{_C.RESET}"
          f"  {_C.BOLD}Open:{_C.RESET} {_C.WHITE}{open_size}bb{_C.RESET}"
          f"  {_C.BOLD}Stack:{_C.RESET} {_C.WHITE}{stack_bb}bb{_C.RESET}"
        f"  {mway}"
    )
    print(f"  {_C.BOLD}{res['tier_label']}{_C.RESET}{eq_str}\n")
    print(f"  {_C.BOLD}Decisão →{_C.RESET}  {decision_col}\n")
    print(f"{_C.GREY}  ⏱  latência: {elapsed_us:.1f} µs{_C.RESET}")
    print(_SEP + "\n")


# ---------------------------------------------------------------------------
# Renderização — pós-flop
# ---------------------------------------------------------------------------

def _render_postflop(res: dict, elapsed_ms: float) -> None:
    board_str = "  ".join(res.get("board", []))
    mway      = f"{_C.YELLOW}Multiway{_C.RESET}" if res.get("is_multiway") else f"{_C.GREY}Heads-up{_C.RESET}"

    decision_col = _colorize_decision(res["decision"])

    # Barra de equidade visual (20 chars)
    eq_adj  = res.get("equity_adjusted", 0)
    bar_len = int(eq_adj / 5)              # 0–100% → 0–20 blocos
    bar_ok  = "█" * bar_len
    bar_no  = "░" * (20 - bar_len)
    eq_color = _C.GREEN if eq_adj >= 50 else (_C.YELLOW if eq_adj >= 30 else _C.RED)
    eq_bar  = f"{eq_color}{bar_ok}{_C.GREY}{bar_no}{_C.RESET}"

    print(_SEP)
    print(
        f"  {_C.BOLD}Mão:{_C.RESET} {_C.WHITE}{res['hand']:<6}{_C.RESET}"
        f"  {_C.BOLD}Pos:{_C.RESET} {_C.WHITE}{res['position']:<4}{_C.RESET}"
        f"  {_C.BOLD}Board:{_C.RESET} {_C.CYAN}{board_str}{_C.RESET}"
        f"  {mway}"
    )
    print(
        f"  {_C.BOLD}Call:{_C.RESET} {_C.WHITE}{res.get('call_size_bb','?')}bb{_C.RESET}"
        f"  {_C.BOLD}Pot:{_C.RESET} {_C.WHITE}{res.get('pot_size_bb','?')}bb{_C.RESET}"
        f"  {_C.BOLD}Força:{_C.RESET} {_C.WHITE}{res.get('hand_label','—')}{_C.RESET}"
    )

    # Outs e draws
    od = res.get("outs_data", {})
    if od and od.get("outs_total", 0) > 0:
        draws = ", ".join(od.get("draw_types", []))
        street = od.get("street", "").upper()
        mult   = od.get("multiplier", "?")
        print(
            f"  {_C.BOLD}Outs:{_C.RESET} {_C.WHITE}{od['outs_total']}{_C.RESET}"
            f"  ({_C.GREY}{street} ×{mult}{_C.RESET})"
            f"  {_C.BLUE}{draws}{_C.RESET}"
        )

    # Equidade
    print(
        f"\n  {_C.BOLD}Equidade :{_C.RESET}  {eq_bar}"
        f"  {eq_color}{eq_adj}%{_C.RESET}"
        f"  {_C.GREY}(bruta: {res.get('equity','?')}%){_C.RESET}"
    )
    print(
        f"  {_C.BOLD}Pot Odds :{_C.RESET}  {_C.WHITE}{res.get('pot_odds','?')}%{_C.RESET}"
        f"   {_C.BOLD}EV ajustado:{_C.RESET}  "
        + _colorize_ev(res.get("ev_adjusted", 0))
    )
    print(f"  {_C.GREY}{res.get('texture_note','')}{_C.RESET}")

    # Mostra IC do Monte Carlo se disponível
    mc_lo  = res.get("mc_ci_lower")
    mc_hi  = res.get("mc_ci_upper")
    mc_n   = res.get("mc_iterations")
    mc_err = res.get("mc_margin_of_error")
    if mc_lo is not None and mc_hi is not None:
        print(
            f"  {_C.GREY}Monte Carlo N={mc_n}"
            f"  IC 95%: [{mc_lo}%–{mc_hi}%]"
            f"  ±{mc_err}pp{_C.RESET}"
        )
    print(f"\n  {_C.BOLD}Decisão  →{_C.RESET}  {decision_col}\n")
    print(f"{_C.GREY}  ⏱  latência: {elapsed_ms:.2f} ms{_C.RESET}")
    print(_SEP + "\n")


# ---------------------------------------------------------------------------
# Helpers de cor
# ---------------------------------------------------------------------------

def _colorize_decision(decision: str) -> str:
    d = decision
    if d.startswith("BET/RAISE") or d.startswith("3BET") or d.startswith("ALL-IN") or d.startswith("RFI"):
        return f"{_C.GREEN}{_C.BOLD}{d}{_C.RESET}"
    if d.startswith("OPEN") or d.startswith("FLAT") or d.startswith("CALL"):
        return f"{_C.CYAN}{_C.BOLD}{d}{_C.RESET}"
    if d.startswith("FOLD"):
        return f"{_C.RED}{_C.BOLD}{d}{_C.RESET}"
    return f"{_C.WHITE}{d}{_C.RESET}"


def _colorize_ev(ev: float) -> str:
    color = _C.GREEN if ev > 0 else (_C.YELLOW if ev == 0 else _C.RED)
    return f"{color}{_C.BOLD}{ev:+.2f}bb{_C.RESET}"


# ---------------------------------------------------------------------------
# Parser — split simples, O(n) no input, zero overhead de regex
# ---------------------------------------------------------------------------

async def parse_and_execute(input_string: str) -> None:
    """
    Parse e execução de um comando.

    Formato base: MÃO POSIÇÃO OPEN_BB STACK_BB MULTIWAY [CARTAS...]
    Roteamento:
      len(tokens) == 5             → pré-flop
      len(tokens) in {8, 9, 10}   → pós-flop com 3/4/5 cartas de board
    """
    tokens = input_string.strip().split()
    n      = len(tokens)

    # Validação de contagem de tokens
    if n < 5:
        raise ValueError(
            f"Tokens insuficientes ({n}). Mínimo: 5. "
            f"Sintaxe: MÃO POSIÇÃO OPEN_BB STACK_BB MULTIWAY [CARTAS...]"
        )
    if n == 6 or n == 7:
        raise ValueError(
            f"Board incompleto ({n - 5} carta(s)). "
            f"Use 3 cartas (flop), 4 (turn) ou 5 (river)."
        )
    if n > 10:
        raise ValueError(
            f"Tokens demais ({n}). Máximo: 10 (mão + 5 tokens base + 5 cartas de board)."
        )

    raw_hand, raw_pos, raw_open, raw_stack, raw_mway = tokens[:5]
    board_tokens = tokens[5:] if n > 5 else []

    # Coerção de tipos
    try:
        open_size_bb = float(raw_open)
    except ValueError:
        raise ValueError(
            f"OPEN_BB/CALL_BB inválido: '{raw_open}'. Use número (ex: 2.5 ou 10)."
        )

    try:
        eff_stack_bb = float(raw_stack)
    except ValueError:
        raise ValueError(
            f"STACK_BB/POT_BB inválido: '{raw_stack}'. Use número (ex: 100 ou 35)."
        )

    mway_upper = raw_mway.upper()
    if mway_upper == "Y":
        is_multiway = True
    elif mway_upper == "N":
        is_multiway = False
    else:
        raise ValueError(
            f"MULTIWAY inválido: '{raw_mway}'. Use 'Y' (sim) ou 'N' (não)."
        )

    if board_tokens:
        if open_size_bb <= 0:
            raise ValueError(f"CALL_BB deve ser positivo no pós-flop. Recebido: {open_size_bb}")
    else:
        if open_size_bb < 0:
            raise ValueError(f"OPEN_BB não pode ser negativo no pré-flop. Recebido: {open_size_bb}")
    if eff_stack_bb <= 0:
        raise ValueError(f"STACK_BB/POT_BB deve ser positivo. Recebido: {eff_stack_bb}")

    # Execução e renderização
    t0 = time.perf_counter()
    result = evaluate_action(
        hand=raw_hand,
        position=raw_pos.upper(),
        open_size_bb=open_size_bb,
        eff_stack_bb=eff_stack_bb,
        is_multiway=is_multiway,
        board=board_tokens if board_tokens else None,
    )
    elapsed = time.perf_counter() - t0

    if result["has_board"]:
        _render_postflop(result, elapsed * 1000)
    else:
        _render_preflop(result, elapsed * 1_000_000)


# ---------------------------------------------------------------------------
# Banner e help
# ---------------------------------------------------------------------------

def _banner() -> None:
    print(f"\n{_SEP2}")
    print(f"{_C.CYAN}{_C.BOLD}{'  ♠  POKER DECISION ENGINE — v2.0':^60}{_C.RESET}")
    print(f"{_C.CYAN}{'  Pré-flop · Pós-flop · EV Dinâmico':^60}{_C.RESET}")
    print(f"{_SEP2}")
    _help(brief=True)


def _help(brief: bool = False) -> None:
    print(f"\n{_C.YELLOW}{_C.BOLD}  FORMATOS:{_C.RESET}")
    print(f"  {_C.WHITE}Pré-flop :{_C.RESET} {_C.GREY}MÃO POSIÇÃO OPEN_BB STACK_BB MULTIWAY{_C.RESET}")
    print(f"  {_C.GREEN}           ex: 66 CO 2.5 35 N{_C.RESET}")
    print(f"  {_C.WHITE}Pós-flop :{_C.RESET} {_C.GREY}MÃO POSIÇÃO CALL_BB POT_BB MULTIWAY CARTA1 CARTA2 CARTA3 [C4] [C5]{_C.RESET}")
    print(f"  {_C.GREEN}           ex: AKs BTN 10 30 N Jh Td 3c{_C.RESET}")
    print(f"  {_C.GREEN}           ex: QJs HJ 8 25 N Th 9d 2c 8h{_C.RESET}")
    if not brief:
        print(f"\n  {_C.GREY}Posições : UTG  LJ  HJ  CO  BTN  SB  BB{_C.RESET}")
        print(f"  {_C.GREY}Multiway : Y = sim   N = não{_C.RESET}")
        print(f"  {_C.GREY}Cartas   : rank + naipe  ex: Ah Td 2c Ks{_C.RESET}")
        print(f"  {_C.GREY}Comandos : help   quit   exit{_C.RESET}")
    print()


def _print_error(msg: str, show_format: bool = False) -> None:
    print(f"\n  {_C.RED}{_C.BOLD}✗  Erro:{_C.RESET} {_C.WHITE}{msg}{_C.RESET}")
    if show_format:
        print(f"  {_C.GREY}Pré-flop : MÃO POSIÇÃO OPEN_BB STACK_BB MULTIWAY{_C.RESET}")
        print(f"  {_C.GREY}Pós-flop : MÃO POSIÇÃO CALL_BB POT_BB MULTIWAY CARTA1 CARTA2 CARTA3{_C.RESET}")
        print(f"  {_C.GREY}Exemplo  : AKs BTN 10 30 N Jh Td 3c{_C.RESET}\n")


# ---------------------------------------------------------------------------
# Event loop principal
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Loop assíncrono persistente.
    asyncio.to_thread() mantém o event loop livre durante input bloqueante.
    ValueError é capturado localmente — o loop nunca quebra por input errado.
    """
    _banner()

    while True:
        try:
            raw: str = await asyncio.to_thread(input, f"{_C.CYAN}>> {_C.RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_C.GREY}  Sessão encerrada. Boa sorte nas mesas. ♠{_C.RESET}\n")
            break

        cmd = raw.strip().lower()

        if not cmd:
            continue

        if cmd in {"quit", "exit", "q", "sair"}:
            print(f"\n{_C.GREY}  Sessão encerrada. Boa sorte nas mesas. ♠{_C.RESET}\n")
            break

        if cmd in {"help", "ajuda", "h", "?"}:
            _help()
            continue

        try:
            await parse_and_execute(raw)
        except ValueError as exc:
            _print_error(str(exc), show_format=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
