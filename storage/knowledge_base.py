"""
knowledge_base.py
=================
Structured poker range data for low-latency (O(1)) lookup.

Author  : Senior Data Engineer
Purpose : Central knowledge base for a poker range analysis engine.
          All top-level structures are plain dicts/lists so attribute
          access and key lookup are guaranteed O(1) by CPython's hash-table
          implementation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. RFI (Raise First In) Ranges
#    Key   → position string  (str)
#    Value → list of hand     (list[str])
#
#    Notation conventions used throughout:
#      "77+"   → pair 77 and all pairs above (88, 99, TT, JJ, QQ, KK, AA)
#      "AJs+"  → AJs, AQs, AKs  (suited, rank of second card moves up)
#      "AJo+"  → AJo, AQo, AKo  (off-suit, same ladder logic)
#      "T9s"   → exact suited connector, no ladder (no '+')
# ---------------------------------------------------------------------------

RFI_RANGES: dict[str, list[str]] = {
    # Under The Gun — tightest opening range, ~13 % of hands
    "UTG": [
        "77+",          # pairs: 77, 88, 99, TT, JJ, QQ, KK, AA
        "AJs+",         # suited aces: AJs, AQs, AKs
        "KQs",          # premium suited broadway
        "AQo+",         # off-suit aces: AQo, AKo
    ],

    # Lojack (LJ) — one seat left of UTG, slight range expansion
    "LJ": [
        "66+",          # pairs: 66 → AA
        "ATs+",         # suited aces: ATs, AJs, AQs, AKs
        "KQs",
        "KJs",          # suited kings
        "QJs",          # suited broadway connector
        "AQo+",         # off-suit aces: AQo, AKo
        "AJo",          # added off-suit ace
    ],

    # Hijack (HJ) — two seats right of BTN
    "HJ": [
        "55+",          # pairs: 55 → AA
        "A9s+",         # suited aces: A9s → AKs
        "KTs+",         # suited kings: KTs, KJs, KQs
        "QTs+",         # suited queens: QTs, QJs
        "JTs",          # suited broadway connector
        "AJo+",         # off-suit aces: AJo, AQo, AKo
        "KQo",          # premium off-suit broadway
    ],

    # Cutoff (CO) — one right of BTN, strong steal position
    "CO": [
        "44+",          # pairs: 44 → AA
        "A5s+",         # suited aces: A5s → AKs (includes wheel nut-flush draws)
        "K9s+",         # suited kings: K9s → KQs
        "Q9s+",         # suited queens: Q9s → QKs
        "J9s+",         # suited jacks: J9s, JTs
        "T9s",          # suited connector
        "ATo+",         # off-suit aces: ATo, AJo, AQo, AKo
        "KJo+",         # off-suit kings: KJo, KQo
        "98s+",         # suited connectors: 98s, (T9s already listed)
    ],

    # Button (BTN) — best positional seat, widest range
    "BTN": [
        "22+",          # all pairs
        "A2s+",         # all suited aces
        "A2o+",         # all off-suit aces (wide steal range)
        "K9o+",         # off-suit kings: K9o → KQo
        "Q9o+",         # off-suit queens: Q9o → QKo
        "J9o+",         # off-suit jacks: J9o, JTo
        "65s+",         # lower suited connectors: 65s, 76s, 87s, 98s, T9s
    ],

    # Small Blind (SB) — positional disadvantage post-flop but steals vs BB
    "SB": [
        "22+",          # all pairs
        "A2s+",         # all suited aces
        "A8o+",         # off-suit aces: A8o → AKo (narrower than BTN)
        "KTs+",         # suited kings: KTs, KJs, KQs
        "QTs+",         # suited queens: QTs, QJs
        "JTs",          # suited broadway connector
        "KJo+",         # off-suit kings: KJo, KQo
        "QJo",          # off-suit broadway
    ],
}


# ---------------------------------------------------------------------------
# 2. Flat (Call) Rules Configuration
#    Centralised config dict — single source of truth for the flat-calling
#    logic layer.  Nested dicts keep lookup O(1) at every level.
# ---------------------------------------------------------------------------

FLAT_RULES_CONFIG: dict[str, dict] = {

    # -------------------------------------------------------------------
    # 2a. Pot-odds multipliers
    #     Used to derive the minimum implied-odds multiple required before
    #     calling (flatting) a raise.
    #
    #     Formula (illustrative):
    #       required_stack_bb = raise_size_bb * multiplier
    # -------------------------------------------------------------------
    "multipliers": {
        "min_flat_general":    9,   # standard heads-up flat call
        "min_flat_multiway":  10,   # extra caller(s) already in the pot
        "min_flat_set_mining": 15,  # small/medium pairs hunting a set
    },

    # -------------------------------------------------------------------
    # 2b. Hand categories
    #     Each category maps to a representative sample list.
    #     The engine uses these buckets to route hands to the correct
    #     multiplier and post-flop strategy.
    # -------------------------------------------------------------------
    "hand_categories": {
        # Suited connectors — strong drawing hands, good implied odds
        "suited_connectors": [
            "JTs", "T9s", "98s", "87s", "76s", "65s", "54s",
        ],

        # Small-to-medium pairs — primary goal is flopping a set (8.5:1)
        "pairs_set_mining": [
            "22", "33", "44", "55", "66", "77",
            "88", "99", "TT", "JJ", "QQ",      # QQ included; AA/KK usually 3-bet
        ],

        # Strong broadway hands — high card equity + nut potential
        "strong_broadways": [
            "AKs", "AQs", "AJs", "KQs",        # suited premiums
            "AKo", "AQo", "AJo", "KQo",        # off-suit premiums
        ],
    },

    # -------------------------------------------------------------------
    # 2c. Position modifiers
    #     Maps logical position groups to concrete seat labels.
    #     Used by the engine to tighten/widen ranges and adjust
    #     multipliers based on positional equity.
    # -------------------------------------------------------------------
    "position_modifiers": {
        # Act early; face maximum players behind — play tighter
        "early_middle": ["UTG", "MP", "LJ", "HJ"],

        # Act late pre-flop AND post-flop — can widen and steal
        "late": ["CO", "BTN"],

        # Small & Big Blind — positional disadvantage post-flop
        "blinds": ["SB", "BB"],
    },
}


# ---------------------------------------------------------------------------
# Quick-access reverse index (O(1) position → group lookup)
#
# Built once at import time so callers never pay the O(n) scan cost.
# Example: POSITION_GROUP["BTN"] → "late"
# ---------------------------------------------------------------------------

POSITION_GROUP: dict[str, str] = {
    seat: group
    for group, seats in FLAT_RULES_CONFIG["position_modifiers"].items()
    for seat in seats
}


# ---------------------------------------------------------------------------
# 3. Hand Rankings
#    Chave → tupla ordenada de ranks (int) — O(1) lookup por hash de tupla.
#    Suited e offsuit são entradas separadas pois a equidade difere
#    significativamente (suited = +3–5% de equidade média).
#
#    Formato da chave:
#      Par       : (rank, rank)           ex: (14, 14) = AA
#      Suited    : (rank_high, rank_low, 's')
#      Offsuit   : (rank_high, rank_low, 'o')
#
#    Valor → dict com:
#      'tier'          : int 1–5  (1 = premium, 5 = especulativo)
#      'label'         : str  descrição legível
#      'preflop_equity': float  equidade média pré-flop vs range aleatório
#      'postflop_playability': float 0.0–1.0  (facilidade de jogar pós-flop)
# ---------------------------------------------------------------------------

HAND_RANKINGS: dict[tuple, dict] = {

    # -----------------------------------------------------------------------
    # Tier 1 — Premium: abrir/3-bet de qualquer posição
    # -----------------------------------------------------------------------
    (14, 14):        {"tier": 1, "label": "AA",  "preflop_equity": 0.852, "postflop_playability": 0.95},
    (13, 13):        {"tier": 1, "label": "KK",  "preflop_equity": 0.823, "postflop_playability": 0.92},
    (12, 12):        {"tier": 1, "label": "QQ",  "preflop_equity": 0.800, "postflop_playability": 0.88},
    (14, 13, "s"):   {"tier": 1, "label": "AKs", "preflop_equity": 0.670, "postflop_playability": 0.90},
    (14, 13, "o"):   {"tier": 1, "label": "AKo", "preflop_equity": 0.655, "postflop_playability": 0.85},

    # -----------------------------------------------------------------------
    # Tier 2 — Strong: abrir de todas as posições, 3-bet vs range certo
    # -----------------------------------------------------------------------
    (11, 11):        {"tier": 2, "label": "JJ",  "preflop_equity": 0.775, "postflop_playability": 0.82},
    (10, 10):        {"tier": 2, "label": "TT",  "preflop_equity": 0.752, "postflop_playability": 0.78},
    (14, 12, "s"):   {"tier": 2, "label": "AQs", "preflop_equity": 0.660, "postflop_playability": 0.88},
    (14, 12, "o"):   {"tier": 2, "label": "AQo", "preflop_equity": 0.640, "postflop_playability": 0.82},
    (14, 11, "s"):   {"tier": 2, "label": "AJs", "preflop_equity": 0.645, "postflop_playability": 0.87},
    (13, 12, "s"):   {"tier": 2, "label": "KQs", "preflop_equity": 0.632, "postflop_playability": 0.86},

    # -----------------------------------------------------------------------
    # Tier 3 — Playable: abrir de posições médias/tardias
    # -----------------------------------------------------------------------
    (9, 9):          {"tier": 3, "label": "99",  "preflop_equity": 0.720, "postflop_playability": 0.72},
    (8, 8):          {"tier": 3, "label": "88",  "preflop_equity": 0.690, "postflop_playability": 0.68},
    (7, 7):          {"tier": 3, "label": "77",  "preflop_equity": 0.662, "postflop_playability": 0.65},
    (14, 11, "o"):   {"tier": 3, "label": "AJo", "preflop_equity": 0.625, "postflop_playability": 0.78},
    (14, 10, "s"):   {"tier": 3, "label": "ATs", "preflop_equity": 0.635, "postflop_playability": 0.85},
    (13, 12, "o"):   {"tier": 3, "label": "KQo", "preflop_equity": 0.615, "postflop_playability": 0.80},
    (13, 11, "s"):   {"tier": 3, "label": "KJs", "preflop_equity": 0.620, "postflop_playability": 0.83},
    (12, 11, "s"):   {"tier": 3, "label": "QJs", "preflop_equity": 0.608, "postflop_playability": 0.82},
    (11, 10, "s"):   {"tier": 3, "label": "JTs", "preflop_equity": 0.598, "postflop_playability": 0.84},

    # -----------------------------------------------------------------------
    # Tier 4 — Speculative: posições tardias, boas implied odds necessárias
    # -----------------------------------------------------------------------
    (6, 6):          {"tier": 4, "label": "66",  "preflop_equity": 0.635, "postflop_playability": 0.60},
    (5, 5):          {"tier": 4, "label": "55",  "preflop_equity": 0.608, "postflop_playability": 0.58},
    (4, 4):          {"tier": 4, "label": "44",  "preflop_equity": 0.582, "postflop_playability": 0.55},
    (3, 3):          {"tier": 4, "label": "33",  "preflop_equity": 0.558, "postflop_playability": 0.52},
    (2, 2):          {"tier": 4, "label": "22",  "preflop_equity": 0.535, "postflop_playability": 0.50},
    (14, 10, "o"):   {"tier": 4, "label": "ATo", "preflop_equity": 0.615, "postflop_playability": 0.72},
    (14,  9, "s"):   {"tier": 4, "label": "A9s", "preflop_equity": 0.622, "postflop_playability": 0.80},
    (13, 10, "s"):   {"tier": 4, "label": "KTs", "preflop_equity": 0.610, "postflop_playability": 0.80},
    (12, 10, "s"):   {"tier": 4, "label": "QTs", "preflop_equity": 0.598, "postflop_playability": 0.80},
    (10,  9, "s"):   {"tier": 4, "label": "T9s", "preflop_equity": 0.585, "postflop_playability": 0.82},
    ( 9,  8, "s"):   {"tier": 4, "label": "98s", "preflop_equity": 0.572, "postflop_playability": 0.80},
    ( 8,  7, "s"):   {"tier": 4, "label": "87s", "preflop_equity": 0.560, "postflop_playability": 0.78},
    ( 7,  6, "s"):   {"tier": 4, "label": "76s", "preflop_equity": 0.548, "postflop_playability": 0.76},
    ( 6,  5, "s"):   {"tier": 4, "label": "65s", "preflop_equity": 0.536, "postflop_playability": 0.74},

    # -----------------------------------------------------------------------
    # Tier 5 — Marginal: apenas BTN/SB com stack profundo, exploitação
    # -----------------------------------------------------------------------
    (14,  8, "s"):   {"tier": 5, "label": "A8s", "preflop_equity": 0.612, "postflop_playability": 0.78},
    (14,  7, "s"):   {"tier": 5, "label": "A7s", "preflop_equity": 0.605, "postflop_playability": 0.76},
    (14,  6, "s"):   {"tier": 5, "label": "A6s", "preflop_equity": 0.598, "postflop_playability": 0.74},
    (14,  5, "s"):   {"tier": 5, "label": "A5s", "preflop_equity": 0.592, "postflop_playability": 0.75},
    (14,  4, "s"):   {"tier": 5, "label": "A4s", "preflop_equity": 0.585, "postflop_playability": 0.73},
    (14,  3, "s"):   {"tier": 5, "label": "A3s", "preflop_equity": 0.578, "postflop_playability": 0.71},
    (14,  2, "s"):   {"tier": 5, "label": "A2s", "preflop_equity": 0.572, "postflop_playability": 0.70},
    (13, 11, "o"):   {"tier": 5, "label": "KJo", "preflop_equity": 0.598, "postflop_playability": 0.72},
    (13, 10, "o"):   {"tier": 5, "label": "KTo", "preflop_equity": 0.588, "postflop_playability": 0.70},
    ( 5,  4, "s"):   {"tier": 5, "label": "54s", "preflop_equity": 0.524, "postflop_playability": 0.72},

    # -----------------------------------------------------------------------
    # Tier 4-6 — Cobertura completa: 125 mãos restantes das 169
    # Dados: equidade vs mão aleatória calibrada em Sklansky Hand Groups
    # e Power Numbers de torneio. Tier 6 = mãos negativas EV em >90% dos spots.
    # -----------------------------------------------------------------------

    # Aces offsuit (A9o–A2o)
    (14,  9, "o"):   {"tier": 4, "label": "A9o",  "preflop_equity": 0.593, "postflop_playability": 0.72},
    (14,  8, "o"):   {"tier": 4, "label": "A8o",  "preflop_equity": 0.590, "postflop_playability": 0.71},
    (14,  7, "o"):   {"tier": 4, "label": "A7o",  "preflop_equity": 0.587, "postflop_playability": 0.70},
    (14,  6, "o"):   {"tier": 4, "label": "A6o",  "preflop_equity": 0.583, "postflop_playability": 0.70},
    (14,  5, "o"):   {"tier": 4, "label": "A5o",  "preflop_equity": 0.580, "postflop_playability": 0.69},
    (14,  4, "o"):   {"tier": 5, "label": "A4o",  "preflop_equity": 0.577, "postflop_playability": 0.68},
    (14,  3, "o"):   {"tier": 5, "label": "A3o",  "preflop_equity": 0.573, "postflop_playability": 0.68},
    (14,  2, "o"):   {"tier": 5, "label": "A2o",  "preflop_equity": 0.570, "postflop_playability": 0.67},

    # Kings suited (K9s–K2s)
    (13,  9, "s"):   {"tier": 4, "label": "K9s",  "preflop_equity": 0.604, "postflop_playability": 0.76},
    (13,  8, "s"):   {"tier": 4, "label": "K8s",  "preflop_equity": 0.600, "postflop_playability": 0.76},
    (13,  7, "s"):   {"tier": 4, "label": "K7s",  "preflop_equity": 0.597, "postflop_playability": 0.75},
    (13,  6, "s"):   {"tier": 4, "label": "K6s",  "preflop_equity": 0.593, "postflop_playability": 0.74},
    (13,  5, "s"):   {"tier": 4, "label": "K5s",  "preflop_equity": 0.589, "postflop_playability": 0.74},
    (13,  4, "s"):   {"tier": 4, "label": "K4s",  "preflop_equity": 0.585, "postflop_playability": 0.73},
    (13,  3, "s"):   {"tier": 5, "label": "K3s",  "preflop_equity": 0.581, "postflop_playability": 0.72},
    (13,  2, "s"):   {"tier": 5, "label": "K2s",  "preflop_equity": 0.577, "postflop_playability": 0.71},

    # Kings offsuit (K9o–K2o)
    (13,  9, "o"):   {"tier": 4, "label": "K9o",  "preflop_equity": 0.587, "postflop_playability": 0.70},
    (13,  8, "o"):   {"tier": 4, "label": "K8o",  "preflop_equity": 0.583, "postflop_playability": 0.69},
    (13,  7, "o"):   {"tier": 4, "label": "K7o",  "preflop_equity": 0.580, "postflop_playability": 0.68},
    (13,  6, "o"):   {"tier": 5, "label": "K6o",  "preflop_equity": 0.577, "postflop_playability": 0.68},
    (13,  5, "o"):   {"tier": 5, "label": "K5o",  "preflop_equity": 0.573, "postflop_playability": 0.67},
    (13,  4, "o"):   {"tier": 5, "label": "K4o",  "preflop_equity": 0.570, "postflop_playability": 0.66},
    (13,  3, "o"):   {"tier": 5, "label": "K3o",  "preflop_equity": 0.566, "postflop_playability": 0.65},
    (13,  2, "o"):   {"tier": 6, "label": "K2o",  "preflop_equity": 0.562, "postflop_playability": 0.64},

    # Queens suited (Q9s–Q2s)
    (12,  9, "s"):   {"tier": 4, "label": "Q9s",  "preflop_equity": 0.593, "postflop_playability": 0.75},
    (12,  8, "s"):   {"tier": 4, "label": "Q8s",  "preflop_equity": 0.586, "postflop_playability": 0.74},
    (12,  7, "s"):   {"tier": 5, "label": "Q7s",  "preflop_equity": 0.579, "postflop_playability": 0.72},
    (12,  6, "s"):   {"tier": 5, "label": "Q6s",  "preflop_equity": 0.572, "postflop_playability": 0.71},
    (12,  5, "s"):   {"tier": 5, "label": "Q5s",  "preflop_equity": 0.565, "postflop_playability": 0.70},
    (12,  4, "s"):   {"tier": 5, "label": "Q4s",  "preflop_equity": 0.558, "postflop_playability": 0.69},
    (12,  3, "s"):   {"tier": 6, "label": "Q3s",  "preflop_equity": 0.551, "postflop_playability": 0.67},
    (12,  2, "s"):   {"tier": 6, "label": "Q2s",  "preflop_equity": 0.544, "postflop_playability": 0.66},

    # Queens offsuit (QJo–Q2o)
    (12, 11, "o"):   {"tier": 4, "label": "QJo",  "preflop_equity": 0.591, "postflop_playability": 0.74},
    (12, 10, "o"):   {"tier": 4, "label": "QTo",  "preflop_equity": 0.582, "postflop_playability": 0.73},
    (12,  9, "o"):   {"tier": 4, "label": "Q9o",  "preflop_equity": 0.573, "postflop_playability": 0.70},
    (12,  8, "o"):   {"tier": 5, "label": "Q8o",  "preflop_equity": 0.563, "postflop_playability": 0.67},
    (12,  7, "o"):   {"tier": 5, "label": "Q7o",  "preflop_equity": 0.554, "postflop_playability": 0.65},
    (12,  6, "o"):   {"tier": 5, "label": "Q6o",  "preflop_equity": 0.545, "postflop_playability": 0.63},
    (12,  5, "o"):   {"tier": 6, "label": "Q5o",  "preflop_equity": 0.536, "postflop_playability": 0.61},
    (12,  4, "o"):   {"tier": 6, "label": "Q4o",  "preflop_equity": 0.527, "postflop_playability": 0.59},
    (12,  3, "o"):   {"tier": 6, "label": "Q3o",  "preflop_equity": 0.518, "postflop_playability": 0.57},
    (12,  2, "o"):   {"tier": 6, "label": "Q2o",  "preflop_equity": 0.509, "postflop_playability": 0.55},

    # Jacks suited (J9s–J2s)
    (11,  9, "s"):   {"tier": 4, "label": "J9s",  "preflop_equity": 0.585, "postflop_playability": 0.76},
    (11,  8, "s"):   {"tier": 4, "label": "J8s",  "preflop_equity": 0.578, "postflop_playability": 0.75},
    (11,  7, "s"):   {"tier": 5, "label": "J7s",  "preflop_equity": 0.569, "postflop_playability": 0.72},
    (11,  6, "s"):   {"tier": 5, "label": "J6s",  "preflop_equity": 0.561, "postflop_playability": 0.70},
    (11,  5, "s"):   {"tier": 5, "label": "J5s",  "preflop_equity": 0.553, "postflop_playability": 0.68},
    (11,  4, "s"):   {"tier": 6, "label": "J4s",  "preflop_equity": 0.545, "postflop_playability": 0.66},
    (11,  3, "s"):   {"tier": 6, "label": "J3s",  "preflop_equity": 0.537, "postflop_playability": 0.64},
    (11,  2, "s"):   {"tier": 6, "label": "J2s",  "preflop_equity": 0.529, "postflop_playability": 0.62},

    # Jacks offsuit (JTo–J2o)
    (11, 10, "o"):   {"tier": 4, "label": "JTo",  "preflop_equity": 0.581, "postflop_playability": 0.74},
    (11,  9, "o"):   {"tier": 4, "label": "J9o",  "preflop_equity": 0.569, "postflop_playability": 0.71},
    (11,  8, "o"):   {"tier": 5, "label": "J8o",  "preflop_equity": 0.557, "postflop_playability": 0.68},
    (11,  7, "o"):   {"tier": 5, "label": "J7o",  "preflop_equity": 0.546, "postflop_playability": 0.65},
    (11,  6, "o"):   {"tier": 5, "label": "J6o",  "preflop_equity": 0.535, "postflop_playability": 0.62},
    (11,  5, "o"):   {"tier": 6, "label": "J5o",  "preflop_equity": 0.524, "postflop_playability": 0.59},
    (11,  4, "o"):   {"tier": 6, "label": "J4o",  "preflop_equity": 0.513, "postflop_playability": 0.56},
    (11,  3, "o"):   {"tier": 6, "label": "J3o",  "preflop_equity": 0.502, "postflop_playability": 0.53},
    (11,  2, "o"):   {"tier": 6, "label": "J2o",  "preflop_equity": 0.491, "postflop_playability": 0.50},

    # Tens suited (T8s–T2s)
    (10,  8, "s"):   {"tier": 4, "label": "T8s",  "preflop_equity": 0.575, "postflop_playability": 0.75},
    (10,  7, "s"):   {"tier": 5, "label": "T7s",  "preflop_equity": 0.566, "postflop_playability": 0.73},
    (10,  6, "s"):   {"tier": 5, "label": "T6s",  "preflop_equity": 0.556, "postflop_playability": 0.70},
    (10,  5, "s"):   {"tier": 5, "label": "T5s",  "preflop_equity": 0.547, "postflop_playability": 0.67},
    (10,  4, "s"):   {"tier": 6, "label": "T4s",  "preflop_equity": 0.538, "postflop_playability": 0.64},
    (10,  3, "s"):   {"tier": 6, "label": "T3s",  "preflop_equity": 0.529, "postflop_playability": 0.61},
    (10,  2, "s"):   {"tier": 6, "label": "T2s",  "preflop_equity": 0.520, "postflop_playability": 0.58},

    # Tens offsuit (T9o–T2o)
    (10,  9, "o"):   {"tier": 4, "label": "T9o",  "preflop_equity": 0.568, "postflop_playability": 0.71},
    (10,  8, "o"):   {"tier": 5, "label": "T8o",  "preflop_equity": 0.555, "postflop_playability": 0.68},
    (10,  7, "o"):   {"tier": 5, "label": "T7o",  "preflop_equity": 0.542, "postflop_playability": 0.64},
    (10,  6, "o"):   {"tier": 5, "label": "T6o",  "preflop_equity": 0.529, "postflop_playability": 0.61},
    (10,  5, "o"):   {"tier": 6, "label": "T5o",  "preflop_equity": 0.516, "postflop_playability": 0.57},
    (10,  4, "o"):   {"tier": 6, "label": "T4o",  "preflop_equity": 0.503, "postflop_playability": 0.54},
    (10,  3, "o"):   {"tier": 6, "label": "T3o",  "preflop_equity": 0.490, "postflop_playability": 0.50},
    (10,  2, "o"):   {"tier": 6, "label": "T2o",  "preflop_equity": 0.477, "postflop_playability": 0.47},

    # Nines suited (97s–92s)
    ( 9,  7, "s"):   {"tier": 4, "label": "97s",  "preflop_equity": 0.563, "postflop_playability": 0.74},
    ( 9,  6, "s"):   {"tier": 5, "label": "96s",  "preflop_equity": 0.553, "postflop_playability": 0.71},
    ( 9,  5, "s"):   {"tier": 5, "label": "95s",  "preflop_equity": 0.543, "postflop_playability": 0.68},
    ( 9,  4, "s"):   {"tier": 6, "label": "94s",  "preflop_equity": 0.533, "postflop_playability": 0.64},
    ( 9,  3, "s"):   {"tier": 6, "label": "93s",  "preflop_equity": 0.523, "postflop_playability": 0.61},
    ( 9,  2, "s"):   {"tier": 6, "label": "92s",  "preflop_equity": 0.513, "postflop_playability": 0.58},

    # Nines offsuit (98o–92o)
    ( 9,  8, "o"):   {"tier": 4, "label": "98o",  "preflop_equity": 0.554, "postflop_playability": 0.68},
    ( 9,  7, "o"):   {"tier": 5, "label": "97o",  "preflop_equity": 0.541, "postflop_playability": 0.64},
    ( 9,  6, "o"):   {"tier": 5, "label": "96o",  "preflop_equity": 0.528, "postflop_playability": 0.60},
    ( 9,  5, "o"):   {"tier": 6, "label": "95o",  "preflop_equity": 0.515, "postflop_playability": 0.56},
    ( 9,  4, "o"):   {"tier": 6, "label": "94o",  "preflop_equity": 0.502, "postflop_playability": 0.52},
    ( 9,  3, "o"):   {"tier": 6, "label": "93o",  "preflop_equity": 0.489, "postflop_playability": 0.48},
    ( 9,  2, "o"):   {"tier": 6, "label": "92o",  "preflop_equity": 0.476, "postflop_playability": 0.45},

    # Eights suited (86s–82s)
    ( 8,  6, "s"):   {"tier": 5, "label": "86s",  "preflop_equity": 0.549, "postflop_playability": 0.72},
    ( 8,  5, "s"):   {"tier": 5, "label": "85s",  "preflop_equity": 0.539, "postflop_playability": 0.69},
    ( 8,  4, "s"):   {"tier": 6, "label": "84s",  "preflop_equity": 0.528, "postflop_playability": 0.65},
    ( 8,  3, "s"):   {"tier": 6, "label": "83s",  "preflop_equity": 0.518, "postflop_playability": 0.61},
    ( 8,  2, "s"):   {"tier": 6, "label": "82s",  "preflop_equity": 0.508, "postflop_playability": 0.57},

    # Eights offsuit (87o–82o)
    ( 8,  7, "o"):   {"tier": 5, "label": "87o",  "preflop_equity": 0.540, "postflop_playability": 0.65},
    ( 8,  6, "o"):   {"tier": 5, "label": "86o",  "preflop_equity": 0.527, "postflop_playability": 0.61},
    ( 8,  5, "o"):   {"tier": 6, "label": "85o",  "preflop_equity": 0.514, "postflop_playability": 0.57},
    ( 8,  4, "o"):   {"tier": 6, "label": "84o",  "preflop_equity": 0.501, "postflop_playability": 0.53},
    ( 8,  3, "o"):   {"tier": 6, "label": "83o",  "preflop_equity": 0.488, "postflop_playability": 0.49},
    ( 8,  2, "o"):   {"tier": 6, "label": "82o",  "preflop_equity": 0.475, "postflop_playability": 0.45},

    # Sevens suited (75s–72s)
    ( 7,  5, "s"):   {"tier": 5, "label": "75s",  "preflop_equity": 0.534, "postflop_playability": 0.70},
    ( 7,  4, "s"):   {"tier": 5, "label": "74s",  "preflop_equity": 0.524, "postflop_playability": 0.66},
    ( 7,  3, "s"):   {"tier": 6, "label": "73s",  "preflop_equity": 0.514, "postflop_playability": 0.62},
    ( 7,  2, "s"):   {"tier": 6, "label": "72s",  "preflop_equity": 0.504, "postflop_playability": 0.58},

    # Sevens offsuit (76o–72o)
    ( 7,  6, "o"):   {"tier": 5, "label": "76o",  "preflop_equity": 0.526, "postflop_playability": 0.62},
    ( 7,  5, "o"):   {"tier": 5, "label": "75o",  "preflop_equity": 0.513, "postflop_playability": 0.58},
    ( 7,  4, "o"):   {"tier": 6, "label": "74o",  "preflop_equity": 0.500, "postflop_playability": 0.54},
    ( 7,  3, "o"):   {"tier": 6, "label": "73o",  "preflop_equity": 0.487, "postflop_playability": 0.49},
    ( 7,  2, "o"):   {"tier": 6, "label": "72o",  "preflop_equity": 0.474, "postflop_playability": 0.45},

    # Sixes suited (64s–62s)
    ( 6,  4, "s"):   {"tier": 5, "label": "64s",  "preflop_equity": 0.519, "postflop_playability": 0.66},
    ( 6,  3, "s"):   {"tier": 6, "label": "63s",  "preflop_equity": 0.509, "postflop_playability": 0.62},
    ( 6,  2, "s"):   {"tier": 6, "label": "62s",  "preflop_equity": 0.499, "postflop_playability": 0.57},

    # Sixes offsuit (65o–62o)
    ( 6,  5, "o"):   {"tier": 5, "label": "65o",  "preflop_equity": 0.512, "postflop_playability": 0.58},
    ( 6,  4, "o"):   {"tier": 6, "label": "64o",  "preflop_equity": 0.499, "postflop_playability": 0.54},
    ( 6,  3, "o"):   {"tier": 6, "label": "63o",  "preflop_equity": 0.486, "postflop_playability": 0.49},
    ( 6,  2, "o"):   {"tier": 6, "label": "62o",  "preflop_equity": 0.473, "postflop_playability": 0.45},

    # Fives suited (53s–52s)
    ( 5,  3, "s"):   {"tier": 6, "label": "53s",  "preflop_equity": 0.503, "postflop_playability": 0.62},
    ( 5,  2, "s"):   {"tier": 6, "label": "52s",  "preflop_equity": 0.493, "postflop_playability": 0.57},

    # Fives offsuit (54o–52o)
    ( 5,  4, "o"):   {"tier": 6, "label": "54o",  "preflop_equity": 0.498, "postflop_playability": 0.54},
    ( 5,  3, "o"):   {"tier": 6, "label": "53o",  "preflop_equity": 0.485, "postflop_playability": 0.49},
    ( 5,  2, "o"):   {"tier": 6, "label": "52o",  "preflop_equity": 0.472, "postflop_playability": 0.45},

    # Fours offsuit (43o–42o)
    ( 4,  3, "s"):   {"tier": 6, "label": "43s",  "preflop_equity": 0.496, "postflop_playability": 0.60},
    ( 4,  2, "s"):   {"tier": 6, "label": "42s",  "preflop_equity": 0.486, "postflop_playability": 0.55},
    ( 4,  3, "o"):   {"tier": 6, "label": "43o",  "preflop_equity": 0.480, "postflop_playability": 0.46},
    ( 4,  2, "o"):   {"tier": 6, "label": "42o",  "preflop_equity": 0.467, "postflop_playability": 0.42},

    # Threes and twos offsuit
    ( 3,  2, "s"):   {"tier": 6, "label": "32s",  "preflop_equity": 0.479, "postflop_playability": 0.54},
    ( 3,  2, "o"):   {"tier": 6, "label": "32o",  "preflop_equity": 0.463, "postflop_playability": 0.40},
}

# ---------------------------------------------------------------------------
# 3a. Pesos de probabilidade pós-flop por textura de board
#     Usado para ajustar o peso de equidade conforme o board favorece
#     draws ou mãos feitas.
#
#     Chave  → tipo de textura (str)
#     Valor  → dict com modificadores multiplicativos sobre a equidade base
# ---------------------------------------------------------------------------

BOARD_TEXTURE_WEIGHTS: dict[str, dict] = {
    # Board seco (ex: K72 rainbow) — favorece mãos feitas
    "dry": {
        "label":          "Board Seco",
        "made_hand_mod":  1.10,   # mãos feitas valem mais
        "draw_mod":       0.85,   # draws valem menos (menos outs implícitos)
        "bluff_ev_mod":   1.15,   # bluffs mais credíveis
    },
    # Board úmido/conectado (ex: JT9 two-tone) — favorece draws
    "wet": {
        "label":          "Board Conectado",
        "made_hand_mod":  0.90,   # mãos feitas mais vulneráveis
        "draw_mod":       1.20,   # draws com alta equidade implícita
        "bluff_ev_mod":   0.80,   # bluffs menos críveis (range de call é largo)
    },
    # Board pareado (ex: KK7) — favorece trips/fullhouses
    "paired": {
        "label":          "Board Pareado",
        "made_hand_mod":  1.05,
        "draw_mod":       0.90,
        "bluff_ev_mod":   1.10,
    },
    # Board monotone (ex: 9h7h3h) — flush draw dominante
    "monotone": {
        "label":          "Board Monotone",
        "made_hand_mod":  0.85,
        "draw_mod":       1.30,
        "bluff_ev_mod":   0.70,
    },
}

# ---------------------------------------------------------------------------
# 3b. Reverse index: label da mão → chave de tupla (para lookup por string)
#     Construído uma vez no import — O(1) em uso.
#     Exemplo: HAND_KEY_BY_LABEL["AKs"] → (14, 13, 's')
# ---------------------------------------------------------------------------

HAND_KEY_BY_LABEL: dict[str, tuple] = {
    meta["label"]: key
    for key, meta in HAND_RANKINGS.items()
}

# ---------------------------------------------------------------------------
# Module-level sanity check (runs only when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== RFI_RANGES ===")
    for position, hands in RFI_RANGES.items():
        print(f"  {position:4s} ({len(hands):2d} combos/groups): {hands}")

    print("\n=== FLAT_RULES_CONFIG — multipliers ===")
    for key, val in FLAT_RULES_CONFIG["multipliers"].items():
        print(f"  {key}: {val}x")

    print("\n=== FLAT_RULES_CONFIG — hand_categories ===")
    for category, hands in FLAT_RULES_CONFIG["hand_categories"].items():
        print(f"  {category}: {hands}")

    print("\n=== FLAT_RULES_CONFIG — position_modifiers ===")
    for group, seats in FLAT_RULES_CONFIG["position_modifiers"].items():
        print(f"  {group}: {seats}")

    print("\n=== POSITION_GROUP reverse index ===")
    for seat, group in POSITION_GROUP.items():
        print(f"  {seat} → {group}")

    print("\n=== HAND_RANKINGS (por tier) ===")
    for tier in range(1, 6):
        hands_in_tier = [
            (meta["label"], meta["preflop_equity"], meta["postflop_playability"])
            for meta in HAND_RANKINGS.values()
            if meta["tier"] == tier
        ]
        print(f"  Tier {tier}: {[h[0] for h in hands_in_tier]}")

    print("\n=== BOARD_TEXTURE_WEIGHTS ===")
    for texture, weights in BOARD_TEXTURE_WEIGHTS.items():
        print(f"  {texture:10s} → made_hand={weights['made_hand_mod']}x  "
              f"draw={weights['draw_mod']}x  bluff_ev={weights['bluff_ev_mod']}x")

    print("\n=== HAND_KEY_BY_LABEL (amostra) ===")
    samples = ["AA", "AKs", "AKo", "JTs", "65s", "22"]
    for label in samples:
        key = HAND_KEY_BY_LABEL.get(label, "não encontrado")
        print(f"  {label:5s} → {key}")

    print("\n=== POSITION_GROUP reverse index ===")
    for seat, group in POSITION_GROUP.items():
        print(f"  {seat} → {group}")
