-- =============================================================================
-- schema.sql — Poker DSS Database
-- =============================================================================
-- Executa: psql -U postgres -d poker_dss -f schema.sql
--
-- Arquitetura de duas camadas:
--   DADOS QUENTES  → tabelas postgres (histórico massivo, ML, análise)
--   CONSCIÊNCIA    → knowledge_base.py, gto_ranges.json (in-memory, decisão)
-- =============================================================================

-- Cria banco se não existir (roda separado se necessário)
-- CREATE DATABASE poker_dss;
-- \c poker_dss

-- =============================================================================
-- Extensões
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- busca de texto eficiente

-- =============================================================================
-- 1. SESSÕES — metadados de cada sessão de jogo
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    session_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hero_name       VARCHAR(64)  NOT NULL,
    source_file     TEXT         NOT NULL UNIQUE,  -- caminho do .txt original
    tournament_name TEXT,
    ingested_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    hands_count     INTEGER      DEFAULT 0,
    hands_won       INTEGER      DEFAULT 0,
    net_chips       NUMERIC(12,2) DEFAULT 0
);

-- =============================================================================
-- 2. MÃOS — uma linha por mão jogada
-- =============================================================================
CREATE TABLE IF NOT EXISTS hands (
    -- Chave primária
    id              BIGSERIAL    PRIMARY KEY,
    hand_id         VARCHAR(32)  NOT NULL UNIQUE,

    -- Referência à sessão
    session_id      UUID         REFERENCES sessions(session_id) ON DELETE CASCADE,

    -- Identificação
    tournament_id   VARCHAR(32),
    tournament_name TEXT,
    table_id        TEXT,
    date_utc        TIMESTAMPTZ,

    -- Estrutura do nível
    level           SMALLINT,
    small_blind     NUMERIC(12,2),
    big_blind       NUMERIC(12,2),
    ante            NUMERIC(12,2),
    max_seats       SMALLINT,
    num_players     SMALLINT,
    btn_seat        SMALLINT,

    -- Hero
    hero_name       VARCHAR(64),
    hero_seat       SMALLINT,
    hero_position   VARCHAR(8),
    hero_stack_start NUMERIC(12,2),
    hero_stack_end   NUMERIC(12,2),
    hero_cards      VARCHAR(16),

    -- Ações por rua
    hero_action_preflop VARCHAR(256),
    hero_action_flop    VARCHAR(256),
    hero_action_turn    VARCHAR(256),
    hero_action_river   VARCHAR(256),

    -- Flags derivadas
    hero_vpip       SMALLINT DEFAULT 0,
    hero_pfr        SMALLINT DEFAULT 0,
    hero_aggressor  SMALLINT DEFAULT 0,
    hero_went_allin SMALLINT DEFAULT 0,

    -- Board
    board_flop      VARCHAR(32),
    board_turn      VARCHAR(16),
    board_river     VARCHAR(16),

    -- Resultado
    hero_result      VARCHAR(16),
    hero_amount_won  NUMERIC(12,2),
    pot_final        NUMERIC(12,2),
    went_to_showdown SMALLINT DEFAULT 0,
    m_ratio          NUMERIC(8,2),

    -- Metadados de ingestão
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- 2.1 AÇÕES — uma linha por ação da mão (street-by-street)
-- =============================================================================
CREATE TABLE IF NOT EXISTS actions (
    action_id        BIGSERIAL PRIMARY KEY,
    hand_id_ref      BIGINT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    action_order     INTEGER NOT NULL,
    street           VARCHAR(16) NOT NULL,
    actor_id         VARCHAR(96) NOT NULL,
    action_type      VARCHAR(16) NOT NULL,
    amount           NUMERIC(12,2) NOT NULL DEFAULT 0,
    pot_size_before  NUMERIC(12,2) NOT NULL DEFAULT 0,
    is_all_in        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (hand_id_ref, action_order)
);

-- =============================================================================
-- 3. ÍNDICES — performance para queries de ML e análise
-- =============================================================================

-- Queries por data (séries temporais)
CREATE INDEX IF NOT EXISTS idx_hands_date     ON hands(date_utc DESC);

-- Queries por posição (análise de leaks)
CREATE INDEX IF NOT EXISTS idx_hands_position ON hands(hero_position);

-- Queries por resultado
CREATE INDEX IF NOT EXISTS idx_hands_result   ON hands(hero_result);

-- Queries por mãos específicas (range analysis)
CREATE INDEX IF NOT EXISTS idx_hands_cards    ON hands(hero_cards);

-- Queries por sessão
CREATE INDEX IF NOT EXISTS idx_hands_session  ON hands(session_id);

-- Queries por torneio
CREATE INDEX IF NOT EXISTS idx_hands_tournament ON hands(tournament_id);

-- Busca de texto nas ações
CREATE INDEX IF NOT EXISTS idx_hands_action_preflop
    ON hands USING gin(hero_action_preflop gin_trgm_ops);

-- Queries de reconstrução de linha por mão/rua
CREATE INDEX IF NOT EXISTS idx_actions_hand_street ON actions(hand_id_ref, street);
CREATE INDEX IF NOT EXISTS idx_actions_hand_order ON actions(hand_id_ref, action_order);
CREATE INDEX IF NOT EXISTS idx_actions_actor_type ON actions(actor_id, action_type);

-- =============================================================================
-- 4. VIEW — stats agregadas por posição (alimenta Scikit-Learn)
-- =============================================================================
DROP VIEW IF EXISTS v_stats_by_position;
CREATE OR REPLACE VIEW v_stats_by_position AS
SELECT
    hero_position,
    COUNT(*)                                    AS total_hands,
    ROUND(AVG(hero_vpip) * 100, 1)             AS vpip_pct,
    ROUND(AVG(hero_pfr) * 100, 1)              AS pfr_pct,
    ROUND(AVG(hero_aggressor) * 100, 1)        AS aggression_pct,
    ROUND(AVG(hero_went_allin) * 100, 1)       AS allin_pct,
    ROUND(AVG(went_to_showdown) * 100, 1)      AS wtsd_pct,
    ROUND(SUM(hero_amount_won), 2)             AS net_chips,
    ROUND(AVG(hero_amount_won), 2)             AS avg_won_per_hand,
    ROUND(AVG(hero_stack_start / NULLIF(big_blind, 0)), 2) AS avg_stack_bb,
    ROUND(
        (
            SUM(hero_amount_won / NULLIF(big_blind, 0))
            / NULLIF(COUNT(*), 0)
        ) * 100.0,
        2
    ) AS bb_per_100
FROM hands
WHERE hero_position IS NOT NULL AND hero_position != ''
GROUP BY hero_position
ORDER BY total_hands DESC;

-- =============================================================================
-- 5. VIEW — performance por mão inicial (range analysis)
-- =============================================================================
CREATE OR REPLACE VIEW v_stats_by_cards AS
SELECT
    hero_cards,
    COUNT(*)                                    AS total_hands,
    SUM(hero_vpip)                              AS times_played,
    ROUND(SUM(hero_amount_won), 2)             AS net_chips,
    ROUND(AVG(hero_amount_won), 2)             AS avg_won,
    ROUND(AVG(went_to_showdown) * 100, 1)      AS wtsd_pct,
    COUNT(*) FILTER (WHERE hero_result = 'win') AS wins
FROM hands
WHERE hero_cards IS NOT NULL AND hero_cards != ''
GROUP BY hero_cards
HAVING COUNT(*) >= 5
ORDER BY net_chips DESC;

-- =============================================================================
-- 6. VIEW — tendência por sessão (tracking de evolução)
-- =============================================================================
CREATE OR REPLACE VIEW v_session_trend AS
SELECT
    s.session_id,
    s.tournament_name,
    s.ingested_at::DATE                         AS session_date,
    s.hands_count,
    ROUND(s.net_chips, 2)                       AS net_chips,
    ROUND(AVG(h.hero_vpip) * 100, 1)           AS vpip_pct,
    ROUND(AVG(h.hero_pfr) * 100, 1)            AS pfr_pct,
    ROUND(AVG(h.m_ratio), 1)                   AS avg_m_ratio
FROM sessions s
JOIN hands h ON h.session_id = s.session_id
GROUP BY s.session_id, s.tournament_name, s.ingested_at, s.hands_count, s.net_chips
ORDER BY s.ingested_at DESC;

-- =============================================================================
-- Confirmação
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Schema Poker DSS criado com sucesso.';
    RAISE NOTICE 'Tabelas: sessions, hands';
    RAISE NOTICE 'Views: v_stats_by_position, v_stats_by_cards, v_session_trend';
END $$;
