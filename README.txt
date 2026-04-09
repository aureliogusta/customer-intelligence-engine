================================================================================
  CS OPS — REPOSITÓRIO DE REUTILIZAÇÃO
  Baseado na arquitetura do projeto poker-engine / statistics (Aurelio)
  Criado em: 2026-04-08
================================================================================

Este repositório contém arquivos extraídos de um projeto de análise de poker
cuja arquitetura é diretamente reutilizável para automação de Customer Success
Operations (CS Ops). A lógica de domínio (poker) é irrelevante — o que importa
são os padrões de engenharia.

O pipeline central é:

  INGESTÃO → SCORING → DECISÃO → AÇÃO

Que no contexto de CS Ops se traduz em:

  CRM/Produto/Tickets → Health Score → Playbook Engine → Alerta/Tarefa/Relatório


================================================================================
  ARQUITETURA GERAL
================================================================================

  ingestion/          Coleta e normaliza dados de múltiplas fontes
  analytics/          Calcula scores, detecta desvios e problemas (leaks)
  decision/           Prioriza e recomenda intervenções
  decision_service/   Treinamento e inferência do modelo de decisão
  ml/                 Machine Learning: churn, expansão, saúde
  realtime/           API WebSocket para scoring em tempo real
  reporting/          Relatórios semanais e dashboards executivos
  storage/            Perfil de entidade (conta/cliente) e base de conhecimento
  study_service/      Gerador de recomendações personalizadas
  study_agent.py      Agente LLM que analisa dados e diz o que melhorar


================================================================================
  PRIORIDADE ALTA — Reutilizar quase sem mudança
================================================================================

------------------------------------------------------------------------
ARQUIVO: ingestion/hand_history_watcher.py
EQUIVALENTE CS OPS: Monitor de eventos de clientes
------------------------------------------------------------------------
O que faz:
  Fica em polling contínuo de uma pasta, detecta arquivos novos, parseia,
  deduplica por hash e insere no Postgres em batch. Suporta múltiplos
  formatos (ACR + PokerStars) via adaptador.

Como adaptar para CS Ops:
  - Trocar a pasta monitorada pela pasta de exportação do CRM/Intercom/Zendesk
  - Trocar o parser (hand_history_parser.py) pelo parser do formato de cada fonte
  - O mecanismo de deduplicação (ON CONFLICT DO NOTHING), polling e batch
    funcionam sem alteração
  - Usar --once para carga histórica, modo contínuo para produção

Casos de uso:
  - Ingerir eventos de uso do produto (login, feature usage, NPS response)
  - Monitorar exportações diárias de tickets de suporte
  - Processar webhooks de pagamento salvos em disco

------------------------------------------------------------------------
ARQUIVO: ingestion/hand_history_parser.py
EQUIVALENTE CS OPS: Parser de dados não estruturados de clientes
------------------------------------------------------------------------
O que faz:
  Usa regex e dataclasses para converter texto não estruturado em registros
  tipados com campos bem definidos (hero, ações por rua, resultado, etc.).

Como adaptar para CS Ops:
  - Substituir as regex de mãos de poker por regex do formato do seu CRM
  - Manter o padrão de dataclass como schema de saída (Account, Interaction,
    HealthEvent, etc.)
  - O padrão de "parse uma linha por vez, acumula em objeto, emite ao final"
    é reusável sem mudança estrutural

Casos de uso:
  - Parsear exportações CSV/JSON do Salesforce para schema normalizado
  - Converter logs de eventos do produto em registros de interação
  - Processar histórico de tickets do Zendesk

------------------------------------------------------------------------
ARQUIVO: ingestion/ingestion_service.py
EQUIVALENTE CS OPS: Serviço de ingestão idempotente multi-fonte
------------------------------------------------------------------------
O que faz:
  Envelope de ingestão com deduplicação por hash, adaptador para múltiplos
  parsers, controle de estado por arquivo já processado.

Como adaptar para CS Ops:
  - Registrar cada fonte de dados como um "adapter" (Salesforce, HubSpot,
    Intercom, banco interno)
  - O hash de deduplicação pode ser feito sobre (account_id + event_type +
    timestamp) para garantir idempotência
  - Funciona tanto para ingestão em batch quanto streaming

------------------------------------------------------------------------
ARQUIVO: ingestion/schema.sql
EQUIVALENTE CS OPS: Schema base do banco de dados
------------------------------------------------------------------------
O que faz:
  Define tabelas (sessions, hands, actions), índices para queries de ML,
  e views para análise agregada por posição e por mão.

Como adaptar para CS Ops:
  Renomear as entidades:
    sessions  → accounts        (metadados de cada conta)
    hands     → interactions    (cada contato, ticket, NPS, renovação)
    actions   → events          (ações granulares por interação)

  Views a criar (mesmo padrão das views existentes):
    v_stats_by_segment          (health score por segmento/indústria)
    v_stats_by_csm              (performance por gerente de conta)
    v_account_trend             (evolução por conta ao longo do tempo)

------------------------------------------------------------------------
ARQUIVO: analytics/stats_engine.py
EQUIVALENTE CS OPS: Motor de health score
------------------------------------------------------------------------
O que faz:
  Calcula métricas agregadas (VPIP, PFR, agressividade, WTSD), compara
  contra benchmarks de referência, detecta desvios e gera alertas
  por threshold.

Como adaptar para CS Ops:
  Substituir as métricas de poker pelas métricas de saúde do cliente:
    VPIP (engajamento)        → DAU/WAU ratio, login frequency
    PFR (iniciativa)          → features ativas, expansão de uso
    VPIP/PFR gap (passividade)→ gap entre features contratadas e usadas
    WTSD (conversão)          → taxa de renovação, upsell
    BB/100 (resultado)        → NRR (Net Revenue Retention) por conta

  A lógica de threshold, benchmark e alerta funciona sem mudança.

------------------------------------------------------------------------
ARQUIVO: analytics/performance_drift.py
EQUIVALENTE CS OPS: Detector de regressão de clientes
------------------------------------------------------------------------
O que faz:
  Compara performance atual vs período anterior, identifica regressão
  por segmento (posição), filtra outliers.

Como adaptar para CS Ops:
  - Comparar saúde da conta semana a semana / mês a mês
  - Detectar contas que eram saudáveis e estão piorando (early churn signal)
  - Segmentar por CSM, indústria, tamanho de conta

------------------------------------------------------------------------
PASTA: analytics/leak_analysis/
EQUIVALENTE CS OPS: Motor de detecção de problemas por cliente
------------------------------------------------------------------------
O que faz (módulo completo):
  Pipeline de 5 etapas — detectar problemas → contextualizar → pontuar
  severidade → gerar recomendações → validar. Cada etapa é um módulo
  separado e independente.

  leak_detector.py    → Detecta padrões problemáticos nos dados
  context_analyzer.py → Contextualiza: em que situação ocorre o problema
  severity_scorer.py  → Pontua: qual o impacto financeiro / de risco
  study_planner.py    → Recomenda: o que fazer para corrigir
  report_generator.py → Formata saída para humanos e sistemas
  analysis_utils.py   → Utilitários compartilhados (deduplica, classifica)
  validation.py       → Valida integridade dos dados antes de processar

Como adaptar para CS Ops:
  Leak detector:   detectar padrões de risco (uso caindo, tickets subindo,
                   NPS negativo, renovação se aproximando sem engajamento)
  Context:         em qual fase do ciclo de vida o cliente está
  Severity scorer: impacto em ARR — conta de $50k em risco vale mais
  Study planner:   gerar plano de ação (call, QBR, oferta de desconto,
                   treinamento, escalação)

  A estrutura modular permite substituir um módulo por vez sem quebrar o resto.

------------------------------------------------------------------------
ARQUIVO: ingestion/backfill_actions_from_hands.py
ARQUIVO: ingestion/backfill_upsert_histories.py
EQUIVALENTE CS OPS: Carga histórica do banco
------------------------------------------------------------------------
O que faz:
  Carrega dados históricos em bulk com upsert (INSERT ON CONFLICT UPDATE),
  sem duplicar registros já existentes.

Como adaptar para CS Ops:
  - Usar para seed inicial: carregar 12 meses de histórico de contas,
    tickets e interações antes de ligar o pipeline em tempo real
  - O padrão de upsert por chave única funciona sem mudança


================================================================================
  PRIORIDADE MÉDIA — Adaptar a lógica
================================================================================

------------------------------------------------------------------------
ARQUIVO: decision/decision_engine.py
EQUIVALENTE CS OPS: Motor de priorização de intervenções
------------------------------------------------------------------------
O que faz:
  Hierarquia de regras em camadas: regra base → ajuste por perfil do
  oponente → contexto especial (ICM/bolha) → ajuste de ML → ação final.

Como adaptar para CS Ops:
  Substituir as camadas por:
    Regra base           → playbook padrão por segmento (SMB, Mid-Market, Enterprise)
    Ajuste por perfil    → histórico do cliente (respondeu bem a calls? a emails?)
    Contexto especial    → perto da renovação, cliente em downgrade, NPS < 7
    Ajuste de ML         → modelo treinado no histórico de intervenções bem-sucedidas
    Ação final           → SEND_SURVEY / SCHEDULE_CALL / ESCALATE / SEND_CONTENT

------------------------------------------------------------------------
PASTA: decision_service/
EQUIVALENTE CS OPS: Treinamento e inferência do modelo de intervenção
------------------------------------------------------------------------
  dataset.py   → monta o dataset de treino a partir do histórico do Postgres
  training.py  → treina o modelo (Random Forest / XGBoost)
  models.py    → define a estrutura do modelo e seus hiperparâmetros
  inference.py → aplica o modelo em produção para recomendar ação

Como adaptar para CS Ops:
  - Feature de treino: engajamento, NPS histórico, tickets, tempo de contrato
  - Label: renovaram? fizeram upsell? churnaram?
  - O pipeline dataset→training→inference funciona sem mudança estrutural

------------------------------------------------------------------------
PASTA: ml/
EQUIVALENTE CS OPS: ML de churn, saúde e expansão
------------------------------------------------------------------------
  ml_engine.py         → Motor principal: feature engineering + Scikit-Learn
  ml_auto_trainer.py   → Retreina automaticamente quando chegam dados novos
  ml_study_collector.py→ Coleta dados do Postgres, enriquece e exporta para treino

Como adaptar para CS Ops:
  ml_engine.py:
    - Substituir features de poker (stack, posição, M-ratio) por features de CS
      (dias desde último login, tickets abertos, NPS score, MRR, contrato restante)
    - Output: probabilidade de churn / probabilidade de expansão

  ml_auto_trainer.py:
    - Disparar retreino semanal quando chegam novos dados de renovação
    - Cooldown configurável para não retreinar toda hora

  ml_study_collector.py:
    - Coletar dados do Postgres de CS, enriquecer com contexto externo (indústria,
      tamanho) e exportar dataset limpo para treino

------------------------------------------------------------------------
PASTA: realtime/
ARQUIVO: realtime/websocket_server.py
EQUIVALENTE CS OPS: API de scoring em tempo real
------------------------------------------------------------------------
O que faz:
  WebSocket server com dois modos de latência:
    Fast (18ms): resposta imediata baseada em regras
    Deep (300ms): análise completa com ML

Como adaptar para CS Ops:
  - Modo fast: scoring de saúde instantâneo para chatbot de CS ou dashboard ao vivo
  - Modo deep: análise completa com recomendação para revisão de conta

  latency_manager.py:
    - Gerencia SLA de latência e decide qual caminho tomar
    - Reutilizável sem mudança

------------------------------------------------------------------------
PASTA: reporting/
EQUIVALENTE CS OPS: Relatórios executivos e por CSM
------------------------------------------------------------------------
  build_production_master_report.py:
    Agrega análises em relatório completo para stakeholders.
    Adaptar: métricas de CS (churn rate, NRR, health score distribution,
    interventions completed, revenue at risk)

  gerar_relatorio_semanal.py:
    Gera PDF semanal com gráficos (matplotlib).
    Adaptar: enviar por email para cada CSM com as contas da sua carteira

  shared_logic.py:
    Funções compartilhadas (atomic write, cálculo de bubble factor, etc.)
    Reutilizar sem mudança.

------------------------------------------------------------------------
ARQUIVO: storage/entity_tracker.py  (era villain_tracker.py)
EQUIVALENTE CS OPS: Perfil incremental de cliente
------------------------------------------------------------------------
O que faz:
  Acumula estatísticas por entidade (numerador/denominador), mantém cache
  derivado, armazena em SQLite com WAL mode para leitura concorrente.

Como adaptar para CS Ops:
  - Entidade: conta (account_id) em vez de jogador (villain_name)
  - Métricas acumuladas: NPS histórico, tickets abertos/fechados, logins,
    features usadas, última interação, próxima renovação
  - A estrutura numerador/denominador (ex: tickets_resolvidos/tickets_abertos)
    é exatamente o padrão de saúde de CS

------------------------------------------------------------------------
ARQUIVO: storage/knowledge_base.py
EQUIVALENTE CS OPS: Playbooks de CS para o agente LLM
------------------------------------------------------------------------
O que faz:
  Lookup O(1) de dados estruturados (dicts/frozensets) para injetar
  contexto em prompts de LLM.

Como adaptar para CS Ops:
  - Substituir ranges de GTO por playbooks de CS:
    por segmento (SMB, Mid-Market, Enterprise)
    por motivo de risco (baixo engajamento, NPS negativo, tickets em aberto)
    por fase do ciclo (onboarding, adoção, renovação, expansão)
  - O padrão de lookup rápido e injeção em prompt funciona sem mudança

------------------------------------------------------------------------
ARQUIVO: storage/session_store.py
EQUIVALENTE CS OPS: Persistência de contexto de conversação
------------------------------------------------------------------------
O que faz:
  Serializa/deserializa contexto de sessão LLM em JSONL, isolado por
  session_id, com contadores de tokens.

Como adaptar para CS Ops:
  - Guardar o contexto de cada conversa de CS (chamada, chat, email thread)
    para continuidade em múltiplos turnos
  - Associar ao account_id para histórico por conta

------------------------------------------------------------------------
PASTA: study_service/
ARQUIVO: study_service/recommendations.py
EQUIVALENTE CS OPS: Gerador de próxima melhor ação
------------------------------------------------------------------------
O que faz:
  A partir da análise de leaks, gera recomendações concretas e priorizadas.

Como adaptar para CS Ops:
  - Input: resultado do leak_analysis (conta X tem risco Y de severidade Z)
  - Output: lista priorizada de ações (CALL, SEND_SURVEY, QBR, DISCOUNT_OFFER)
  - Integrar com calendar/CRM para criar tarefas automaticamente

------------------------------------------------------------------------
ARQUIVO: study_agent.py
EQUIVALENTE CS OPS: Agente LLM de análise e recomendação
------------------------------------------------------------------------
O que faz:
  Puxa dados do Postgres, calcula métricas, monta contexto estruturado,
  envia ao Llama (Ollama local) via streaming e exibe análise no terminal.
  Tem modo de identificação de casos específicos problemáticos.

Como adaptar para CS Ops:
  - Trocar as queries de poker pelas queries de saúde do cliente
  - Trocar o system prompt de "coach de poker" por "coach de CS"
  - O pipeline Postgres → contexto → LLM streaming funciona sem mudança
  - Adicionar flag --conta XPTO para análise de uma conta específica


================================================================================
  O QUE CRIAR DO ZERO (pequeno escopo)
================================================================================

1. CONECTORES DE FONTE
   Arquivo sugerido: ingestion/connectors/
   O que é: Integração com as APIs de cada fonte de dados do negócio.
   O que construir:
     - salesforce_connector.py  → REST API do Salesforce (SOQL queries)
     - hubspot_connector.py     → API do HubSpot (contatos, deals, tickets)
     - intercom_connector.py    → API do Intercom (conversas, eventos de usuário)
     - produto_connector.py     → API interna do produto (eventos de uso, features)
   Como conectar ao pipeline existente:
     Cada conector retorna o mesmo dict padronizado que o parser atual usa.
     O hand_history_watcher.py aceita o retorno via adaptador — basta
     registrar o novo conector como um adapter.

2. SCHEMA DE CS (banco de dados)
   Arquivo sugerido: ingestion/schema_cs.sql
   O que é: Substituição do schema.sql atual com entidades de CS.
   O que construir (seguindo o mesmo padrão do schema.sql existente):
     CREATE TABLE accounts (
         account_id      UUID PRIMARY KEY,
         name            TEXT NOT NULL,
         segment         VARCHAR(32),   -- SMB / MID / ENTERPRISE
         industry        VARCHAR(64),
         mrr             NUMERIC(12,2),
         contract_start  DATE,
         contract_end    DATE,
         csm_id          VARCHAR(64),
         ingested_at     TIMESTAMPTZ DEFAULT NOW()
     );
     CREATE TABLE interactions (
         id              BIGSERIAL PRIMARY KEY,
         account_id      UUID REFERENCES accounts,
         type            VARCHAR(32),   -- call / email / ticket / nps / login
         direction       VARCHAR(16),   -- inbound / outbound
         outcome         VARCHAR(32),   -- resolved / pending / escalated
         sentiment_score NUMERIC(4,2),
         date_utc        TIMESTAMPTZ,
         created_at      TIMESTAMPTZ DEFAULT NOW()
     );
     CREATE TABLE health_scores (
         account_id      UUID REFERENCES accounts,
         score_date      DATE,
         engagement_pct  NUMERIC(5,2),
         nps_score       SMALLINT,
         tickets_open    INTEGER,
         churn_risk      NUMERIC(4,3),  -- 0.0–1.0
         expansion_prob  NUMERIC(4,3),
         PRIMARY KEY (account_id, score_date)
     );
   Views a criar (mesmo padrão do schema.sql):
     v_health_by_segment     → health score médio por segmento
     v_health_by_csm         → performance por gerente de conta
     v_account_trend         → evolução por conta (últimas 12 semanas)
     v_at_risk_accounts      → contas com churn_risk > 0.6 ordenadas por MRR

3. PLAYBOOK DE DOMÍNIO
   Arquivo sugerido: storage/cs_playbooks.json  (substitui gto_ranges.json)
   O que é: Base de conhecimento de CS que o agente LLM consulta como contexto.
   O que construir:
   {
     "segments": {
       "SMB": {
         "onboarding": ["check-in 7 dias", "treinamento em grupo", "email de boas-vindas"],
         "at_risk":    ["call de descoberta", "oferta de extensão de trial"],
         "renewal":    ["QBR simplificado", "email 60 dias antes"]
       },
       "ENTERPRISE": {
         "onboarding": ["kick-off presencial", "CSM dedicado", "plano de sucesso 90 dias"],
         "at_risk":    ["escalação para gerência", "executive sponsor meeting"],
         "renewal":    ["QBR trimestral", "proposta de expansão", "business case de ROI"]
       }
     },
     "risk_thresholds": {
       "churn_high":      0.70,
       "churn_medium":    0.40,
       "expansion_ready": 0.65
     },
     "intervention_priority": ["ESCALATE", "SCHEDULE_CALL", "SEND_SURVEY", "SEND_CONTENT"]
   }


================================================================================
  PIPELINE COMPLETO PARA CS OPS (como os arquivos se conectam)
================================================================================

  [1] INGESTÃO (diária / contínua)
      connectors/ → hand_history_watcher.py → hand_history_parser.py
      → ingestion_service.py → Postgres (schema_cs.sql)

  [2] SCORING (a cada ingestão ou sob demanda)
      Postgres → stats_engine.py → performance_drift.py
      → leak_analysis/ (detect → context → severity → recommend)
      → health_scores table

  [3] DECISÃO (por conta, sob demanda ou scheduled)
      health_scores + cs_playbooks.json → decision_engine.py
      → decision_service/ (ML inference)
      → lista de intervenções priorizadas

  [4] AÇÃO (em tempo real ou batch)
      Modo real-time: websocket_server.py → dashboard ao vivo / chatbot
      Modo batch:     study_agent.py → relatório terminal
                      reporting/ → PDF semanal / dashboard executivo

  [5] APRENDIZADO CONTÍNUO (semanal)
      ml_study_collector.py → ml_engine.py → ml_auto_trainer.py
      → modelo atualizado com novos dados de outcome


================================================================================
  DEPENDÊNCIAS PYTHON
================================================================================

  pip install psycopg2-binary    # Postgres
  pip install watchdog           # monitoramento de pasta (hand_watcher)
  pip install scikit-learn       # ML engine
  pip install pandas numpy       # analytics
  pip install matplotlib         # relatórios PDF
  pip install joblib             # serialização de modelos ML
  pip install websockets         # servidor WebSocket real-time
  pip install requests           # conectores de API (Salesforce, HubSpot, etc.)

  LLM (agente de análise):
  ollama serve
  ollama pull llama3.2:3b        # modelo padrão — pode trocar por qualquer outro


================================================================================
  ORDEM DE IMPLEMENTAÇÃO SUGERIDA
================================================================================

  FASE 1 — BASE DE DADOS (1–2 dias)
    1. Criar schema_cs.sql (baseado em ingestion/schema.sql)
    2. Rodar no Postgres local
    3. Testar conexão

  FASE 2 — INGESTÃO (2–3 dias)
    4. Criar um conector simples (ex: CSV export do CRM)
    5. Adaptar hand_history_parser.py para o formato do conector
    6. Rodar hand_history_watcher.py em modo --once para carga histórica

  FASE 3 — SCORING (2–3 dias)
    7. Adaptar stats_engine.py com métricas de CS
    8. Configurar thresholds no cs_playbooks.json
    9. Verificar as views do banco funcionando

  FASE 4 — DECISÃO E AGENTE (1–2 dias)
    10. Adaptar decision_engine.py com playbook de CS
    11. Adaptar study_agent.py (trocar queries e system prompt)
    12. Testar: python study_agent.py --maos-erradas (vira --contas-em-risco)

  FASE 5 — RELATÓRIOS E ML (opcional)
    13. Adaptar gerar_relatorio_semanal.py
    14. Treinar primeiro modelo com ml_engine.py + ml_auto_trainer.py


================================================================================
  CONTATO / ORIGEM DOS ARQUIVOS
================================================================================

  Projeto original : C:\projeto-spade\statistics  (poker DSS)
  Arquitetura base : C:\Users\aurel\Downloads\claw-code-main\claw-code-main
  Gerado em        : 2026-04-08
  Autor            : Aurelio

================================================================================
