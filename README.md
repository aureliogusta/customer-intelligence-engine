# Revenue Intervention Engine

### Event-Driven Data Pipeline & Predictive Decision Engine


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/postgresql-13+-336791.svg)](https://www.postgresql.org/)
[![ML Stack](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost-orange.svg)]()
[![Real-time](https://img.shields.io/badge/real--time-WebSocket%20API-green.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-event--driven-lightgrey.svg)]()



## 🎯 Visão Geral

Plataforma orientada a eventos e dados (event-driven/data-driven) que orquestra a ingestão de dados de múltiplas fontes, processa métricas em tempo real (<20ms) e utiliza modelos de Machine Learning para automatizar decisões de negócio e disparar intervenções sistêmicas.

Transforma dados de CRM, produto e suporte em um pipeline contínuo de scoring, predição e execução de ações — sem intervenção manual. Tudo para auxiliar equipes ou gestores competentes a controlar carteiras enterprise, mid-market ou low-market sem a necessidade de contatar equipes de CS e Account Manager, trabalhando com um sistema de predição automática de métricas gerais, focadas no churn para retenção e geração de receita (upsell e crosell)

### 💰 Impacto de Negócio e do Sistema

**Negócio:**
- Identifica e mapeia contas em risco com até 30 dias de antecedência
- Prioriza automaticamente clientes com maior MRR em risco
- Executa ações sem intervenção manual e humana
- Permite medir receita potencialmente preservada para ação

**Sistema:**
- Processamento de eventos em batch e streaming com deduplicação idempotente
- Latência de scoring garantida em <20ms via WebSocket
- Arquitetura de serviços isolados e escaláveis via Docker
- Pipeline de ML com retraining automático em caso de drift

**Exemplo:**
- 100 contas analisadas na carteira
- 20 contas em risco
- R$ 200k em MRR em risco
- Sistema prioriza top 5 contas críticas automaticamente
- Dispara relatórios de métricas e aciona via WhatsApp, E-mail, SMS ou CRM (FastAPI) para a equipe automaticamente (Diário, Semanal, Quinzenal e Mensal)
- Determina intervenção e ação através de Agente IA (LLM)
- Sistema armazena e reaprende de acordo com a intervenção humana
- CLI Rust usa tool de healthcode para acrescentar/modificar a própria estrutura (datebase ou code) para melhorar constantemente;
- Sistema se adapta num ciclo de começo-meio-fim;
- Avalia capacidade de intervenção realizada, ajusta e reporta para líderes e gestores;

### Principais Capacidades da Engine:

- 📊 **Scoring em Tempo Real** — 
Health score instantâneo com latência <20ms para integração em CRM (Hubspot, Salesforce...) ou Frontend/Dashboard próprio
- 🤖 **Decisão Automática** — 
Políticas baseadas em Machine Learning, regras de negócio pré-estabelecida e ajustada de acordo com a empresa;
- 📈 **Previsão de Churn** — 
Modelos treinados (XGBoost, Random Forest, CLI Rust) para antecipar qualquer probabilidade de churn baixo
- 🔔 **Multi-Channel Dispatch** — Email, Slack, SMS, API webhooks, Dashboard próprio
- 📋 **Auditoria 100% Rastreável** — Log imutável de todas as ações realizadas pelo usuário e pelo próprio agente IA
- 🧠 **Agente LLM Integrado** — Análise contextual com Llama/GPT/Claude para veredito e estudo dos dados para intervenção;
- 🔄 **ETL Robusto** — Deduplicação, idempotência, retry automático, ingestão automática e em tempo real; reorganização de dados sem intervenção humana;

---

## 📋 Índice

- [Stack Tecnológico](#-stack-tecnológico)
- [Arquitetura](#-arquitetura)
- [Quick Start](#-quick-start)
- [Componentes Principais](#-componentes-principais)
- [Use Cases](#-use-cases)
- [Instalação](#-instalação)
- [Roadmap](#-roadmap)
- [Contribuindo](#-contribuindo)

---

## 🛠️ Stack Tecnológico

| Layer | Tecnologias |
|-------|-------------|
| **Backend API** | FastAPI, WebSocket, REST |
| **ML/Data** | Scikit-learn, XGBoost, Pandas, NumPy |
| **Database** | PostgreSQL 13+, pgvector (embeddings), SQLite (cache) |
| **Streaming** | Watchdog (file monitoring), JSONL logs |
| **LLM** | Ollama (local), Llama 3.2, GPT-4 (optional) |
| **Infra** | Docker, Docker Compose, systemd |
| **Monitoring** | Prometheus metrics, custom audit trails |
| **Reporting** | Matplotlib, Pandas, Jinja2 templates |

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Source Data Input                       │
│   (Salesforce, HubSpot, Intercom, Producto, Support Tickets)    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼────────────┐
                    │   INGESTION LAYER     │
                    │ (event_watcher.py)    │
                    │ - Polling contínuo    │
                    │ - Deduplicação hash   │
                    │ - Batch insert        │
                    └──────────────┬────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │   ANALYTICS LAYER          │
                    │ - Health Score Calculation │
                    │ - Performance Drift        │
                    │ - Risk Detection (5-step)  │
                    └──────────────┬─────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
    ┌───────▼────────┐    ┌────────▼──────────┐   ┌──────▼──────────┐
    │ DECISION LAYER │    │   ML LAYER        │   │  REALTIME LAYER │
    │ - Rule engine  │    │ - Churn predictor │   │ - WebSocket API │
    │ - Policies     │    │ - Expansion model │   │ - Low latency   │
    └───────┬────────┘    └────────┬──────────┘   └──────┬──────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    ACTION DISPATCH LAYER     │
                    │  - Email, Slack, SMS, API    │
                    │  - Retry logic & fallback    │
                    │  - Result tracking           │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PERSISTENCE & AUDIT        │
                    │  - PostgreSQL (main DB)     │
                    │  - JSONL logs (audit trail) │
                    │  - SQLite (distributed)     │
                    └──────────────────────────────┘

```

---

## ⚡ Quick Start

### 1️⃣ Usando Docker (Recomendado)

```bash
# Clone e setup
git clone <repo-url>
cd cs-churn-predictor
docker-compose up -d

# Aguarde 10s para PostgreSQL ficar pronto
sleep 10

# Treina modelos (primeira vez)
docker-compose exec app python notebooks/02_training.ipynb

# API disponível em http://localhost:8001
# Swagger Docs: http://localhost:8001/docs
```

### 2️⃣ Instalação Local

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

# Dependências
pip install -r requirements.txt

# Setup
export MODELS_DIR="./ml/models"
export PGVECTOR_DSN="postgresql://user:pass@localhost:5432/churn"
mkdir -p ml/models logs data

# Treina modelos
python notebooks/02_training.ipynb

# Inicia API
uvicorn app:app --reload --port 8001
```

### 3️⃣ Primeira Predição

```bash
curl -X POST http://localhost:8001/api/predict-churn/ACC_001234 \
  -H "Content-Type: application/json" \
  -d '{
    "engajamento_pct": 35.0,
    "nps_score": 4,
    "tickets_abertos": 2,
    "dias_no_contrato": 180,
    "engagement_trend": -5.0,
    "tickets_trend": 1.0,
    "nps_trend": -1.0,
    "dias_sem_interacao": 14,
    "mrr": 2500,
    "max_users": 15,
    "segment": "MID_MARKET"
  }'
```

**Resposta:**
```json
{
  "churn_risk": 0.82,
  "risk_level": "HIGH",
  "retention_prob": 0.18,
  "mrr_at_risk": 2050,
  "session_id": "sess_abc123...",
  "recommended_actions": [
    {"action": "ESCALATE_TO_MANAGER", "reason": "Conta crítica"},
    {"action": "SEND_SLACK_ALERT", "reason": "Risco alto"},
    {"action": "SEND_EMAIL", "reason": "Re-engajamento"}
  ]
}
```

---

## 🔧 Componentes Principais

### 1. **Ingestion Layer** (`ingestion/`)
Coleta dados de múltiplas fontes com deduplicação automática e idempotência.

| Módulo | Função |
|--------|--------|
| `event_watcher.py` | Polling contínuo, detecta arquivos novos, batch insert |
| `event_parser.py` | Regex + dataclasses para normalização |
| `ingestion_service.py` | Orquestrador com suporte a múltiplos adaptadores |
| `schema.sql` | Tabelas: accounts, interactions, events com índices MB tree |

**Destaques:**
- ✅ Deduplicação ON CONFLICT DO NOTHING
- ✅ Suporta CSV, JSON, API webhooks
- ✅ Modo batch (histórico) e streaming (tempo real)

---

### 2. **Analytics Layer** (`analytics/`)
Calcula métricas de saúde, detecta desvios e problemas de risco.

| Módulo | Responsabilidade |
|--------|-----------------|
| `stats_engine.py` | Health score agregado (engajamento, NPS, tickets, churn risk) |
| `performance_drift.py` | Detecção de regressão por período |
| `leak_analysis/` | Pipeline 5-etapas (detect → context → severity → plan → report) |
| `audit_trail.py` | Log JSONL 100% rastreável |

**Health Score = f()**
```
score = (
  0.35 * engagement_ratio +        # uso do produto
  0.25 * nps_normalized +          # satisfação
  0.20 * (1 - tickets_open/limit) +  # suporte
  0.15 * retention_trend +         # direção
  0.05 * expansion_signal           # oportunidade
) * segment_multiplier
```

---

### 3. **ML Layer** (`ml/` + `decision_service/`)
Modelos preditivos treinados em histórico real.

| Modelo | Target | Features | Algoritmo |
|--------|--------|----------|-----------|
| **Churn Predictor** | `renovado` (bool) | 11 features | GradientBoosting (n_estimators=100) |
| **Expansion Model** | `fez_upsell` (bool) | 8 features | RandomForest (n_estimators=200) |

**Features de Input:**
```python
[
  "engajamento_pct",       # % features ativas
  "nps_score",             # Net Promoter Score (1-10)
  "tickets_abertos",       # Support tickets
  "dias_no_contrato",      # Tempo de cliente
  "engagement_trend",      # Δ % últimos 30d
  "tickets_trend",         # Δ tickets
  "nps_trend",             # Δ NPS
  "dias_sem_interacao",    # Inatividade
  "mrr",                   # Revenue mensal
  "max_users",             # Limite do plano
  "segment_encoded"        # SMB(0), MID(1), ENTERPRISE(2)
]
```

**Performance:**
- AUC-ROC: 0.85+ 
- Avg Precision: 0.82+
- Test rows: ~200 (20% holdout)

---

### 4. **Decision Layer** (`revenue_automation/`)
Políticas inteligentes baseadas em YAML + ML.

**Fluxo:**
```
Análise → PolicyEngine.evaluate(context) → ações priorizadas
          ↓
        Condições (AND):
          - churn_risk > 0.70?
          - mrr > 5000?
          - dias_sem_interacao > 7?
          ↓
        Ações prioritizadas:
          - ESCALATE_TO_MANAGER (priority=1)
          - SEND_SLACK_ALERT
          - SEND_EMAIL
          - CREATE_CRM_TASK
        ↓
        ActionDispatcher → Email, Slack, SMS, Webhooks
```

**Exemplo de Regra (default_rules.yaml):**
```yaml
- name: critical_high_mrr
  priority: 1
  conditions:
    churn_risk: { gt: 0.80 }
    mrr: { gt: 5000 }
  actions:
    - code: ESCALATE_TO_MANAGER
      channels: [slack, email, console]
    - code: SEND_EMAIL
      channels: [email]
  reason: "Conta crítica: risco alto + MRR elevado"
```

---

### 5. **Real-Time Layer** (`realtime/`)
API WebSocket com dois modos de latência.

```
Cliente → WebSocket → latency_manager.decide()
                        ├─ FAST (18ms):  regras simples
                        └─ DEEP (300ms): ML full inference
                      ← Score + recomendação
```

**Use case:** Dashboard ao vivo de CSMs, chatbot, alertas instantâneos.

---

### 6. **Storage Layer** (`storage/`)
Perfil incremental de cliente + base de conhecimento.

| Componente | Função |
|------------|--------|
| `account_tracker.py` | Acumula stats (numerador/denominador), SQLite com WAL |
| `knowledge_base.py` | Playbooks estruturados para injetar em prompts LLM |
| `session_store.py` | Persistência de contexto de conversação (JSONL) |

---

### 7. **Reporting Layer** (`reporting/`)
Relatórios executivos e análises por CSM.

| Saída | Frequência |
|-------|-----------|
| `build_production_master_report.py` | Demanda (para stakeholders) |
| `gerar_relatorio_semanal.py` | Semanal (PDF por CSM) |
| Dashboards (Grafana/Metabase) | Real-time |
| Audit trail export (CSV) | On-demand |

---

## 💡 Use Cases Reais

### Caso 1: Redução de Churn
```
Problema: Conta de $50k com engagement caindo, NPS = 3
Solução:
  1. Analytics detecta padrão (engagement ↓ 30% vs mês anterior)
  2. ML prediz churn_risk = 0.82
  3. PolicyEngine dispara ESCALATE_TO_MANAGER
  4. ActionDispatcher envia Slack → CSM
  5. CSM agendar call de re-engajamento
  6. Auditoria registra tudo: timestamp, action, outcome
Resultado: Conta salva, histórico documentado
```

### Caso 2: Oportunidade de Expansão
```
Problema: Conta usando só 30% das features, pero com alto NPS
Solução:
  1. Analytics calcula gap (features_contratadas vs features_usadas)
  2. ML prediz expansion_prob = 0.78
  3. PolicyEngine dispara SCHEDULE_QBR
  4. Gera proposta de upsell automática
  5. CSM apresenta na QBR, fecha deal
Resultado: Aumento de $10k ARR, cliente mais satisfeito
```

### Caso 3: Detecção de Problema Latente
```
Problema: Conta com scores normais, mas tickets subindo
Solução:
  1. Leak_analysis → detecção de padrão (tickets ↑, dias_resposta ↑)
  2. Severity scorer → impacto = $5k ARR
  3. Study planner → recomenda análise raiz
  4. LLM agent → gera insights: "problema de feature X"
  5. Produto team recebe alert, planeja fix
Resultado: Problema evitado antes de churn, cliente mantido
```

---

## 📦 Instalação Completa

### Pré-requisitos
- Python 3.10+
- PostgreSQL 13+ (com extensão `pgvector`)
- Docker + Docker Compose (opcional but recommended)
- Git

### Setup de Desenvolvimento

```bash
# 1. Clone e crie venv
git clone <repo>
cd cs-churn-predictor
python -m venv venv
source venv/bin/activate

# 2. Instale dependências
pip install -r requirements.txt

# 3. Configure banco de dados
# Opção A: PostgreSQL local
createdb churn
psql -d churn -f ingestion/schema.sql

# Opção B: Docker (fica esperando porta 5432)
docker run -d --name postgres-churn \
  -e POSTGRES_DB=churn \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:13-alpine

# 4. Gere dados de treino
python data/generate_training_data.py

# 5. Treine modelos
jupyter notebook notebooks/02_training.ipynb
# Execute todas as células

# 6. Inicie API
uvicorn app:app --reload --port 8001

# 7. Teste
curl http://localhost:8001/api/health
```

### Variáveis de Ambiente

```bash
# Obrigatórias
export MODELS_DIR="./ml/models"
export DATA_DIR="./data"
export LOGS_DIR="./logs"

# Database (opcional, usa memória se não setado)
export PGVECTOR_DSN="postgresql://user:password@localhost:5432/churn"
export EMBEDDING_PROVIDER="hash"  # "hash" (offline) ou "openai"

# API
export API_HOST="0.0.0.0"
export API_PORT=8001

# ML
export HIGH_RISK_THRESHOLD=0.70
export MEDIUM_RISK_THRESHOLD=0.40
export RETRAIN_ON_DRIFT=true

# Alertas
export MRR_ALERT_THRESHOLD=5000
```

---

## 📊 Estrutura de Diretórios

```
cs-churn-predictor/
├── ingestion/              # ETL: coleta e normalização
│   ├── event_watcher.py
│   ├── event_parser.py
│   ├── ingestion_service.py
│   └── schema.sql
├── analytics/              # Scoring e detecção
│   ├── stats_engine.py
│   ├── performance_drift.py
│   ├── audit_trail.py
│   └── leak_analysis/      # 5-step analysis pipeline
├── decision_service/       # ML models
│   ├── models.py
│   ├── training.py
│   ├── inference.py
│   ├── query_engine.py
│   └── memory.py
├── ml/                     # Model artifacts
│   └── models/
│       ├── churn_model.pkl
│       ├── churn_scaler.pkl
│       └── expansion_model.pkl
├── revenue_automation/     # Policies & actions
│   ├── engine.py
│   ├── policy/             # YAML rules engine
│   ├── dispatch/           # Multi-channel dispatcher
│   │   └── channels/       # Email, Slack, SMS, API
│   └── schemas/
├── realtime/               # WebSocket API
│   ├── websocket_server.py
│   └── latency_manager.py
├── reporting/              # Dashboards & exports
│   ├── build_production_master_report.py
│   └── gerar_relatorio_semanal.py
├── storage/                # Persistence
│   ├── account_tracker.py
│   ├── knowledge_base.py
│   └── session_store.py
├── notebooks/              # Jupyter analysis
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_metrics.ipynb
├── tests/                  # Unit tests
├── app.py                  # FastAPI main
├── config/                 # Configuration
├── requirements.txt
└── docker-compose.yml
```

---

## 🚀 API Endpoints

### Health & Setup
```http
GET /api/health
GET /api/setup
```

### Predições
```http
POST /api/predict-churn/{account_id}
POST /api/predict-manual
GET  /api/accounts
POST /api/batch-predict
POST /api/drift-check
```

### Intervenções
```http
POST /api/interventions/run/{account_id}
POST /api/interventions/batch
GET  /api/interventions/summary
```

### Analytics
```http
GET /api/audit?account_id=ACC_001&limit=10
GET /api/performance
GET /api/reports/{period}
```

**Documentação Interativa:** http://localhost:8001/docs (Swagger UI)

---

## 📈 Roadmap

- [ ] **Dashboard Web** (React + D3.js real-time)
- [ ] **Explicabilidade** (SHAP values por predição)
- [ ] **Multi-Tenancy** (múltiplos workspaces)
- [ ] **Integração Salesforce** (bi-directional sync)
- [ ] **A/B Testing de Políticas** (test rule variations)
- [ ] **Mobile App** (alerts/action trigger)
- [ ] **Fine-tuning LLM** (custom domain knowledge)
- [ ] **Integração Tableau/Looker** (BI conectado)

---

## 🤝 Contribuindo

1. **Fork** este repositório
2. **Create** uma branch (`git checkout -b feature/nova-feature`)
3. **Commit** suas mudanças (`git commit -m 'Add: nova funcionalidade'`)
4. **Push** para a branch (`git push origin feature/nova-feature`)
5. **Open** um Pull Request

**Padrões:**
- Python: PEP 8, type hints, docstrings
- Commits: conventional commits (feat:, fix:, docs:, etc)
- Tests: pytest, >80% coverage

---

## 📝 License

MIT License — Veja [LICENSE](LICENSE) para detalhes.

---

## 👥 Autores

- **Aurelio** — Engenheiro de Sistemas
  - GitHub: [@aurel](https://github.com/aurel)
  - Expertise: Data Engineering, Event-Driven Architecture, Backend (Python), Machine Learning

---

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/you/cs-churn-predictor/issues)
- **Email**: engineering@yourcompany.com
- **Slack**: #churn-predictor

---

## 🙏 Agradecimentos

Arquitetura inspirada em padrões enterprise de:
- claw-code (QueryEngine, VectorMemory, PolicyEngine)
- event-driven systems (Kafka, Pub/Sub)
- MLOps best practices (model serving, monitoring, retraining)

---

**Last Updated:** April 8, 2026

Made with ❤️ for high-performance data operations.
