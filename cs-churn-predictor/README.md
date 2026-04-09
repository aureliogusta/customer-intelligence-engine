# CS Churn Predictor 2.0

**Sistema de Predição de Churn e Automação de Revenue com Memória Semântica, Políticas Inteligentes e Monitoramento em Tempo Real**

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Instalação e Configuração](#instalação-e-configuração)
4. [Como Usar](#como-usar)
5. [Guia Completo dos Componentes](#guia-completo-dos-componentes)
6. [Treinamento de Modelos](#treinamento-de-modelos)
7. [Políticas de Intervenção](#políticas-de-intervenção)
8. [Monitoramento e Analytics](#monitoramento-e-analytics)
9. [Exemplos Práticos](#exemplos-práticos)
10. [Troubleshooting](#troubleshooting)

---

## Visão Geral

O **CS Churn Predictor 2.0** é um sistema enterprise de **predição de churn de clientes** combinado com **automação de ações de retenção e expansão**. É um assistente de IA que:

- 🎯 **Prediz risco de churn** 30 dias antes usando modelos ML (GradientBoosting)
- 📊 **Recomenda ações automáticas** baseado em políticas YAML configuráveis
- 🤖 **Dispara intervenções** em múltiplos canais (Slack, Email, API, arquivo, etc)
- 💾 **Memoriza histórico semântico** de cada conta (PostgreSQL + pgvector)
- 📈 **Detecta data drift** e recomenda retreino automático
- 📋 **Audita absolutamente tudo** em trail imutável para compliance
- ⚡ **Executa em tempo real** via FastAPI (pode processar centenas de contas/minuto)

**Casos de Uso:**
- Priorizar esforços de customer success para contas críticas
- Reduzir churn automaticamente com intervenções contextualizadas
- Maximizar expansão identificando oportunidades de upsell
- Manter compliance com trilha auditável de todas as decisões

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                          │
│                      (app.py — porta 8001)                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  ChurnQueryEngine│ │ InterventionEngine│ │    Analytics     │
│  (predição + ML) │ │ (ações + políticas)│ │  (drift, auditoria)
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
    ┌────┴─────────────────────┴────────────────────┴────┐
    │                                                    │
    ▼                      ▼              ▼              ▼
┌─────────────────┐ ┌────────────────┐ ┌───────────┐ ┌──────────────┐
│ ChurnPredictor  │ │PolicyEngine    │ │ Audit     │ │DriftDetector │
│ (GradientBoost) │ │(rules YAML)    │ │ Trail     │ │ (KS + PSI)   │
└────────┬────────┘ └────────┬───────┘ │           │ └──────────────┘
         │                   │         └───────────┘
         │                   │
         │              ┌────┴──────────────────┐
         │              │                       │
         ▼              ▼                       ▼
    ┌─────────┐ ┌──────────────────┐ ┌──────────────────┐
    │Inference│ │ActionDispatcher  │ │  MemoryStore     │
    │(pred)   │ │ (multi-channel)  │ │  (pgvector)      │
    └─────────┘ └────────┬─────────┘ └──────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    ┌────────┐      ┌────────┐       ┌────────┐
    │ Slack  │      │ Email  │       │ API    │
    │Channel │      │Channel │       │ Hook   │
    └────────┘      └────────┘       └────────┘
```

### Componentes Principais

#### 1. **ChurnQueryEngine** (decision_service/query_engine.py)
Orquestra central do sistema. A cada análise:
- Carrega modelos treinados
- Recupera histórico semântico da conta
- Executa predições (churn + expansão)
- Gera recomendações
- Salva resultado na memória
- Retorna `PredictionTurn` estruturado

#### 2. **ChurnPredictor / ExpansionPredictor** (decision_service/inference.py)
Modelos treinados em disco:
- `churn_model.pkl` → Predição de renovação/churn (GradientBoosting)
- `churn_scaler.pkl` → Normalização das features
- `expansion_model.pkl` → Predição de upsell (RandomForest, opcional)

#### 3. **PolicyEngine** (revenue_automation/policy/engine.py)
Interpreta arquivo `default_rules.yaml` e decide ações:
- Lê condições estruturadas (churn_risk > 0.7, MRR > 5000, etc)
- Avalia contra contexto da análise
- Dispara regras (stop = não avalia próximas)
- Retorna `PolicyDecision` com ações e canais

#### 4. **ActionDispatcher** (revenue_automation/dispatch/dispatcher.py)
Executa ações em múltiplos canais:
- Tenta enviar em cada canal especificado
- Fallback seguro: se Slack falha, tenta Email
- Sempre tenta Console como último recurso
- Registra resultado (sucesso/falha) em cada canal

#### 5. **ChurnAuditTrail** (analytics/audit_trail.py)
Log imutável JSONL de todas as predições e ações:
- Formato: 1 JSON por linha (fácil análise)
- Campos: timestamp, session_id, account_id, event_type, details
- Permite compliance, debugging e análise histórica

#### 6. **DriftDetector** (analytics/drift_monitor.py)
Detecta mudanças na distribuição de dados (data drift):
- Usa testes estatísticos: KS (Kolmogorov-Smirnov) e PSI (Population Stability Index)
- Identifica se features críticas desviaram do treino original
- Recomenda retreino automático se drift > threshold

#### 7. **AccountMemoryStore** (decision_service/memory.py)
Memória semântica de cada conta (PostgreSQL + pgvector):
- Persiste histórico de predições e ações passadas
- Recupera contexto automático em novas análises
- Suporta proveedores offline (hash) e online (OpenAI embeddings)
- Fallback gracioso se DB não está disponível

---

## Instalação e Configuração

### Pré-requisitos
- Python 3.10+
- PostgreSQL 13+ com extension `pgvector` (opcional, para memória semântica)
- Docker + Docker Compose (para ambiente completo)
- Git

### Opção 1: Docker (Recomendado)

```bash
# Clone o repositório
cd cs-churn-predictor

# Inicie tudo (API, PostgreSQL, etc)
docker-compose up -d

# Aguarde 10s para o PostgreSQL estar pronto, depois treina modelos:
docker-compose exec app python notebooks/02_training.ipynb

# A API estará disponível em: http://localhost:8001
# Docs Swagger: http://localhost:8001/docs
```

### Opção 2: Instalação Local

```bash
# Crie um virtualenv
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

# Instale dependências
pip install -r requirements.txt

# Configure variáveis de ambiente (opcional)
export MODELS_DIR="./ml/models"
export DATA_DIR="./data"
export PGVECTOR_DSN="postgresql://user:password@localhost:5432/churn"  # opcional
export OLLAMA_URL="http://localhost:11434"  # opcional, para análise avançada
export API_PORT=8001

# Crie diretórios necessários
mkdir -p ml/models logs data

# Treina os modelos (ver seção "Treinamento" abaixo)
python notebooks/02_training.ipynb

# Inicie a API
uvicorn app:app --reload --port 8001
```

### Variáveis de Ambiente

```bash
# Paths
MODELS_DIR="./ml/models"                    # Onde guardar modelos treinados
DATA_DIR="./data"                           # Dados de treino/teste
LOGS_DIR="./logs"                           # Audit trail e logs

# Database (opcional, para memória semântica)
PGVECTOR_DSN="postgresql://user:pass@localhost:5432/churn"
EMBEDDING_PROVIDER="hash"                   # "hash" (offline) ou "openai"

# API
API_HOST="0.0.0.0"
API_PORT=8001

# ML
HIGH_RISK_THRESHOLD=0.70                    # > 0.7 = HIGH risk
MEDIUM_RISK_THRESHOLD=0.40                  # 0.4-0.7 = MEDIUM risk
RETRAIN_ON_DRIFT=true                       # Auto-retrain se drift detectado

# Alertas
MRR_ALERT_THRESHOLD=5000                    # Alertar se MRR > 5k em risco
```

---

## Como Usar

### 1. Via API REST

A API está documentada e testável em `http://localhost:8001/docs` (Swagger).

#### **GET /api/health** — Status do Sistema
```bash
curl http://localhost:8001/api/health
```
Resposta:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "database_connected": true
}
```

#### **POST /api/predict-churn/{account_id}** — Predição Individual
```bash
curl -X POST http://localhost:8001/api/predict-churn/ACC_000042 \
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

Resposta:
```json
{
  "account_id": "ACC_000042",
  "session_id": "sess_abc123...",
  "churn_risk": 0.82,
  "risk_level": "HIGH",
  "retention_prob": 0.18,
  "upsell_probability": 0.15,
  "upsell_signal": "LOW",
  "mrr": 2500,
  "mrr_at_risk": 2050,
  "recommended_actions": [
    {
      "action": "ESCALATE_TO_MANAGER",
      "reason": "Conta crítica: churn > 80% com MRR alto"
    },
    {
      "action": "SEND_SLACK_ALERT",
      "reason": "Escalamento obrigatório executado"
    },
    {
      "action": "SEND_EMAIL",
      "reason": "Enviar email de reativação"
    }
  ],
  "memory_hits": [
    {
      "text": "Conta ACC_000042 teve churno em Q2 2024...",
      "distance": 0.12
    }
  ]
}
```

#### **POST /api/batch-predict** — Predição em Massa
```bash
curl -X POST http://localhost:8001/api/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/accounts_to_check.csv"
  }'
```

#### **POST /api/interventions/run/{account_id}** — Executar Intervenção
Combina predição + política + despacho:
```bash
curl -X POST http://localhost:8001/api/interventions/run/ACC_000042 \
  -H "Content-Type: application/json" \
  -d '{
    "engajamento_pct": 35.0,
    "nps_score": 4,
    ...
  }'
```

Despachará automaticamente ações (Slack, Email, etc) acording to `revenue_automation/policy/default_rules.yaml`.

#### **GET /api/audit** — Trilha Auditável
```bash
curl "http://localhost:8001/api/audit?account_id=ACC_000042&limit=10"
```

```json
{
  "entries": [
    {
      "timestamp": "2026-04-08T15:32:10+00:00",
      "session_id": "sess_abc...",
      "account_id": "ACC_000042",
      "event_type": "prediction",
      "details": {
        "churn_risk": 0.82,
        "risk_level": "HIGH",
        "mrr_at_risk": 2050
      }
    }
  ]
}
```

#### **POST /api/drift-check** — Detectar Data Drift
```bash
curl -X POST http://localhost:8001/api/drift-check \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/recent_accounts.csv"
  }'
```

Compara distribuição vs baseline de treino e recomenda retreino se necessário.

#### **GET /api/performance** — Métricas de Performance
```bash
curl http://localhost:8001/api/performance
```

Mostra tendência de predições, alertas e métricas de negócio dos últimos N dias.

### 2. Via Python (Library)

```python
from decision_service.query_engine import ChurnQueryEngine
from revenue_automation.engine import InterventionEngine

# Inicializa o engine
engine = ChurnQueryEngine.from_env()

# Análise individual
turn = engine.analyze(
    account_id="ACC_000042",
    features={
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
        "segment": "MID_MARKET",
    }
)

# Verificar resultado
print(f"Churn Risk: {turn.churn_risk:.2%}")
print(f"Risk Level: {turn.risk_level}")
print(f"MRR at Risk: ${turn.mrr_at_risk:,.0f}")

# Executar intervenção (com políticas)
intervention_engine = InterventionEngine.from_env()
record = intervention_engine.intervene_one(
    account_id="ACC_000042",
    features={...}
)

# Ver ações despachadas
for action in record.actions_taken:
    print(f"Action: {action}")
for channel in record.channels_used:
    print(f"Channel: {channel}")
```

### 3. Via Notebooks (Análise Interativa)

Três notebooks prontos:

**01_eda.ipynb** — Exploração de Dados
- Distribuição de features
- Correlação com churn
- Casos extremos (anomalias)

**02_training.ipynb** — Treinamento dos Modelos
```python
# Executa este notebook para treinar:
# - ChurnModel (GradientBoosting)
# - ExpansionModel (RandomForest)
# Salva em ml/models/
```

**03_metrics.ipynb** — Análise de Performance
- Métricas do modelo (AUC, Precision/Recall)
- Feature importance
- Comparação antes/depois

---

## Guia Completo dos Componentes

### decision_service/

#### **models.py** — Configuração dos Modelos

Define hyperparâmetros de treino:

```python
@dataclass
class ChurnModel:
    model_type: str = "gradient_boosting"
    target: str = "renovado"  # bool
    features = [
        "engajamento_pct",       # % de features ativas
        "nps_score",             # Net Promoter Score (1-10)
        "tickets_abertos",       # Support tickets abertos
        "dias_no_contrato",      # Tempo desde assinatura
        "engagement_trend",      # Mudança % engagement últimos 30d
        "tickets_trend",         # Mudança tickets últimos 30d
        "nps_trend",             # Mudança NPS últimos 30d
        "dias_sem_interacao",    # Dias desde último login
        "mrr",                   # Monthly Recurring Revenue
        "max_users",             # Limite de usuários do plano
        "segment_encoded",       # Segment: SMB(0), MID(1), ENTERPRISE(2)
    ]
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.10
```

#### **training.py** — Treinamento

```python
from decision_service.training import ChurnTrainer

# Carregar dados
df = pd.read_csv("data/train_dataset.csv")
X = df.drop(columns=["renovado"])
y = df["renovado"]

# Treinar
config = ChurnModel()
trainer = ChurnTrainer(config)
model, report = trainer.train(X, y, feature_names=X.columns)

# Salvar
trainer.save("ml/models/churn_model.pkl")

print(f"AUC-ROC: {report.auc:.4f}")
print(f"Avg Precision: {report.avg_precision:.4f}")
```

#### **inference.py** — Predição

```python
from decision_service.inference import ChurnPredictor

predictor = ChurnPredictor(
    model_path="ml/models/churn_model.pkl",
    scaler_path="ml/models/churn_scaler.pkl"
)

result = predictor.predict({
    "engajamento_pct": 35.0,
    "nps_score": 4,
    ...
})

# result = {
#   "churn_risk": 0.82,      # float 0–1
#   "retention_prob": 0.18,
#   "risk_level": "HIGH"     # LOW / MEDIUM / HIGH
# }
```

#### **query_engine.py** — Orquestrador Central

```python
from decision_service.query_engine import ChurnQueryEngine

engine = ChurnQueryEngine.from_env()

turn = engine.analyze("ACC_000042", features_dict)
# turn.churn_risk → float 0–1
# turn.risk_level → str ("LOW" | "MEDIUM" | "HIGH")
# turn.recommended_actions → list[dict]
# turn.memory_hits → list[MemoryMatch]
# turn.mrr_at_risk → float
# turn.session_id → str UUID
```

### revenue_automation/

#### **policy/default_rules.yaml** — Políticas de Ação

Arquivo YAML que define regras de decisão:

```yaml
rules:
  - name: critical_high_mrr
    priority: 1              # 1 = mais urgente
    stop: true               # Para de avaliar após match
    conditions:
      churn_risk: { gt: 0.80 }      # > 80%
      mrr: { gt: 5000 }             # MRR > $5k
    actions:
      - code: ESCALATE_TO_MANAGER
        channels: [console, file, slack]
      - code: SEND_EMAIL
        channels: [email, console]
    reason: "Conta crítica com MRR alto"
```

**Operadores condicionais:**
- `gt` (greater than), `lt`, `gte`, `lte`, `eq`, `ne`
- `in`, `not_in` (para listas)

**Ações disponíveis:**
- `ESCALATE_TO_MANAGER` → Escalar para manager
- `CREATE_CRM_TASK` → Criar task no CRM
- `SEND_EMAIL` → Email automático
- `SEND_SLACK_ALERT` → Slack notification
- `SCHEDULE_CALL` → Agendar call automático
- `SEND_CUSTOMER_CHECKIN` → Checkin automático

**Canais de despacho:**
- `console` → Stdlib output
- `file` → Escrever em arquivo
- `slack` → Enviar para Slack
- `email` → Enviar email
- `api_hook` → Chamar API externa

#### **dispatch/ — Multi-Channel Dispatcher**

Cada canal tem sua própria função de envio:

```python
# dispatch/channels/slack.py
def send(action_code: str, payload: dict) -> bool:
    """Envia message para Slack via webhook"""
    
# dispatch/channels/email.py
def send(action_code: str, payload: dict) -> bool:
    """Envia email via SMTP"""

# dispatch/channels/api_hook.py
def send(action_code: str, payload: dict) -> bool:
    """Chama URL externa via HTTP POST"""
```

#### **engine.py — InterventionEngine**

```python
from revenue_automation.engine import InterventionEngine

engine = InterventionEngine.from_env()

# Intervir em 1 conta
record = engine.intervene_one(
    account_id="ACC_000042",
    features={...}
)

# record.actions_taken → ["ESCALATE_TO_MANAGER", "SEND_EMAIL"]
# record.channels_used → ["console", "slack", "email"]
# record.dispatch_results → [DispatchResult, ...]

# Intervir em lote
records = engine.intervene_batch(df_accounts)
```

### analytics/

#### **audit_trail.py** — Trilha Auditável

```python
from analytics.audit_trail import ChurnAuditTrail

trail = ChurnAuditTrail(log_path="logs/audit.jsonl")

trail.log_prediction(
    account_id="ACC_000042",
    session_id="sess_abc",
    churn_risk=0.82,
    risk_level="HIGH",
    mrr=2500,
    mrr_at_risk=2050,
    memory_hits=3
)

# Arquivo audit.jsonl:
# {"timestamp": "...", "session_id": "...", "account_id": "ACC_000042", ...}
# {"timestamp": "...", "session_id": "...", "account_id": "ACC_000042", ...}
# ...

# Exportar para CSV
csv = trail.to_csv()
with open("audit_export.csv", "w") as f:
    f.write(csv)
```

#### **drift_monitor.py** — Detecção de Data Drift

```python
from analytics.drift_monitor import DriftDetector

# Carregar baseline (dados de treino originais)
baseline = pd.read_csv("data/train_dataset.csv")
detector = DriftDetector(baseline)

# Nova data (dados recentes de produção)
new_data = pd.read_csv("data/recent_predictions.csv")

# Detectar drift
report = detector.check_drift(new_data)

# report.drifts → List[FeatureDrift]
# report.should_retrain → bool (True se drift > threshold)
# report.critical_features → List[str] (features críticas com drift)

print(report.summary_line())
# [DriftReport 2026-04-08] 3/11 features com drift. 
# Retreinar: True. Críticas: ['engajamento_pct', 'nps_score']
```

#### **performance_monitor.py** — Métricas de Performance

```python
from analytics.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor(log_path="logs/perf.jsonl")

# Após uma rodada de predições
metric = monitor.record_batch(prediction_turns, session_id="sess_123")

# metric.account_count → 1000
# metric.avg_churn_risk → 0.45
# metric.high_risk_count → 124
# metric.total_mrr_at_risk → $5.2M

# Ver tendência nos últimos 30 dias
summary = monitor.trend_summary()
```

### decision_service/memory.py — Memória Semântica

```python
from decision_service.memory import AccountMemoryStore

# Conectar ao PostgreSQL com pgvector
store = AccountMemoryStore.from_env()
store.init_schema()

# Salvar predição no histórico
store.remember_prediction(
    account_id="ACC_000042",
    session_id="sess_abc",
    churn_risk=0.82,
    risk_level="HIGH",
    actions=["ESCALATE_TO_MANAGER", "SEND_EMAIL"]
)

# Recuperar contexto histórico
past = store.recall_account("ACC_000042", limit=5)
# past → List[MemoryMatch] com predições antigasas

for match in past:
    print(f"Similaridade: {match.distance:.2f}")
    print(f"Texto: {match.text}")
```

---

## Treinamento de Modelos

### Passo 1: Preparar Dados

```python
# data/generate_training_data.py
# Gera dados sintéticos com distribuição realista

python data/generate_training_data.py
# → data/train_dataset.csv (filas históricas de contas + labels)
```

**Features esperadas:**
```
engajamento_pct,nps_score,tickets_abertos,dias_no_contrato,...,renovado
35.0,4,2,180,...,0
85.5,9,0,365,...,1
...
```

### Passo 2: Treinar Modelos

Abra e execute `notebooks/02_training.ipynb`:

```python
# Carrega dados
df = pd.read_csv("data/train_dataset.csv")

# Prepara X, y
from decision_service.models import ChurnModel, ExpansionModel
from decision_service.training import ChurnTrainer, ExpansionTrainer

config_churn = ChurnModel()
trainer_churn = ChurnTrainer(config_churn)

X = df[config_churn.features]
y = df["renovado"]

model, report = trainer_churn.train(X, y, feature_names=X.columns)

# Salva artefatos
trainer_churn.save("ml/models/churn_model.pkl")
trainer_churn.save_scaler("ml/models/churn_scaler.pkl")

# Resultado:
# AUC-ROC: 0.85
# Avg Precision: 0.82
# Feature Importance: engajamento_pct (0.31), nps_score (0.25), ...
```

### Passo 3: Validar e Deployed

```python
# Testa em dados não-vistos
from decision_service.inference import ChurnPredictor

predictor = ChurnPredictor(
    model_path="ml/models/churn_model.pkl",
    scaler_path="ml/models/churn_scaler.pkl"
)

# Predição unitária
result = predictor.predict({
    "engajamento_pct": 35.0,
    "nps_score": 4,
    ...
})

print(f"Churn Risk: {result['churn_risk']:.2%}")

# Predição em lote
batch_results = predictor.batch_predict(df[config.features])
```

### Retreinamento Automático (Drift-Triggered)

Quando data drift é detectado:

```python
# 1. Monitor detecta drift
from analytics.drift_monitor import DriftDetector

baseline = pd.read_csv("data/train_dataset.csv")
detector = DriftDetector(baseline)

recent = pd.read_csv("data/recent_data.csv")
report = detector.check_drift(recent)

# 2. Se drift crítico, retreinar
if report.should_retrain:
    # Combina baseline + novo data
    updated_df = pd.concat([df_baseline, recent])
    
    # Treina novo modelo
    trainer = ChurnTrainer(config)
    model, report = trainer.train(X_updated, y_updated)
    
    # Deploy automático
    trainer.save("ml/models/churn_model.pkl")
    print("Novo modelo ativado!")
```

---

## Políticas de Intervenção

### Estrutura de uma Regra

```yaml
rules:
  - name: my_rule                    # identificador único
    priority: 3                       # 1=máxima prioridade
    stop: false                       # continua avaliando após match
    conditions:                       # condições AND
      churn_risk: { gt: 0.60 }        # > 60%
      mrr: { gte: 1000 }              # >= $1k
      dias_sem_interacao: { gt: 7 }   # > 7 dias
    actions:
      - code: SEND_EMAIL              # action code
        channels: [email, console]    # onde executar
      - code: CREATE_CRM_TASK
        channels: [console, file]
    reason: "Clientes desengajados com risco"
```

### Fluxo de Avaliação

1. **Ordenar regras** por `priority` (crescente)
2. **Avaliar cada regra**:
   - Todas as `conditions` devem ser verdadeiras (AND)
   - Se verdadeiro, adiciona `actions` à lista
   - Se `stop: true`, não avalia próximas
3. **Despachar todas as ações** coletadas

**Exemplo:**
```
Entrada: churn_risk=0.82, mrr=6000, dias_sem_interacao=14

Rule 1 (priority 1): churn_risk > 0.80 AND mrr > 5000
  ✓ Match! → Adiciona [ESCALATE_TO_MANAGER, SEND_SLACK_ALERT, SEND_EMAIL]
  stop=true → Não avalia mais regras

Saída: [ESCALATE_TO_MANAGER, SEND_SLACK_ALERT, SEND_EMAIL]
```

### Customizar Políticas

Edite `revenue_automation/policy/default_rules.yaml`:

```yaml
# Exemplo: Enviar email para todos com risco médio
- name: medium_risk_email
  priority: 10
  stop: false
  conditions:
    churn_risk: { gte: 0.40, lt: 0.70 }
  actions:
    - code: SEND_EMAIL
      channels: [email, console]
  reason: "Risco médio: reativar cliente"

# Exemplo: Escalação urgente (MRR muito alto)
- name: critical_mrr_300k
  priority: 0
  stop: true
  conditions:
    churn_risk: { gt: 0.50 }
    mrr: { gt: 300000 }
  actions:
    - code: ESCALATE_TO_C_LEVEL
      channels: [slack, email]
    - code: SCHEDULE_CALL
      channels: [console, api_hook]
  reason: "Conta enterprise crítica em risco"
```

Salve e a mudança é aplicada imediatamente (arquivo é relido a cada execução).

---

## Monitoramento e Analytics

### Dashboard de Métricas (via API)

```bash
# Último checkpoint perf
curl http://localhost:8001/api/performance

# Resultado:
{
  "metrics": [
    {
      "timestamp": "2026-04-08T15:30:00Z",
      "account_count": 1000,
      "avg_churn_risk": 0.45,
      "high_risk_count": 127,
      "medium_risk_count": 453,
      "low_risk_count": 420,
      "total_mrr_at_risk": 5200000
    }
  ],
  "trend": {
    "avg_risk_trend": "+2.3%",  # subiu 2.3%
    "high_risk_trend": "+12"    # +12 contas em risco alto
  }
}
```

### Derivar Insights do Audit Trail

```python
import pandas as pd
import json

# Carregar audit trail
entries = []
with open("logs/audit.jsonl") as f:
    for line in f:
        entries.append(json.loads(line))

df = pd.DataFrame(entries)

# Contas com mais previsões de alto risco
high_risk = df[
    (df.event_type == "prediction") & 
    (df.details.apply(lambda x: x.get("risk_level") == "HIGH"))
]
high_risk_accounts = high_risk.groupby("account_id").size().sort_values(ascending=False)
print(high_risk_accounts.head(10))

# MRR total em risco por período
df['date'] = pd.to_datetime(df['timestamp']).dt.date
mrr_by_date = df[df.event_type == "prediction"].groupby('date').apply(
    lambda g: g.details.apply(lambda x: x.get("mrr_at_risk", 0)).sum()
)
print(mrr_by_date)

# Taxa de sucesso de despacho
dispatched = df[df.event_type == "intervention"]
success_rate = dispatched.details.apply(lambda x: x.get("success_rate", 0)).mean()
print(f"Dispatch Success Rate: {success_rate:.1%}")
```

### Monitorar Drift Automaticamente

```python
# Setup: executar diariamente

from analytics.drift_monitor import DriftDetector
import pandas as pd

# Baseline (dados de treino originais)
baseline = pd.read_csv("data/train_dataset.csv")
detector = DriftDetector(baseline)

# Dados recentes
recent = pd.read_csv("data/recent_accounts.csv")

# Check
report = detector.check_drift(recent)

# Log resultado
import json
with open("logs/drift_reports.jsonl", "a") as f:
    f.write(json.dumps(report.as_dict(), ensure_ascii=False) + "\n")

# Alertar se crítico
if report.should_retrain:
    send_alert(
        f"DRIFT DETECTADO: {report.summary_line()}\n"
        f"Críticas: {report.critical_features}\n"
        f"Retreinar: TRUE"
    )
```

---

## Exemplos Práticos

### Exemplo 1: Prever Churn de Uma Conta

```python
from decision_service.query_engine import ChurnQueryEngine

engine = ChurnQueryEngine.from_env()

# Dados da conta (típicamente vêm de um CRM ou data warehouse)
account_features = {
    "engajamento_pct": 32.5,          # usuários ativos / total
    "nps_score": 3,                   # Net Promoter Score
    "tickets_abertos": 5,             # support tickets
    "dias_no_contrato": 120,          # há quanto tempo são clientes
    "engagement_trend": -10.0,         # % mudança últimos 30d
    "tickets_trend": 5.0,              # # mudança tickets
    "nps_trend": -2.0,                 # mudança NPS
    "dias_sem_interacao": 21,         # inativo há 21 dias
    "mrr": 3400,                      # revenue mensal
    "max_users": 25,                  # limite do plano
    "segment": "MID_MARKET",
}

# Analisar
turn = engine.analyze("ACC_001234", account_features)

# Interpretar
print(f"Account: {turn.account_id}")
print(f"Session: {turn.session_id}")
print(f"Timestamp: {turn.timestamp}")
print()
print(f"Risk Score: {turn.churn_risk:.1%} ({turn.risk_level})")
print(f"Retention Prob: {turn.retention_prob:.1%}")
print(f"MRR at Risk: ${turn.mrr_at_risk:,.0f}")
print()
print("Memory Insights (historias do passado):")
for match in turn.memory_hits:
    print(f"  - {match.text} (relevância: {1-match.distance:.0%})")
print()
print("Recomendações:")
for action in turn.recommended_actions:
    print(f"  - {action}")
```

### Exemplo 2: Executar Intervenção Automática

```python
from revenue_automation.engine import InterventionEngine

# Inicializa engine com políticas
engine = InterventionEngine.from_env(
    rules_path="revenue_automation/policy/default_rules.yaml",
    dry_run=False  # verdadeiro = loga sem enviar
)

# Intervir em conta
record = engine.intervene_one(
    account_id="ACC_001234",
    features=account_features
)

# Resultados
print(f"Correlation ID: {record.correlation_id}")
print(f"Churn Risk: {record.churn_risk:.1%}")
print(f"Risk Level: {record.risk_level}")
print()
print("Ações Executadas:")
for action in record.actions_taken:
    print(f"  ✓ {action}")
print()
print("Canais Usados:")
for channel in record.channels_used:
    print(f"  ✓ {channel}")
print()
print("Razões:")
for reason in record.reasons:
    print(f"  - {reason}")
```

### Exemplo 3: Batch Prediction (Múltiplas Contas)

```python
import pandas as pd
from decision_service.query_engine import ChurnQueryEngine
from revenue_automation.engine import InterventionEngine

# Carregar dados de múltiplas contas
df = pd.read_csv("data/accounts_to_analyze.csv")
# colunas: account_id, engajamento_pct, nps_score, mrr, ...

engine = ChurnQueryEngine.from_env()

# Processar
results = []
for idx, row in df.iterrows():
    turn = engine.analyze(
        account_id=row['account_id'],
        features=row.to_dict()
    )
    results.append({
        'account_id': turn.account_id,
        'churn_risk': turn.churn_risk,
        'risk_level': turn.risk_level,
        'mrr': row['mrr'],
        'mrr_at_risk': turn.mrr_at_risk,
        'retention_prob': turn.retention_prob,
    })

# Exportar
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('churn_risk', ascending=False)
df_results.to_csv("output/churn_predictions.csv", index=False)

print(df_results.describe())
# account_count: 1000
# churn_risk (mean): 0.45
# total_mrr_at_risk: $5.2M
```

### Exemplo 4: Customizar Channel de Email

```python
# revenue_automation/dispatch/channels/email.py
import smtplib
from email.mime.text import MIMEText

def send(action_code: str, payload: dict) -> bool:
    """Envia email customizado para ação específica"""
    
    if action_code == "SEND_EMAIL":
        to_email = payload.get("email")
        account_id = payload.get("account_id")
        
        subject = f"We've noticed your engagement is declining — let's talk!"
        body = f"""
Hello {account_id},

We noticed your usage has declined recently, and we'd love to understand why.
A member of our success team would like to reach out.

...
"""
        
        # Enviar via SMTP
        msg = MIMEText(body, 'html')
        msg['Subject'] = subject
        msg['From'] = 'success@company.com'
        msg['To'] = to_email
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login('user@company.com', 'password')
            server.send_message(msg)
        
        return True
    
    return False
```

### Exemplo 5: Análise Histórica com Audit Trail

```python
import json
import pandas as pd
from datetime import datetime, timedelta

# Carregar últimos 30 dias de auditoria
entries = []
cutoff = datetime.now() - timedelta(days=30)

with open("logs/audit.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        ts = datetime.fromisoformat(entry['timestamp'])
        if ts > cutoff:
            entries.append(entry)

df = pd.DataFrame(entries)

# Top 10 contas em risco
high_risk_preds = df[
    (df.event_type == "prediction") &
    (df.details.apply(lambda x: x.get("risk_level") == "HIGH"))
]
top_accounts = high_risk_preds.groupby("account_id").size().sort_values(ascending=False).head(10)

print("Top 10 contas em risco nos últimos 30 dias:")
print(top_accounts)

# Tendência de risk_level
risk_trend = df[df.event_type == "prediction"].groupby('event_type').apply(
    lambda g: g.details.apply(lambda x: x.get("risk_level")).value_counts()
)

print("\nDistribuição de Risk Levels:")
print(risk_trend / len(df[df.event_type == "prediction"]))

# MRR total sempre em risco
total_at_risk = df[df.event_type == "prediction"].details.apply(
    lambda x: x.get("mrr_at_risk", 0)
).sum()

print(f"\nTotal MRR em risco (últimos 30 dias): ${total_at_risk:,.0f}")
```

---

## Troubleshooting

### "Models not trained" (503 Service Unavailable)

**Problema:** API retorna erro 503, modelos não carregados.

**Solução:**
```bash
# 1. Verificar se arquivos existem
ls -la ml/models/
# Deve ter: churn_model.pkl, churn_scaler.pkl

# 2. Treinar modelos
python notebooks/02_training.ipynb

# 3. Restart API
uvicorn app:app --reload --port 8001
```

### "Database connection error"

**Problema:** `ERROR: could not connect to PostgreSQL`

**Solução:**
```bash
# Se usando Docker:
docker-compose logs postgres
# Verificar se está rodando:
docker ps | grep postgres

# Se usando postgres local:
psql -U postgres -c "SELECT 1"

# Se memory store é optional, desativar:
unset PGVECTOR_DSN
# API continuará funcionando sem memória semântica
```

### "Drift detected — model may be stale"

**Problema:** Data drift foi detectado no monitoramento.

**Solução:**
```bash
# 1. Visualizar relatório de drift
curl http://localhost:8001/api/drift-check?csv_path=data/recent_accounts.csv

# 2. Se drift > threshold, retreinar:
python notebooks/02_training.ipynb

# 3. Deploy novo modelo:
# (o sistema recarrega automaticamente ao próximo restart)
```

### "Action dispatch failed (Slack/Email)"

**Problema:** Ações não estão sendo despachadas em canais específicos.

**Solução:**
```bash
# 1. Verificar configuração de credenciais
echo $SLACK_WEBHOOK_URL  # Deve estar setado
echo $SMTP_PASSWORD       # Deve estar setado

# 2. Testar canal diretamente
from revenue_automation.dispatch.channels import slack
result = slack.send("SEND_SLACK_ALERT", {"text": "Test"})
print(f"Slack sent: {result}")

# 3. Checar logs da API
tail -f logs/dispatch.log

# 4. Se canal não essencial, usar dry_run:
engine = InterventionEngine.from_env(dry_run=True)
# Loga ações mas não envia de verdade
```

### "Memory queries are slow"

**Problema:** Recuperação de histórico (memory.recall_account) é lenta.

**Solução:**
```bash
# 1. Verificar se índices pgvector estão criados
psql -c "SELECT indexname FROM pg_indexes WHERE tablename='churn_memories';"

# 2. Se não tiver, recriar schema:
from decision_service.memory import AccountMemoryStore
store = AccountMemoryStore.from_env()
store.init_schema()

# 3. Aumentar memory_limit para menos recuperações
engine = ChurnQueryEngine(
    query_engine=qe,
    config=ChurnEngineConfig(memory_limit=3)  # padrão = 5
)

# 4. Usar embedding provider "hash" (mais rápido) ao invés de "openai"
export EMBEDDING_PROVIDER="hash"
```

### "Feature encoding mismatch"

**Problema:** `ValueError: unknown segment value`

**Solução:**
```python
# O sistema suporta: SMB, MID_MARKET, ENTERPRISE
# Se receber outro valor:

# 1. Mapear customizado
SEGMENT_MAP = {
    "small": "SMB",
    "mid": "MID_MARKET",
    "large": "ENTERPRISE",
    "startup": "SMB",
}

features["segment"] = SEGMENT_MAP.get(features.get("segment"), "MID_MARKET")

# 2. Ou treinar novo encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['segment'].unique())
# Salvar: joblib.dump(encoder, "ml/models/segment_encoder.pkl")
```

### "Notebook execution fails"

**Problema:** 02_training.ipynb falha no meio.

**Solução:**
```bash
# 1. Verificar arquivo de dados
head -5 data/train_dataset.csv

# 2. Se vazio, gerar dados:
python data/generate_training_data.py

# 3. Se falta dependência:
pip install -r requirements.txt

# 4. Executar cell por cell para isolar erro:
# (abra o notebook no VS Code ou Jupyter e teste manual)
```

---

## FAQ

**P: Posso usar o sistema sem PostgreSQL?**
R: Sim! A memória semântica é opcional. Sem `PGVECTOR_DSN`, o sistema funciona normalmente, mas não recuperará histórico automático de contas.

**P: Como faço para retreinar os modelos?**
R: Execute `notebooks/02_training.ipynb`. Você pode deixar rodar diariamente ou manualmente, ou ativar retrain automático quando drift é detectado (config `RETRAIN_ON_DRIFT=true`).

**P: Posso customizar as features do modelo?**
R: Sim! Edite `decision_service/models.py`:
```python
@dataclass
class ChurnModel:
    features = [
        "sua_feature_1",
        "sua_feature_2",
        ...
    ]
```
Depois retreine com os mesmos nomes de coluna no CSV.

**P: Como exportar relatórios?**
R: Via API (`/api/audit`, `/api/performance`) ou via Python:
```python
from analytics.audit_trail import ChurnAuditTrail
trail = ChurnAuditTrail(log_path="logs/audit.jsonl")
csv = trail.to_csv()
with open("export.csv", "w") as f:
    f.write(csv)
```

**P: Posso rodar em produção?**
R: Sim! Use Docker + Docker Compose (já configurado). Para escala, considere:
- Load balancer na frente da API
- PostgreSQL gerenciado (AWS RDS, Azure Database)
- Ollama em servidor separado para análise
- Workers assíncronos para batch processing

**P: Como faço debug de uma predição?**
R: Use `verbose_mode`:
```python
from decision_service.query_engine import ChurnQueryEngine
engine = ChurnQueryEngine.from_env()
turn = engine.analyze("ACC_001234", features)
print(turn)  # Todos os detalhes
```

---

## Roadmap Futuro

- [ ] Dashboard web (React + D3.js)
- [ ] Explicabilidade (SHAP values para cada predição)
- [ ] Multi-tenancy (múltiplos workspaces)
- [ ] Fine-tuning de LLM local (Ollama integration avançada)
- [ ] Predição de upsell automática
- [ ] A/B testing de políticas
- [ ] Integração com CRM (Salesforce, HubSpot)
- [ ] Alertas SMS via Twilio

---

## Contacto e Suporte

- **Issues:** GitHub Issues
- **Email:** engineering@company.com
- **Slack:** #churn-predictor

---

## License

MIT — Livre para usar, modificar e distribuir.
