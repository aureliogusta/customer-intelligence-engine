#!/usr/bin/env bash
# .ollama_init.sh
# ================
# Instala Ollama e baixa o modelo LLM para o CS Analysis Agent.
#
# Uso:
#   chmod +x .ollama_init.sh
#   ./.ollama_init.sh

set -e

echo "================================================"
echo "  CS Churn Predictor — Ollama Setup"
echo "================================================"

# 1. Verificar se Ollama já está instalado
if command -v ollama &> /dev/null; then
    echo "[OK] Ollama já instalado: $(ollama --version)"
else
    echo "[INFO] Instalando Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "[AVISO] Windows detectado."
        echo "  Baixe manualmente: https://ollama.ai/download"
        echo "  Depois execute: ollama pull mistral:7b"
        exit 0
    fi
fi

# 2. Iniciar servidor Ollama em background (se não estiver rodando)
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "[INFO] Iniciando servidor Ollama em background..."
    ollama serve &
    sleep 3
fi

# 3. Baixar modelo
PRIMARY_MODEL="mistral:7b"
FALLBACK_MODEL="llama3.2:3b"   # mais leve, para máquinas com menos RAM

echo ""
echo "Escolha o modelo LLM:"
echo "  1) mistral:7b    — melhor qualidade (4GB RAM)"
echo "  2) llama3.2:3b   — mais leve (2GB RAM)"
echo ""
read -rp "Opção [1]: " choice
choice=${choice:-1}

if [[ "$choice" == "2" ]]; then
    MODEL=$FALLBACK_MODEL
else
    MODEL=$PRIMARY_MODEL
fi

echo ""
echo "[INFO] Baixando $MODEL ..."
ollama pull "$MODEL"

echo ""
echo "================================================"
echo "  Configuração concluída!"
echo "================================================"
echo "  Modelo     : $MODEL"
echo "  Servidor   : http://localhost:11434"
echo ""
echo "  Para testar:"
echo "    ollama run $MODEL 'Olá, como você pode me ajudar?'"
echo ""
echo "  Para rodar o agente CS:"
echo "    python study_agent.py --demo"
echo "================================================"
