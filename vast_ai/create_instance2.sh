#!/bin/bash
set -euo pipefail

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================
MODEL_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
MODEL_FILE="Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"
MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
LLAMA_REPO="https://github.com/ggerganov/llama.cpp"
WORKSPACE="/workspace"
MODEL_DIR="${WORKSPACE}/models"
LLAMA_DIR="${WORKSPACE}/llama.cpp"

# Parâmetros otimizados para Tesla V100 32GB
GPU_LAYERS=35          # Camadas na GPU (reduzir se houver OOM)
CTX_SIZE=2048          # Tamanho do contexto (economiza VRAM)
THREADS=8              # Threads da CPU
PARALLEL=1             # Requisições paralelas

# Token HuggingFace (opcional, para modelos gated)
HF_TOKEN="${HF_TOKEN:-}"

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

error_exit() {
    log "❌ ERRO: $*"
    exit 1
}

# =============================================================================
# 1. AUTENTICAÇÃO E BUSCA DE OFERTA
# =============================================================================
log "🔐 Carregando credenciais..."
API_KEY="${VAST_API_KEY:-$(cat ~/.vast_api_key 2>/dev/null || echo '')}"
[[ -z "$API_KEY" ]] && error_exit "API key não encontrada. Defina VAST_API_KEY ou ~/.vast_api_key"

log "🔍 Buscando Tesla V100 com ~32GB VRAM..."
OFFER_ID=$(vastai search offers 'gpu_ram>=32 gpu_ram<40 gpu_name=Tesla_V100' --order dph --limit 1 --raw | grep ask_contract_id | awk '{print$2}' | sed 's|,||')
[[ -z "$OFFER_ID" ]] && error_exit "Nenhuma oferta Tesla V100 disponível encontrada"

log "✅ Oferta selecionada: ID=$OFFER_ID"

# =============================================================================
# 2. CRIAÇÃO DA INSTÂNCIA
# =============================================================================
log "🚀 Criando instância..."

# Script de inicialização automatizado (onstart-cmd)
ONSTART_CMD=$(cat <<'ONSTART_EOF'
#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }
error_exit() { log "❌ $*"; exit 1; }

# 1. Instalar dependências
log "📦 Instalando dependências..."
apt-get update -qq && apt-get install -y -qq \
    git build-essential cmake curl wget libcurl4-openssl-dev \
    > /dev/null 2>&1 || error_exit "Falha ao instalar dependências"

# 2. Clonar e compilar llama.cpp com CUDA
log "🔨 Compilando llama.cpp com suporte CUDA..."
if [[ ! -d "/workspace/llama.cpp/build/bin/llama-server" ]]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp /workspace/llama.cpp 2>/dev/null || true
    cd /workspace/llama.cpp
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    cmake --build build --config Release -j$(nproc) --target llama-server > /dev/null 2>&1 || \
        error_exit "Falha na compilação do llama.cpp"
fi

# 3. Baixar o modelo GGUF
log "📥 Baixando modelo ${MODEL_FILE}..."
mkdir -p /workspace/models
MODEL_PATH="/workspace/models/${MODEL_FILE}"

if [[ ! -f "$MODEL_PATH" ]] || [[ $(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0) -lt 1000000000 ]]; then
    wget_cmd="wget --continue --progress=bar:force"
    [[ -n "${HF_TOKEN:-}" ]] && wget_cmd+=" --header='Authorization: Bearer ${HF_TOKEN}'"
    
    $wget_cmd -O "$MODEL_PATH.tmp" "${MODEL_URL}" 2>/dev/null || \
        error_exit "Falha no download do modelo"
    
    mv "$MODEL_PATH.tmp" "$MODEL_PATH"
    log "✅ Modelo baixado: $(ls -lh "$MODEL_PATH" | awk '{print $5}')"
else
    log "✅ Modelo já existe, pulando download"
fi

# 4. Iniciar llama-server com parâmetros otimizados para V100 32GB
log "🔥 Iniciando llama-server..."
exec /workspace/llama.cpp/build/bin/llama-server \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --ctx-size ${CTX_SIZE} \
    --n-gpu-layers ${GPU_LAYERS} \
    --threads ${THREADS} \
    --parallel ${PARALLEL} \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --verbose 2>&1 | tee /workspace/llama-server.log
ONSTART_EOF
)

# Criar instância
INSTANCE_RESULT=$(vastai create instance "$OFFER_ID" \
    --image ubuntu:22.04 \
    --disk 60 \
    --ssh --direct \
    --env "-p 8000:8000 -e MODEL_FILE='${MODEL_FILE}' -e MODEL_URL='${MODEL_URL}' -e HF_TOKEN='${HF_TOKEN}'" \
    --onstart-cmd "$ONSTART_CMD" \
    --raw 2>/dev/null)

INSTANCE_ID=$(echo "$INSTANCE_RESULT" | jq -r '.new_contract // empty' 2>/dev/null)
[[ -z "$INSTANCE_ID" ]] && error_exit "Falha ao criar instância: $INSTANCE_RESULT"

log "✅ Instância criada: ID=$INSTANCE_ID"

# =============================================================================
# 3. AGUARDAR INICIALIZAÇÃO E TESTAR API
# =============================================================================
log "⏳ Aguardando inicialização (pode levar 15-45 min para download + compilação)..."

MAX_WAIT=3600  # 1 hora máxima
WAITED=0
INTERVAL=30

while [[ $WAITED -lt $MAX_WAIT ]]; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | jq -r '.actual_status // "unknown"')
    
    case "$STATUS" in
        running)
            # Tentar conectar à API
            IP_PORT=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | \
                jq -r '.ports["8000/tcp"][0] // empty | "127.0.0.1:\(.HostPort)"' 2>/dev/null)
            
            if [[ -n "$IP_PORT" ]] && curl -sf "http://$IP_PORT/health" >/dev/null 2>&1; then
                log "🎉 API pronta em http://$IP_PORT"
                
                # Teste rápido
                log "🧪 Executando teste de API..."
                curl -sf "http://$IP_PORT/v1/models" | jq -c '.data[0] // {error: "sem modelos"}' || true
                
                echo ""
                echo "============================================"
                echo "✅ INSTÂNCIA PRONTA PARA USO!"
                echo "============================================"
                echo "API Endpoint: http://$IP_PORT/v1"
                echo "Modelo: ${MODEL_FILE}"
                echo ""
                echo "Exemplo de uso com curl:"
                echo "  curl http://$IP_PORT/v1/chat/completions \\"
                echo "    -H 'Content-Type: application/json' \\"
                echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Olá!\"}],\"max_tokens\":100}'"
                echo ""
                echo "Para parar a instância depois:"
                echo "  vastai stop instance $INSTANCE_ID"
                echo "Para destruir (para de cobrar tudo):"
                echo "  vastai destroy instance $INSTANCE_ID"
                echo "============================================"
                
                exit 0
            fi
            ;;
        exited|stopped|offline)
            error_exit "Instância parou inesperadamente. Verifique logs: vastai logs $INSTANCE_ID --tail 50"
            ;;
    esac
    
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
    log "⏳ Aguardando... (${WAITED}s/${MAX_WAIT}s) - Status: $STATUS"
done

error_exit "Tempo limite excedido. Verifique: vastai logs $INSTANCE_ID --tail 100"
