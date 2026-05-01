OPEN_BUTTON_TOKEN=$(cat /home/4n4tm4n/.OPEN_BUTTON_TOKEN)
ID=$(vastai search offers 'gpu_ram>=32 gpu_ram<64 gpu_name=RTX_A6000' --order dph --limit 1 --raw | grep ask_contract_id | awk '{print$2}' | sed 's|,||')
vastai create instance "$ID" \
  --image vastai/vllm:v0.19.0-cuda-12.9 \
  --env '-p 1111:1111 -p 7860:7860 -p 8080:8080 -p 8000:8000 -p 8265:8265 \
         -e OPEN_BUTTON_TOKEN="$OPEN_BUTTON_TOKEN"
         -e OPEN_BUTTON_PORT="1111" \
         -e JUPYTER_DIR="/" \
         -e DATA_DIRECTORY="/workspace/" \
         -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:7860:17860:/:Model UI|localhost:8000:18000:/docs:vLLM API|localhost:8265:28265:/:Ray Dashboard|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal" \
         -e VLLM_MODEL="Qwen/Qwen3.6-35B-A3B-FP8" \
         -e VLLM_ARGS="--max-model-len 8192 --download-dir /workspace/models --host 127.0.0.1 --port 18000 --enable-auto-tool-choice --tool-call-parser hermes" \
         -e AUTO_PARALLEL="true" \
         -e RAY_ADDRESS="127.0.0.1" \
         -e RAY_ARGS="--head --port 6379 --dashboard-host 127.0.0.1 --dashboard-port 28265"' \
  --onstart-cmd 'entrypoint.sh' \
  --disk 50 \
  --jupyter --ssh --direct
