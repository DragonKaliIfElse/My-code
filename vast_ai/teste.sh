OPEN_BUTTON_TOKEN=$(cat ~/.OPEN_BUTTON_TOKEN)
IP=""
PORTA=""
curl -X POST "http://$IP:$PORTA/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPEN_BUTTON_TOKEN" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "Explique computação quântica em 2 frases."}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
