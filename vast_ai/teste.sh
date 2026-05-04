curl http://108.179.129.245:57239/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "Olá! Explique em 2 frases o que é um MoE."}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
