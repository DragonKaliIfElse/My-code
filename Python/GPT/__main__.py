import openai

# Configure sua chave de API
openai.api_key = 'sua-chave-de-api'

# Fazer uma chamada para o GPT-4
response = openai.Completion.create(
    model="text-davinci-004",  # Supondo que "davinci-004" seja uma versão do GPT-4
    prompt="Escreva um poema sobre a lua.",
    max_tokens=50
)

print(response.choices[0].text.strip())
