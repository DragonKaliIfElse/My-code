from pathlib import Path

def get_api_key():
    token_file = Path.home() / ".openrouter_token"
    try:
        return token_file.read_text().strip()
    except:
        raise FileNotFoundError(f"Chave não encontrada em {token_file}")

def main():
    return 0
if __name__ == '__main__':
    main()
