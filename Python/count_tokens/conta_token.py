import tiktoken
import sys

def contar_tokens(texto: str, modelo: str = "gpt-4o-mini") -> int:
    if not texto:
        return 0

    encoding = tiktoken.encoding_for_model(modelo)
    tokens = encoding.encode(texto)
    return len(tokens)

def extrair_linhas_com_prefixo(texto: str, prefixos) -> str:
    """
    Retorna uma string única com todas as linhas que começam com
    qualquer um dos prefixos informados.
    
    prefixos: pode ser string ou lista/tupla de strings
    """
    if isinstance(prefixos, str):
        prefixos = (prefixos,)
    else:
        prefixos = tuple(prefixos)

    linhas_filtradas = []
    
    for linha in texto.splitlines():
        if linha.startswith(prefixos):
            linhas_filtradas.append(linha)

    return "".join(linhas_filtradas)

def remover_linhas_com_prefixo(texto: str, prefixos) -> str:
    """
    Retorna uma string com todas as linhas que NÃO começam com
    os prefixos informados.
    """
    if isinstance(prefixos, str):
        prefixos = (prefixos,)
    else:
        prefixos = tuple(prefixos)

    linhas_restantes = []
    
    for linha in texto.splitlines():
        if not linha.startswith(prefixos):
            linhas_restantes.append(linha)

    return "".join(linhas_restantes)

def main():
    arquivo = sys.argv[1]
    file = open(arquivo, 'r')
    texto = file.read()
    file.close()
    
    text_sem_prefixo = remover_linhas_com_prefixo(texto, '#meu_prompt')
    text_com_prefixo = extrair_linhas_com_prefixo(texto, '#meu_prompt')

    print(f'input: {contar_tokens(text_com_prefixo)}')
    print(f'output: {contar_tokens(text_sem_prefixo)}')
if __name__ == '__main__': main();
