import xlsxScan as xs
import re

def acha_lc(df, expressao):
    padrao = f".*{re.escape(expressao)}.*"  # evita erro com caracteres especiais

    posicoes = []

    # percorre todas as células do DataFrame
    for linha in range(len(df)):
        for coluna in range(len(df.columns)):
            valor = str(df.iat[linha, coluna])  # pega valor da célula como string
            if re.search(padrao, valor):
                posicoes.append((linha, coluna))

    if posicoes:
        return posicoes
    else:
        print("Expressão não encontrada")
        return []
    # 'row' é uma tupla de objetos Cell
        print(cell.value, end="\t") # Imprimir o valor da célula
    print() # Pular para a próxima linha
arquivo = "ORÇAMENTO DE EVENTOS-backup.xlsx"
arquivo_editavel = xs.criar_copia_editavel(arquivo)
df, wb, ws = xs.carregar_dados_com_formato(arquivo_editavel, "LANÇAMENTOS")

evento = input("qual o nome do evento?\n")
evento = evento.strip()
coluna_a = ws['C']

print(f"Conteúdo da coluna 'C':")

# Iterar sobre cada célula da coluna para imprimir o valor
celulas=[]
for i, celula in enumerate(coluna_a):
    if celula.value==None: break;
    if re.search(evento, celula.value):   
        print(f'{celula.value} na linha {i+1}')
        for i, celula in enumerate(ws[i+1]):
            if i > 12: break;
            print(celula.value, end='\t') # O \t adiciona um espaço entre os valores
        print()
resposta = input('Editar celulas?')
if resposta == 's':
    while True:
        linha = input('Qual linha quer editar?')
        linha = int(linha)
        new = input('Quais os novos valores?')
        new = new.split(',')
        for i, cell in enumerate(new):
            xs.editar_celula(arquivo,"LANÇAMENTOS", linha=linha, coluna=i+1, valor=cell)
        continuar = input('Editar de novo?')
        if continuar == 's':
            continue
        else:
            break
else:
    print('ok')

