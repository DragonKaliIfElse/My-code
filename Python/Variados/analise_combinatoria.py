#ANÁLISE COMBINATÓRIA-----------------------------------------------------------------------------------------------------------------------------------------------
def combinacoes(inicio = 0, fim = 9, meu_arquivo='combinacoes.txt'):
	with open(meu_arquivo, 'w') as file:
		combinacoes = []
		for primeiro in range(inicio, fim + 1):
			for segundo in range(inicio, fim + 1):
				for terceiro in range(inicio, fim + 1):
					for quarto in range(inicio, fim + 1):
						for quinto in range(inicio, fim + 1):
							combinacoes.append(str(primeiro) + str(segundo) + str(terceiro) + str(quarto) + str(quinto))
							file.write(f'{primeiro}{segundo}{terceiro}{quarto}{quinto}\n')

	return combinacoes
combinacoes = combinacoes()
print(len(combinacoes))
