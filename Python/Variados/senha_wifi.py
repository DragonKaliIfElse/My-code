
def combinacoes(inicio_d = 01, fim_d = 30, inicio_m = 01, fim_m = 12):
  combinacoes = []
  for primeiro in range(inicio_d, fim_d + 1):
    for segundo in range(inicio_m, fim_m + 1):
      
     
    	combinacoes.append(str('pizza')+ str(primeiro) + str(segundo))
  return combinacoes
combinacoes = combinacoes()
print(combinacoes)
