import os
def main():
	texto = []
	with open('cabecalho', 'r') as file:
		while True:
			raw = file.readline() 
			if not raw:
				break
			texto.append(raw)
			
	with open('cabecalho2', 'w') as file:
		for linha in texto:
			linha = linha.replace('\n','')
			file.write(f'"{linha}",\n')
			
	texto = []		
	with open('cabecalho2', 'r') as file:
		while True:
			raw = file.readline() 
			texto.append(raw)
			if not raw:
				break
	return texto			
if __name__=='__main__': main();
