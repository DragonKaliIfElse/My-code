from googletrans import Translator
import os

def main():
	translator = Translator()
	with open("Dicionário", 'r') as file:
		lines=[]
		for line in file:
			for char in line:
				if char == ' ':
					char='\n'
				lines.append(char)
	with open("check",'w') as file:
		for char in lines:
			file.write(char)
	words=[]
	with open("check",'r') as file:
		for word in file:
			word = word.strip()
			words.append(word)
			
	for i in range(len(words)):
		for i2 in range(len(words)):
			if words[i] == words[i2] and i != i2 and (words[i] != '-' and words[i] != '----'):
				contador = 0
				for char in words[i]:
					if 'a'<=char<='z' and contador==0: 
						idioma = translator.detect(words[i].strip())
						if idioma.lang == "en":
							print(f'{words[i]}')
						contador +=1
							
	Check = "/home/dragon/Documents/Anotações/Check/check"
	os.remove(Check)
	print("***CHECAGEM CONCLUÍDA***")

if __name__ == "__main__":
	main()
