from googletrans import Translator
import os

class Tradutor():
	def __init__(self):
		self.dicionário = "/home/dragon/Python/Tradutor/Dicionário"

	def word_list(self):
		with open(self.dicionário, 'r') as file:
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
		check = "check"
		os.remove(check)
		
		return words

	def remove_line(self):
		with open(self.dicionário, 'r') as file:
			dic = file.readlines()
		dic = dic[:-1]
		with open(self.dicionário, 'w') as file:	
			file.writelines(dic)
		return None
			
	def edit_word(self):
		word_chosed = input("Qual palavra deseja alterar?\n")
		with open(self.dicionário,'r') as file:
			dic = file.readlines()
		lines = []
		for line in dic:
			lines.append(line)
		lines = ' '.join(lines)
		
		with open('file', 'w') as file:
			count = 0 
			for char in lines:
				if (char == '-' and count == 0): char = '\n'; count=1;
				elif (char == '-' and count ==1): char = '';
				elif char != '-': count = 0;
				file.write(char)
				
		with open('file', 'r') as file:
			old_words = file.readlines()
		wordss = []
		for wrd in old_words:
			wrd_s = []
			count = 0
			for char in wrd:
				if count == 0 and char == ' ': char = '';
				wrd_s.append(char)
				count+=1
			palavra = ''.join(wrd_s)
			palavra = palavra.replace(' \n','')
			palavra = palavra.replace('\n','')
			wordss.append(palavra) 
		os.remove('file')
		
		existe = False
		indice = 0
		for i, line in enumerate(wordss):
			if line == word_chosed:	
				existe = True
				indice = i+1
		if existe == False: print('Palavra não encontrada\n'); 
		elif existe == True: 
			traducao = input("Qual a nova tradução?\n")
			words = []
			for i,linha in enumerate(wordss):
				if i == indice: linha = traducao;
				words.append(linha)
				
			with open(self.dicionário, 'w') as file:
				i1 = 0
				i2 = 1
				while i1 < (len(words)) and i2 <= (len(words)):
					word1 = words[i1]
					word2 = words[i2] 
					word1 = word1.replace('\n','')
					word2 = word2.replace('\n','')
					file.write(f'{word1} ---- {word2}\n')
					i1+=2
					i2+=2
		return None
		
	def add_word(self,text,translation,words):
		with open(self.dicionário,'a') as file:
			verify = 0
			text = text.replace('+ ','')
			text = text.replace('+','')
			for word in words:
				if text == word:	
					verify = 1
			if verify == 0:
				traducao = translation
				traducao = traducao.replace('+ ','')
				traducao = traducao.replace('+','')
				file.write(f'{text} ---- {traducao}\n')
				print("PALAVRA ADICIONADA AO DICIONÁRIO\n")
		return None
	
	def search_word(self,word_chosed, mode):
		word_chosed = word_chosed.replace('+ ','')
		word_chosed = word_chosed.replace('+','')
		with open(self.dicionário,'r') as file:
			dic = file.readlines()
		lines = []
		for line in dic:
			lines.append(line)
		lines = ' '.join(lines)
		
		with open('file', 'w') as file:
			count = 0 
			for char in lines:
				if (char == '-' and count == 0): char = '\n'; count=1;
				elif (char == '-' and count ==1): char = '';
				elif char != '-': count = 0;
				file.write(char)
				
		with open('file', 'r') as file:
			old_words = file.readlines()
		words = []
		for wrd in old_words:
			wrd_s = []
			count = 0
			for char in wrd:
				if count == 0 and char == ' ': char = '';
				wrd_s.append(char)
				count+=1
			palavra = ''.join(wrd_s)
			palavra = palavra.replace(' \n','')
			palavra = palavra.replace('\n','')
			words.append(palavra) 
		os.remove('file')
		
		existe = False
		indice = 0
		for i, line in enumerate(words):
			if line == word_chosed:	
				existe = True
				indice = i+1
		if existe == True: 	
			print(f'{words[indice]}\n')
			return True
		elif existe == False and mode == 'offline': print('Palavra não encontrada\n');
		
		return False
		
		
Tradutor = Tradutor()
translator = Translator()

def main():		
	text=''
	mode = 'online'
	while True:
		text = input()
		verify = None
		if text == 'off()': break;
		elif text == 'remove()': Tradutor.remove_line();print('ÚLTIMA PALAVRA REMOVIDA\n'); continue;
		elif text == 'edit()': Tradutor.edit_word();print('TRADUÇÃO EDITADA COM SUCESSO\n') ;continue;
		elif text == 'offline()': mode = 'offline'; print('MODO OFFLINE\n'); continue; 
		elif text == 'online()': mode = 'online'; print('MODO ONLINE\n'); continue;
		
		print("traduzindo...\n")
		
		if mode == 'online': 
			verify = Tradutor.search_word(text,mode)
			if verify == True: continue;
			
		if mode == 'online' and verify == False:
			translation = translator.translate(text, dest="pt")
			print(f'{translation.text}\n')
			words = Tradutor.word_list()
		if text[0] == ('+') and verify == False:
			Tradutor.add_word(text,translation.text,words)
			
		if mode == 'offline':
			Tradutor.search_word(text,mode)
			
		
	return None
			
if __name__ == "__main__":
	main()
