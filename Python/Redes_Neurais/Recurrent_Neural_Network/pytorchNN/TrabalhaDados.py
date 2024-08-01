import numpy as np
import pandas as pd

train=[]
with open("train.csv", "r") as file:
	linhas = file.readlines()
	count=0
	for linha in linhas:
		if count==0:
			count+=1
			continue
		else:
			train.append(count)
			count+=1
		"""
		linha = linha[:8]
#		linha = linha.replace('20','')
		linha = linha.replace('jan','.01')
		linha = linha.replace('fev','.02')
		linha = linha.replace('mar','.03')
		linha = linha.replace('abr','.04')
		linha = linha.replace('mai','.05')
		linha = linha.replace('jun','.06')
		linha = linha.replace('jul','.07')
		linha = linha.replace('ago','.08')
		linha = linha.replace('set','.09')
		linha = linha.replace('out','.10')
		linha = linha.replace('nov','.11')
		linha = linha.replace('dez','.12')
		linha = linha.replace(' ','')
		train.append(linha)
		"""
with open("train_plot", "w") as file:
	for i in range(len(train)):
		file.write(f'{train[i]}\n')
