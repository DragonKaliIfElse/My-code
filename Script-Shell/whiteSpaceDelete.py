#!/bin/python3
import sys

def main():
	arquivo = sys.argv[1]
	with open(arquivo, 'r') as file:
		script=[]
		line = file.readlines()
		if line:
			script.append(line)
	with open(arquivo, 'w') as file:
		onlyWhiteSpace=True
		for lines in script:
			for line in lines:
				if any(char.isalpha() for char in line):
					pass
				elif not any(char.isalpha() for char in line):
					for index,_ in enumerate(line):
						char = repr(line[index])
						charMenosUm = repr(line[index-1])
						if char == repr('\n') and (charMenosUm == repr(' ') or charMenosUm == repr('	') or charMenosUm == repr('\t')):
							line = '\n'
				file.writelines(line)
	return None
if __name__=='__main__':main();

