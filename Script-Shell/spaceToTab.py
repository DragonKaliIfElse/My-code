#!/bin/python3
import sys

def main():
	arquivo = sys.argv[1]
	with open(arquivo, 'r') as file:
		editavel = file.read()
	with open(arquivo, 'w') as file:
		editavel = editavel.replace('    ','	')
		file.write(editavel)

if __name__=="__main__": main();
