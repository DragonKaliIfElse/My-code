#!/bin/python3
import sys

def main():
    arquivo = sys.argv[1]
    with open(arquivo, 'r') as file:
        script = file.read()
    with open(arquivo, 'w') as file:
        script = script.replace("    ","    ")
        file.write(script)

if __name__=='__main__':main();
