#!/bin/python3
import sys
from nltk.corpus import wordnet as wn
def main():
    word = sys.argv[1]
    # Palavra em inglês
    english_word = word

    # Buscar o synset (conjunto de sinônimos) da palavra
    synsets = wn.synsets(english_word, lang='eng')

    # Listar possíveis traduções para o português
    for synset in synsets:
        # Verificar as palavras correspondentes em português
        translations = synset.lemma_names('por')  # 'por' para português
        if translations == []:
            continue
        print(translations)
        return 0
if __name__ == '__main__': main();
