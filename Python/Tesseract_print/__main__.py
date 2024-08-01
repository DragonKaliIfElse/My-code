import os
import pytesseract
from PIL import Image

def main():
	file = Image.open('print_para_ler.png')
	text = pytesseract.image_to_string(file, lang='eng')
	os.remove('print_para_ler.png')
	with open('texto_do_print','w') as file:
		file.write(text)
if __name__ == '__main__': main();
