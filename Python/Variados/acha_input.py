import pyautogui 
import pytesseract
import time

def find_word():

	region = (465, 349, 192, 24)
	image = pyautogui.screenshot(region = region)
	text = pytesseract.image_to_string(image).strip().lower()
	return text

text = None

while not text == 'email or phone':

	text = find_word()
	time.sleep(0.2)
	
	if text == 'email or phone':
		print('achei {}'.format(text))
		
	elif text != 'email or phone': 
		print(text)	
	
		

