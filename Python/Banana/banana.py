import pyautogui
import time

x,y = pyautogui.position()
print(f"{x},{y}")
while True:
	# Clique no botão inicialmente
	pyautogui.click(x=x, y=y)

	# Espere 10 segundos
	time.sleep(5)

	# Clique novamente no mesmo botão
	pyautogui.click(x=x, y=y)
