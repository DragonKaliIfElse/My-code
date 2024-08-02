import numpy as np
array = [[0]*3 for _ in range(10)]
array = np.asarray(array)
print(f'{array}\n{type(array)}')

array = np.zeros((10,3))
print(f'{array}\n{type(array)}')
