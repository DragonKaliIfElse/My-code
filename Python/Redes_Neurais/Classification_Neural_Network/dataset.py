import sklearn.datasets as datasets
import numpy as np
#import matplotlib.pyplot as plt

def xy():
	# Gerar os dados das duas luas
	x, y = datasets.make_moons(n_samples=500, noise=0.05, random_state=42)

	return x, y

