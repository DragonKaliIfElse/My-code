import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size):
		super(LSTMModel, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

		out, _ = self.lstm(x, (h0, c0))
		out = self.fc(out[:, -1, :])
#		print(f'out\n{out}')
		return out

def createSequences(data, seq_length):
    x=[]
    y=[]
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.stack(x), torch.stack(y)

def main():

	batch_size = 12
	sequence_Length = 3
	input_size = 1
	hidden_size = 50
	num_layers = 2
	output_size = 1
	num_epochs = 200
	learning_rate = 0.001

	x_train = pd.read_csv("train.csv")
	x_test = pd.read_csv("test.csv")
	x_trainNumpy = x_train.to_numpy()
	x_testNumpy = x_test.to_numpy()

	x_train = np.vstack(x_trainNumpy[:,1]).astype(np.cfloat)
	x_test = np.vstack(x_testNumpy[:,1]).astype(np.cfloat)

	x_train = torch.tensor(x_train, dtype=torch.float32)
	x_test = torch.tensor(x_test, dtype=torch.float32)

	x_train, y_train = createSequences(x_train, sequence_Length)
#	x_train = x_train.unsqueeze(-1)
#	print(f'x_train\n{x_train.shape}')
	train_plot=[]
	with open("train_plot",'r') as file:
		linhas = file.readlines()
		for linha in linhas:
			linha = linha.replace('\n','')
			linha = float(linha)
			train_plot.append(linha)

	train_dataset = TensorDataset(x_train, y_train)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
	criterion = nn.MSELoss()
	evaluation = nn.L1Loss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	find = False
	arrayOutput=[]
	for epoch in range(num_epochs):
		if find == True:
			break
		for i, (inputs, targets) in enumerate(train_loader):
			inputs, targets = inputs.to(device), targets.to(device)

			outputs = model(inputs)
			out = model(x_train)
			loss = criterion(outputs, targets)
			MAE = evaluation(outputs, targets)
			MAEGeral = evaluation(out, y_train)
#			if epoch+1 == num_epochs:
#				for output in outputs:
#					arrayOutput.append(output.item())
			optimizer.zero_grad()
#			if loss.item()<1:
#				MAE.backward()
#			else:
			loss.backward()
			optimizer.step()
			if epoch+1 == num_epochs and i+1 == len(train_loader):
				print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss_Batch: {MAE.item():.4f}, Loss_Geral: {MAEGeral.item():.4f} (usando MAE)')
#				print("Printando gráfico")
#				find = True
#				train_plot = np.asarray(train_plot)
#				arrayOutput = np.asarray(arrayOutput)
#				train_plot = train_plot.reshape(-1,1)
#				plt.plot(train_plot[sequence_Length:], arrayOutput, color='blue')
#				plt3.plot(train_plot[sequence_Length:], x_trainNumpy[sequence_Length:,1], color='red')
#				plt.show()
				"""
				colum1 = train_plot[:,0].flatten()
				real = x_trainNumpy[:,0].flatten()
				result = arrayOutput.flatten()
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='2d')
				ax = fig.add_subplot(111, projection='2d')

				ax.plot_trisurf(colum1, result)
				ax.plot_trisurf(colum1, real)
				"""
				break
			elif (i+1) % len(train_loader) == 0 and (epoch+1) % 10 == 0:
				if loss.item()>=1:
					print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss_Batch: {loss.item():.4f}, Loss_Geral: {MAEGeral.item():.4f}  (usando MSE)')
				else:
					print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss_Batch: {MAE.item():.4f}, Loss_Geral: {MAEGeral.item():.4f}  (usando MAE)')

	print('Treinamento concluído.')

if __name__=="__main__":main();
