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
        return out

def createSequences(data, seq_length):
    x=[]
    y=[]
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.stack(x), torch.stack(y)

def mainLSTM(batch_size, sequence_Length, hidden_size, num_layers, learning_rate, tamanho_populacao=50, input_size=1,num_epochs=30, output_size=1):
    x_train = pd.read_csv("/home/dragon/Python/Algoritmo_genetico/LSTM_project/train.csv")
    x_test = pd.read_csv("/home/dragon/Python/Algoritmo_genetico/LSTM_project/test.csv")
    x_trainNumpy = x_train.to_numpy()
    x_testNumpy = x_test.to_numpy()

    x_train = np.vstack(x_trainNumpy[:,1]).astype(np.complex128)
    x_test = np.vstack(x_testNumpy[:,1]).astype(np.complex128)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    MAEs = np.zeros(tamanho_populacao)
    for i in range(tamanho_populacao):
        x_train, y_train = createSequences(x_train, sequence_Length[i])
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size[i], shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(input_size, hidden_size[i], num_layers[i], output_size).to(device)
        criterion = nn.MSELoss()
        evaluation = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate[i])

        find = False
        arrayOutput=[]
        for epoch in range(num_epochs):
            if find == True:
                break
            for _, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                out = model(x_train)
                loss = criterion(outputs, targets)
                MAE = evaluation(outputs, targets)
                MAEGeral = evaluation(out, y_train)
                optimizer.zero_grad()
#                if loss.item()<1:
#                    MAE.backward()
#                else:
                loss.backward()
                optimizer.step()
                MAEs[i] = MAEGeral.item()
                print(f'batch_size={batch_size[i]}, sequence_Length={sequence_Length[i]}, hidden_size={hidden_size[i]}, num_layers={num_layers[i]}, learning_rate={learning_rate[i]}\nMAE={MAEGeral.item()}\n')
        return MAEs
        print('Treinamento concluído.')

if __name__=="__main__":main();
