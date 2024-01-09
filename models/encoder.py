import torch.nn as nn
import torch
import copy
import numpy as np
import pandas as pd
import sklearn.model_selection
import os
import matplotlib.pyplot as plt
from somclustering import SOMClustering
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    
    self.seq_len = seq_len
    self.n_features =  n_features
    self.embedding_dim = embedding_dim
    self.hidden_dim = 2 * embedding_dim
    
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers = 1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers = 1,
      batch_first=True
    )
    

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features)) #batch_size, seq_len, n_features
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.seq_len = seq_len
    self.input_dim =  input_dim
    self.hidden_dim = 2 * input_dim
    self.n_features =  n_features
    
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers = 1,
      batch_first=True
      
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers = 1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
    
    
  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)


class LSTMAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(LSTMAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


#Dane
#Poklastruj i zbnierz dane w formacie jak poniżej dla każdego kalstra
# na razie leci prowizorka nauki z jednego przepływu danych
dir_path = '..\data\pattern_swap\change_three_quarters'
dataset = []
# before_dryft_dataset = []
# test_dataset = []
# ds_anomaly = []
for file in os.scandir(dir_path):
    with open(file) as file:
        data = file.readlines()
        
    data = np.array(list(map(float, data)))
    dataset.append(data)
    # before_dryft_dataset.append(data[:3*int(len(data)/4)]) # dryft koncepcji w 3/4 przebiegu
    # ds_anomaly.append(data[3*int(len(data)/4):])


som = SOMClustering(dataset)
som.train()
clusters_map = som.get_clusters_map()
som.plot_som_series_averaged_center()

DRYFT_PLACEMENT=3/4


def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.L1Loss(reduction='sum').to(device)
  criterion = nn.MSELoss().to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()
    total_loss = 0.0
    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())
      total_loss += loss.item()

    #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    print(f'Epoch {epoch}/{n_epochs}, Loss: {total_loss/len(train_dataset)}')
  # model.load_state_dict(best_model_wts)
  return model.eval()

def detect_anomalies(model, test_dataset, threshold):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for sequence in test_dataset:
            sequence = sequence.to(device)
            prediction = model(sequence)
            mse = nn.MSELoss(reduction='none')(prediction, sequence).mean(dim=(1, 2))
            anomalies.extend(mse > threshold)

    return anomalies
  
  
#funkcja pomocnicza do orabiania danych
def create_dataset(sequences):
  sequences = np.array(sequences).astype(np.float32)
  dataset = [torch.tensor(s).unsqueeze(1) for s in sequences]
  _, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features


models_cnt = 1
MODELS_PATH = '..\\trained_models'
#trenowanie modeli
for cluster in clusters_map:
  train_dataset = cluster[:math.ceil(len(cluster)*2/3)]
  test_dataset =  cluster

  for series in train_dataset:
    series = series[:int(len(series)*DRYFT_PLACEMENT)] #usuwamy dryft z serii do trenowania modelu    
  
  train_sequences, seq_len, n_features = create_dataset(train_dataset)
  test_sequences, _, _ = create_dataset(test_dataset)
    
  #model
  model = LSTMAutoencoder(seq_len, n_features, embedding_dim=128)  
  model.to(device)
  model = train_model(model, train_sequences, train_sequences, n_epochs=100)
  print("saveing model")
  torch.save(model, MODELS_PATH+ f'\{models_cnt}')
  # ax = plt.figure().gca()
  # ax.plot(history['train'])
  # ax.plot(history['val'])
  # plt.ylabel('Loss')
  # plt.xlabel('Epoch')
  # plt.legend(['train', 'test'])
  # plt.title('Loss over training epochs')
  # plt.show();
  
#train model
