import torch.nn as nn
import torch
import copy
import numpy as np
import pandas as pd
import sklearn.model_selection
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
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
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
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
before_dryft_dataset = []
test_dataset = []
ds_anomaly = []
for file in os.scandir(dir_path):
    with open(file) as file:
        data = file.readlines()
        
    data = np.array(list(map(float, data)))
    dataset.append(data)
    before_dryft_dataset.append(data[:3*int(len(data)/4)]) # dryft koncepcji w 3/4 przebiegu
    ds_anomaly.append(data[3*int(len(data)/4):])

# plt.subplot(1,2,1)
# plt.plot(dataset[0])
# plt.subplot(1,2,2)
# plt.plot(dataset[1])
# plt.show()


ds_train, ds_validate, ds_test = [], [], []
for i in range(len(before_dryft_dataset)):
    x, y = sklearn.model_selection.train_test_split(before_dryft_dataset[i], test_size=0.4)
    a, b = sklearn.model_selection.train_test_split(y, test_size=0.5)
    ds_train.append(x)
    ds_validate.append(a)
    ds_test.append(b)
    
train_sequence = np.array(ds_train)
validation_sequence = np.array(ds_validate)
test_sequence = np.array(ds_test)
anomaly_sequence = np.array(ds_anomaly)


def create_dataset(sequence):
    dataset = [torch.tensor(s).unsqueeze(1) for s in sequence]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features
    
train_dataset, seq_len, n_features = create_dataset(train_sequence)
validation_dataset, _, _ = create_dataset(validation_sequence)
test_dataset, _, _ = create_dataset(test_sequence)
test_anomaly_dataset, _, _ = create_dataset(anomaly_sequence)


#model
model = LSTMAutoencoder(seq_len, n_features, embedding_dim=128)
model.to(device)

#train model
def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  #criterion = nn.MSELoss()
  
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(n_epochs):
    model = model.train()
    train_losses = []
    for sequence in train_dataset:
      optimizer.zero_grad()
      sequence = sequence.to(device)
      prediction = model(sequence)
      loss = criterion(prediction, sequence)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
    
    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for sequence in val_dataset:
        sequence = sequence.to(device)
        prediction = model(sequence)
        loss = criterion(prediction, sequence)
        val_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)
  return model.eval(), history

