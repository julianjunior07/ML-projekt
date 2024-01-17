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
from sklearn.preprocessing import MinMaxScaler



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


def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for train_sequence in train_dataset:
      optimizer.zero_grad()

      train_sequence = train_sequence.to(device)
      output = model(train_sequence)

      loss = criterion(output, train_sequence)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for train_sequence in val_dataset:

        train_sequence = train_sequence.to(device)
        output = model(train_sequence)

        loss = criterion(output, train_sequence)
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


def detect_anomalies(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for sequence in dataset:
      sequence = sequence.to(device)
      prediction = model(sequence)

      loss = criterion(prediction, sequence)

      predictions.append(prediction.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses

# def detect_anomalies(model, test_dataset, threshold):
#     model.eval()
#     anomalies = []
#     with torch.no_grad():
#         for sequence in test_dataset:
#             sequence = sequence.to(device)
#             prediction = model(sequence)
#             mse = nn.MSELoss(reduction='sum')(prediction, sequence).mean(dim=(1, 2))
#             anomalies.extend(mse > threshold)

#     return anomalies
