import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from data import Data

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers=num_layers, batch_first=True)
        self.latent_size = latent_size

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        encoded = hidden.view(-1, self.latent_size)
        decoded, _ = self.decoder(encoded.view(x.size(0), 1, -1))
        return decoded

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for data in train_loader:
            inputs = data
            inputs = torch.stack(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

def detect_anomalies(model, test_loader, threshold):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for data in test_loader:
            inputs = data
            outputs = model(inputs)
            mse = nn.MSELoss(reduction='none')(outputs, inputs).mean(dim=(1, 2))
            anomalies.extend(mse > threshold)

    return anomalies



data = Data()

input_dim = int(len(data.list_of_data[0]))
latent_dim = int(len(data.list_of_data[0])/8)



X_train, X_test = np.array(data.list_of_data), np.array(data.list_of_data)

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, input_dim)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, input_dim)).reshape(X_test.shape)

train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inicjalizacja modelu
model = LSTMAutoencoder(input_size=input_dim, hidden_size=latent_dim, latent_size=latent_dim)

# Trening modelu
train_model(model, train_loader, num_epochs=10, learning_rate=0.001)

# Detekcja anomalii
threshold = 0.04  # Przykładowy próg
anomalies = detect_anomalies(model, test_loader, threshold)

# Wyświetlenie wyników
print("Anomaly mask:", anomalies)