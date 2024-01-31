import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math

import numpy as np
import torch.nn as nn
import torch

from encoder import LSTMAutoencoder
from encoder import device, train_model, detect_anomalies

def create_dataset(sequences):
    scaler = StandardScaler()
    sequences = scaler.fit_transform(sequences)
    sequences = np.array(sequences).astype(np.float32)
    dataset = [torch.tensor(s).unsqueeze(1) for s in sequences]
    _, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features

def normalize_data(dataset):
    for i in range(len(dataset)):
        x_min = np.array(dataset)[i].min()
        x_max = np.array(dataset)[i].max()
        for x in range(len(dataset[i])):
            dataset[i][x]=((dataset[i][x]-x_min) / (x_max - x_min))

def train_test_split(series, train_size, TIME_STAMPS):
    train_len = math.floor(len(series)/TIME_STAMPS*train_size)
    train = series[:train_len*TIME_STAMPS]
    test  = series[train_len*TIME_STAMPS:]
    return train, test


FILE_PATH = '..\data\\bitrate_fluctuation\change_three_quarters\\1.txt'
MODELS_PATH = '..\\trained_models\\test_model'
 # możliwe rozmiary: 5, 17, 85, 227, 1135, 3859
TIME_STAMPS = 85
EPOCHS = 150
DIM = 16

with open(FILE_PATH) as file:
    data = file.readlines()
    
data = np.array(list(map(float, data)))


# plt.plot(data)
# plt.show()
train_dataset = []
test_normal_dataset = []
test_anomaly_dataset = []
validate_dataset = []

#normalize_data([data])

test_normal_ds, test_anomaly_ds = train_test_split(data, train_size=0.75, TIME_STAMPS=TIME_STAMPS)
train_ds, test_normal_ds = train_test_split(test_normal_ds, train_size=0.8, TIME_STAMPS=TIME_STAMPS)
validate_ds, train_ds = train_test_split(train_ds, train_size=0.8, TIME_STAMPS=TIME_STAMPS)


train_dataset = (np.array(train_ds).reshape(int(len(train_ds)/TIME_STAMPS),TIME_STAMPS))
validate_dataset = (np.array(validate_ds).reshape(int(len(validate_ds)/TIME_STAMPS),TIME_STAMPS))
test_normal_dataset = (np.array(test_normal_ds).reshape(int(len(test_normal_ds)/TIME_STAMPS),TIME_STAMPS))
test_anomaly_dataset = (np.array(test_anomaly_ds).reshape(int(len(test_anomaly_ds)/TIME_STAMPS),TIME_STAMPS))

print(f'train_dataset len {len(train_dataset)}')

train_sequences, seq_len, n_features = create_dataset(train_dataset)
test_normal_sequences, _, _ = create_dataset(test_normal_dataset)
test_anomaly_sequences, _, _ = create_dataset(test_anomaly_dataset)
validate_sequences, _, _ = create_dataset(validate_dataset)



# plt.plot(train_sequences[0])
# plt.show()
# print(train_sequences[0].shape)

print(f'train sequences len: {len(train_sequences)}')

model = LSTMAutoencoder(seq_len, n_features,embedding_dim=DIM)  
model, history, train_loss = train_model(
    model, 
    train_sequences, 
    validate_sequences, 
    n_epochs=EPOCHS
    )

print("saveing model")
torch.save(model, MODELS_PATH)

#zapis nauki modelu
ax = plt.figure().gca()
ax.plot(history['train'])
ax.plot(history['val'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.savefig('..\imgs\model_training_test_model.png')
plt.close()

train_predict, train_loss = detect_anomalies(model, train_sequences)
THRESHOLD = np.percentile(train_loss, 95)
print(THRESHOLD)

test_normal_predictions, test_normal_loss = detect_anomalies(model, test_normal_sequences)
test_anomaly_predictions, test_anomaly_loss = detect_anomalies(model, test_anomaly_sequences)


fig, axs = plt.subplots(
                nrows=2,
                ncols=6,
                sharey=True,
                sharex=True,
                figsize=(22, 8)
            )

i=0
for seq in test_normal_sequences:
    if i<6:
        prediction = test_normal_predictions[i]
        axs[0,i].plot(seq, label="test_normal")
        axs[0,i].plot(prediction, label = "prediction")
        axs[0,i].set_title(f'Normal (loss: {test_normal_loss[i]})')
        axs[0,i].legend()
    i+=1
    
i=0
for seq in test_anomaly_sequences:
    if i<6:
        prediction = test_anomaly_predictions[i]
        axs[1,i].plot(seq, label="test_anomaly")
        axs[1,i].plot(prediction, label = "prediction")
        axs[1,i].set_title(f'Anomaly (loss: {test_anomaly_loss[i]})')
        axs[1,i].legend()    
    i+=1
          
fig.tight_layout()
plt.xlabel("Time Stamps")
plt.ylabel("Normalized Bitrate")
plt.savefig('..\imgs\models_prediction_test_model.png')
plt.close()