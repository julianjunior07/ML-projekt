import torch.nn as nn
import torch
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from somclustering import SOMClustering
import math
from sklearn.preprocessing import MinMaxScaler
from encoder import LSTMAutoencoder
from encoder import device, train_model

dryft_plcements = [
    "\change_three_quarters",
    "\change_halfway"
]
dryft_types = [
    "\\bitrate_fluctuation",
    "\pattern_swap",
    "\sum_diff"
]
DRYFT_PLACEMENT=3/4

#funkcja pomocnicza do orabiania danych
def create_dataset(sequences):
  scaler = MinMaxScaler()
  sequences = scaler.fit_transform(sequences)
  sequences = np.array(sequences).astype(np.float32)
  dataset = [torch.tensor(s).unsqueeze(1) for s in sequences]
  _, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features

for dryft_type in dryft_types:
    for dryft_placement in dryft_plcements:
        try:
            dir_path = '..\data'+dryft_type+dryft_placement
        except IOError:
            print(IOError)
            print("Path is not correct")

        dataset = []
        for file in os.scandir(dir_path):
            with open(file) as file:
                data = file.readlines()
                
            data = np.array(list(map(float, data)))
            dataset.append(data)
        
        data_frame_len = int(math.floor(len(dataset[0])/16))



        som = SOMClustering(dataset)
        som.train()
        clusters_map = som.get_clusters_map()
        som.plot_som_series_averaged_center()


        model_cnt = 1
        MODELS_PATH = '..\\trained_models'+dryft_type+dryft_placement
        #trenowanie modeli
        for cluster in clusters_map:
            train_dataset = []
            test_dataset = []
            validate_dataset = []

            #podział na zbiory treningowe, walidacyjne i testowe. Proporcje do przedyskutowania
            for series in cluster:

                train_ds, validate_ds = train_test_split(series, test_size=0.625)     # 6/16 na train_dataset 
                validate_ds, test_ds = train_test_split(validate_ds, test_size=0.8)   # 2/16 na validate_set i reszta na test

                train_ds = np.append(train_ds, [train_ds[len(train_ds)-1]])    #dodaje jedną próbkę żeby się ładnie dzieliło na 16 równych części
                    
                train_dataset.extend(np.split(np.array(train_ds), 6))
                validate_dataset.extend(np.split(np.array(validate_ds), 2))
                test_dataset.extend(np.split(np.array(test_ds), 8))

            train_sequences, seq_len, n_features = create_dataset(train_dataset)
            test_sequences, _, _ = create_dataset(test_dataset)
            validate_sequences, _, _ = create_dataset(validate_dataset)


            #model
            model = LSTMAutoencoder(seq_len, n_features, embedding_dim=128)  
            model.to(device)
            model, history = train_model(model, train_sequences, train_sequences, n_epochs=50)
            print("saveing model")
            torch.save(model, MODELS_PATH+ f'\{model_cnt}')
            
            
            ax = plt.figure().gca()
            ax.plot(history['train'])
            ax.plot(history['val'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'test'])
            plt.title('Loss over training epochs')
            plt.savefig('..\imgs'+dryft_type+dryft_placement+'\model_training_'+ str(model_cnt)+'.png')
            model_cnt+=1
        
        #train model
