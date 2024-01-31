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
from encoder import device, train_model, detect_anomalies


dryft_types = [
    "\\bitrate_fluctuation",
    #"\pattern_swap",
    #"\sum_diff"
]
dryft_plcements = [
    "\change_three_quarters",
    #"\change_halfway"
]

DRYFT_PLACEMENT=3/4

EPOCHS = 100
# możliwe rozmiary: 5, 17, 85, 227, 1135, 3859
TIME_STAMPS = 1135
        
        
#funkcja pomocnicza do orabiania danych
def create_dataset(sequences):
  scaler = MinMaxScaler()
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

        if dryft_placement == "\change_halfway":
            DRYFT_PLACEMENT = 1/2

        som = SOMClustering(dataset)
        som.train()
        clusters_map = som.get_clusters_map()
        som.plot_som_series_averaged_center()


        model_cnt = 1
  
        #trenowanie modeli
        cluster_cnt=1
        for cluster in clusters_map:
            train_dataset = []
            test_normal_dataset = []
            test_anomaly_dataset = []
            validate_dataset = []
            
          #  normalize_data(cluster)
            
            #podział na zbiory treningowe, walidacyjne i testowe. Proporcje do przedyskutowania
            for series in cluster:
                # plt.plot(series)
                # plt.show()
                #okna po 1206 próbek                
                
                test_normal_ds, test_anomaly_ds = train_test_split(series, train_size=DRYFT_PLACEMENT, TIME_STAMPS=TIME_STAMPS)
                #np.random.shuffle(test_normal_ds)
                train_ds, test_normal_ds = train_test_split(test_normal_ds, train_size=0.8, TIME_STAMPS=TIME_STAMPS)
                validate_ds, train_ds = train_test_split(train_ds, train_size=0.8, TIME_STAMPS=TIME_STAMPS)


                #train_ds = np.append(train_ds, [train_ds[len(train_ds)-1]])    #dodaje jedną próbkę żeby się ładnie dzieliło na 16 równych części
  
                train_dataset.extend(np.array(train_ds).reshape(int(len(train_ds)/TIME_STAMPS),TIME_STAMPS))
                validate_dataset.extend(np.array(validate_ds).reshape(int(len(validate_ds)/TIME_STAMPS),TIME_STAMPS))
                test_normal_dataset.extend(np.array(test_normal_ds).reshape(int(len(test_normal_ds)/TIME_STAMPS),TIME_STAMPS))
                test_anomaly_dataset.extend(np.array(test_anomaly_ds).reshape(int(len(test_anomaly_ds)/TIME_STAMPS),TIME_STAMPS))
                
            train_sequences, seq_len, n_features = create_dataset(train_dataset)
            test_normal_sequences, _, _ = create_dataset(test_normal_dataset)
            test_anomaly_sequences, _, _ = create_dataset(test_anomaly_dataset)
            validate_sequences, _, _ = create_dataset(validate_dataset)


            #train model
            model = LSTMAutoencoder(seq_len, n_features)  
            model.to(device)
            model, history, train_loss = train_model(model, train_sequences, validate_sequences, n_epochs=50)
            
            file_losses = open('..\imgs'+dryft_type+dryft_placement+'\model_training_losses'+ str(model_cnt)+'.txt', "a")
            file_losses.write(f'{train_loss}\n')
            file_losses.close()
            
            print("saveing model")
            MODELS_PATH = '..\\trained_models'+f'{dryft_type}\\' + f'{dryft_placement}\\'
            torch.save(model, MODELS_PATH+ f'\{model_cnt}')
        
            #zapis nauki modelu
            ax = plt.figure().gca()
            ax.plot(history['train'])
            ax.plot(history['val'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'test'])
            plt.title('Loss over training epochs')
            plt.savefig('..\imgs'+dryft_type+dryft_placement+'\model_training_'+ str(model_cnt)+'.png')
            model_cnt+=1
            plt.close()
            
            #testing model
            train_predict, train_loss = detect_anomalies(model, train_sequences)
            THRESHOLD = np.percentile(train_loss, 95)
            print(THRESHOLD)
    
            #test data without anomalies
            test_normal_predictions, test_normal_loss = detect_anomalies(model, test_normal_sequences)
            # correct = sum(l <= THRESHOLD for l in test_loss)
            # print(f'Correct normal predictions: {correct}/{len(test_normal_sequences)}')
            test_anomaly_predictions, test_anomaly_loss = detect_anomalies(model, test_anomaly_sequences)

            fig, axs = plt.subplots(
                nrows=2,
                ncols=6,
                sharey=True,
                sharex=True,
                figsize=(22, 8)
            )
            fig.suptitle(f'Cluster: {cluster_cnt}')
            
            i=0
            while enumerate(test_normal_sequences) and i < 5:
                seq = test_normal_sequences[i]
                prediction = test_normal_predictions[i]
                axs[0,i].plot(seq, label="test_normal")
                axs[0,i].plot(prediction, label = "prediction")
                axs[0,i].set_title(f'Normal (loss: {test_normal_loss[i]})')
                axs[0,i].legend()
                i+=1
            i=0
            while enumerate(test_anomaly_sequences) and i < 5:
                seq = test_anomaly_sequences[i]
                prediction = test_anomaly_predictions[i]
                axs[1,i].plot(seq, label="test_anomaly")
                axs[1,i].plot(prediction, label = "prediction")
                axs[1,i].set_title(f'Anomaly (loss: {test_anomaly_loss[i]})')
                axs[1,i].legend()    
                i+=1            
            fig.tight_layout()
            plt.xlabel("Time Stamps")
            plt.ylabel("Normalized Bitrate")
            plt.savefig('..\imgs'+dryft_type+dryft_placement+'\models_prediction_'+str(cluster_cnt)+'.png')
            plt.close()
            cluster_cnt+=1