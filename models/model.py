import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from read_file import *
import math

from sklearn.metrics import mean_absolute_percentage_error as sklearnMAPE

from sklearn.preprocessing import MinMaxScaler
import torch # Library for implementing Deep Neural Network 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import mean_absolute_percentage_error

from clustering import Clustering

# dla danych us26
# path = 'D:\\Polibudka\\Magister\\Sezon 2\\Proj Sieci Komp i ML\\ML\\_dane\\us26\\demands_for_students'
# path_figures = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_26'

#dla danych int9
path = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_dane\int9\demands_for_students'
path_figures = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_9'

path_to_model = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_9\models'
# nie dziala zapis xD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def model_function():
    
    cluster = Clustering(path)
    # print(cluster.data_list[0])
    print("inicjalizacja algorytmu klastrów")
    somclusters = cluster.getSomCluster()
    print("trenowanie algorytmu klastrów")
    somclusters.train(0.3, 0.5)
    print("wyświetlanie wykresów")
    somclusters.plot_som_series_averaged_center()
    print("uzyskanie listy z wartościami średnich")
    clusters_avg_lists = somclusters.get_clusters_average()
    print("liczba uzyskanych klastrów " + str(len(clusters_avg_lists)))
    
    df = []
    
    for k in range(0, 2):
    
        for z in range(len(clusters_avg_lists)): #petla wykonuje sie dla kazdego klastra, liczba pobrana z liczby uzyskanych klastrów
            
            df = pd.DataFrame(data={'value': clusters_avg_lists[z]})
            # df = pd.DataFrame(data={'value': measurements})
            cluster_number = z+1

            #podział danych na treningowe i testowe
            ratio = 0.8
            total_rows = df.shape[0]
            train_size = int(total_rows*ratio)
            
            train_data = df[0:train_size]
            test_data = df[train_size:]

            # print("train")
            # print(train_data[:5])
            # print("test")
            # print(test_data[:5])
            
            print("uruchomienie modelu LSTM")
            print("rozmiar danych: ")
            
            print(df.shape)
            print(train_data.shape)
            print(test_data.shape)
            
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_train = scaler.fit_transform(train_data)
            
            print("wyswietlanie znormalizowanych treningowych wartosci")
            print(*scaled_train[:5]) #pierwsze 5 wierszy treningowe
            
            scaled_test = scaler.fit_transform(test_data)
            print("wyswietlanie znormalizowanych testowych wartosci")
            print(*scaled_test[:5]) #pierwsze 5 wierszy testowe

            #tworzenie danych treningowych
            # sequence_length = 11000  # liczba próbek
            # sequence_length = 950 
            sequence_length = 4000 
            # ustalenie liczby probek
            # samples = len(train_data) * 0.6
            # sequence_length = math.floor(samples)
            
            X_train, y_train = [], []
            for i in range(len(scaled_train) - sequence_length):
                X_train.append(scaled_train[i:i+sequence_length])
                y_train.append(scaled_train[i+1:i+sequence_length+1])
            X_train, y_train = np.array(X_train), np.array(y_train)

            # konwersja na tensory pytorch
            X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
            print("wyswietlanie rozmiaru danych w formie torch.tensor")
            print(X_train.shape,y_train.shape)

            
            #tworzenie danych testowych
            # sequence_length = 3850  # liczba próbek
            # sequence_length = 990
            sequence_length = 3850
            # ustalenie liczby probek
            # samples = len(test_data) * 0.6 
            # sequence_length = math.floor(samples)
            
            X_test, y_test = [], []
            for i in range(len(scaled_test) - sequence_length):
                X_test.append(scaled_test[i:i+sequence_length])
                y_test.append(scaled_test[i+1:i+sequence_length+1])
            X_test, y_test = np.array(X_test), np.array(y_test)
            
            # konwersja na tensory pytorch
            X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
            y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
            # X_test.shape, y_test.shape
            print("wyswietlanie rozmiaru danych w formie torch.tensor")
            print(X_test.shape,y_test.shape)
            
            class LSTMModel_a(nn.Module):
                # input_size : liczba danych wejsciowych jednoczesnie
                # hidden_size : liczba jednostek LSTM 
                # num_layers : liczba warstw LSTM 
                def __init__(self, input_size, hidden_size, num_layers): 
                    super(LSTMModel_a, self).__init__() #inicjalizuje klasę rodzica nn.Module
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, device=device)
                    self.linear = nn.Linear(hidden_size, 1, device=device)
            
                def forward(self, x): # przejscie w przod sieci neuronowej
                    out, _ = self.lstm(x)
                    out = self.linear(out)
                    return out
            
            input_size = 1
            num_layers = 2
            hidden_size = 64
            output_size = 1
            
            #definiowanie modelu, funkcji błędu dla epoch i optymalizacji 
            model = LSTMModel_a(input_size, hidden_size, num_layers).to(device)
            
            # wczytanie modelu dla klastra
            # model = torch.load(path_to_model+'model_klaster_'+str(cluster_number)+'.pth')
            # model.eval()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            print("wyswietlanie modelu")
            print(model)

            
            batch_size = 16
            #dataloader dla danych treningowych
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            #dataloader dla danych testowych
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            #trenowanie modelu
            #ilosc epoch, czyli ile razy algorytm przejdzie przez dane treningowe
            num_epochs = 30
            train_hist =[]
            test_hist =[]
            
            if(k==0):
                file_losses = open(path_figures+"\_epoch_losses_"+str(len(clusters_avg_lists))+"_MAPE_fn.txt", "a")
                file_losses.write("EPOCH LOSSES klaster nr "+str(cluster_number)+ ": \n")
                # file_losses = open("D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_single_file\_epoch_losses.txt", "a")
                # file_losses.write("484   EPOCH LOSSES MAPE: \n")
            elif(k==1):
                file_losses = open(path_figures+"\_epoch_losses_"+str(len(clusters_avg_lists))+"_MSE_fn.txt", "a")
                file_losses.write("EPOCH LOSSES klaster nr "+str(cluster_number)+ ": \n")
                # file_losses = open("D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_single_file\_epoch_losses.txt", "a")
                # file_losses.write("484   EPOCH LOSSES MSE: \n")
                loss_fn = torch.nn.MSELoss(reduction='mean').to(device=device)
            
            # pętla treningowa
            for epoch in range(num_epochs):
                total_loss = 0.0

                # trenowanie
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    predictions = model(batch_X)
                    if(k==0):
                        loss = mean_absolute_percentage_error(predictions, batch_y).to(device)
                    elif(k==1):
                        loss = loss_fn(predictions, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                #obliczenie sredniego bledu dla treningu
                average_loss = total_loss / len(train_loader)
                train_hist.append(average_loss)

                #sprawdzenie poprawnosci danych testowych
                model.eval()
                with torch.no_grad():
                    total_test_loss = 0.0

                    for batch_X_test, batch_y_test in test_loader:
                        batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                        predictions_test = model(batch_X_test)
                        if(k==0):
                            test_loss = mean_absolute_percentage_error(predictions_test, batch_y_test).to(device)
                        elif(k==1):
                            test_loss = loss_fn(predictions_test, batch_y_test)

                        total_test_loss += test_loss.item()

                    #sredni blad
                    average_test_loss = total_test_loss / len(test_loader)
                    test_hist.append(average_test_loss)
                if (epoch+1)%10==0:
                    print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.7f}, Test Loss: {average_test_loss:.7f}')
                    file_losses.write(f'    Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.7f}, Test Loss: {average_test_loss:.7f} \n')
                    
            file_losses.close()
            
            #ile probek ma przewidziec model
            num_forecast_steps = math.floor(len(test_data)/2)

            #konwersja na numpy
            sequence_to_plot = X_test.squeeze().cpu().numpy()

            #ostatni punkt danych jako punkt startowy
            historical_data = sequence_to_plot[-1]
            print("wyswietlanie rozmiaru danych historycznych")
            print(historical_data.shape)

            #lista z predykcjami
            forecasted_values = []

            #predykcja wartosci przez model
            with torch.no_grad():
                for _ in range(num_forecast_steps*2):
                    #przygotowanie tensora dla danych przeszlych
                    historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
                    #uzycie modelu do przewidzenia nastepnej wartosci
                    predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]

                    #dodanie przewidzanej wartosci do listy
                    forecasted_values.append(predicted_value[0])

                    #zamiana danych historycznych na przewidziane
                    historical_data = np.roll(historical_data, shift=-1)
                    np.append(historical_data, predicted_value)
                    

            # przeskalowanie wartosci przewidzianych
            forecasted_cases = scaler.inverse_transform(np.expand_dims(forecasted_values, axis=0)).flatten() 
            
            # rysowanie wykresu
            fig, ax = plt.subplots(figsize=(15,5))
            plt.plot(train_data, label='train data')
            plt.plot(test_data, label='test data')
            plt.plot(test_data.index, forecasted_cases, label='forecasted')
            plt.plot([])
            # cluster_number = z+1
            plt.suptitle(f"Cluster {cluster_number}")
            # plt.suptitle(f"File 177")
            
            plt.xlabel('number')
            plt.ylabel('bitrate')
            plt.legend()
            plt.tight_layout()
            if(k==0):
                plt.savefig(path_figures+'\Cluster_'+str(cluster_number)+'_MAPE_fn.png')
                # plt.savefig('D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_single_file\File_484_MAPE_fn.png')
            elif(k==1):
                plt.savefig(path_figures+'\Cluster_'+str(cluster_number)+'_MSE_fn.png')
                # plt.savefig('D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_single_file\File_484_MSE_fn.png')

            plt.close()
            # plt.show()
            
                # zapis modelu dla jedego klastra
            # torch.save(model, path_to_model+'model_klaster_'+str(cluster_number)+'.pth')
            
            #blad
            error_mape = sklearnMAPE(
                y_true=test_data['value'],
                y_pred=forecasted_cases
            )

            print("Klaster nr " + str(cluster_number) + " wyswietlanie bledu MAPE")
            # print("File 484 wyswietlanie bledu MAPE")
            print(f"test error MAPE: {error_mape*100}%")
            
            # zapis wartosci bledu do pliku
            if(k==0):
                file_errors = open(path_figures+"\error_values_"+str(len(clusters_avg_lists))+"_MAPE_fn.txt", "a")
                # file_errors = open("D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_single_file\error_values_MAPE_fn.txt", "a")
            elif(k==1):
                file_errors = open(path_figures+"\error_values_"+str(len(clusters_avg_lists))+"_MSE_fn.txt", "a")
                # file_errors = open("D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_single_file\error_values_MSE_fn.txt", "a")
            file_errors.write("MAPE klaster nr "+str(cluster_number)+ ": " +str(error_mape*100)+"% \n")
            # file_errors.write("MAPE file 484: " +str(error_mape*100)+"% \n")
            file_errors.close()

        k+1
 
model_function()







