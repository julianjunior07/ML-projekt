import os
import math
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
#klasa zczytująca dane ze wszystkich plików z podanego folderu
#przed uruchomieniem zmień ściezke
# ma 3 pola z danymi: 
#   data - macież, której indeksy to wierzchołki sieci np data[4][5] oznacza dane transwerowe z wierzchołka 4 do 5
#   list_of_data dane zapisane w liście w kolejności czytania plików (pewnie alfabetycznie)
#   dataset dane w postaci słownika, kluczem są node'y startowy i docelowy.
#       dane dla transmisji z node'a 4 do node'a 8 można pobrać za pomocą dataset[4,8]
class Data(Dataset):
    data = []
    list_of_data = []
    dataset = {}
    normalized_data_list = []
    def __init__(self, data_path='D:\Seba\Studia\Semestr2\ML\dataset1'):
        self.data = []
        self.list_of_data = []
        self.dataset = {}
        # self.data_path = data_path
        # for path in os.listdir(self.data_path):
        #     # check if current path is a file
        #     file_name = os.path.join(self.data_path, path)
        #     if os.path.isfile(file_name):
        #         self.count += 1
        # self.data = [[ [] for i in range(self.count)] for j in range(self.count)]       
        
        for file in os.scandir(data_path):
            if(file.is_file()):
                data = []
                with open(file) as f:
                    
                    f.readline()    #read comment
                    start_node = int(f.readline().rstrip()) #read start node
                    end_node = int(f.readline().rstrip())   #read end node
                    
                    #read data
                    data = f.readlines()
                    
                #map data from strings to floats
                data = np.array(list(map(float, data)))
                #self.data[start_node][end_node] = data
                self.list_of_data.append(data)
                self.dataset[(start_node, end_node)] = data
                
                
        self.normalize_data()
    
    def __getitem__(self, *args):
        if len(args[0]) == 1:
            return self.list_of_data[args[0][0]]
        elif len(args[0]) == 2:
            return self.dataset[(args[0])]
        else:
            raise ValueError('Wymagany jest 1 lub 2 argumenty: start_node, end_node')
    
    def __len__(self):
        return len(self.list_of_data[0])
    
    def get_list_of_data(self):
        return self.list_of_data
    
    def normalize_data(self):
        for i in range(len(self.list_of_data)):
            x_min = np.array(self.list_of_data)[i].min()
            x_max = np.array(self.list_of_data)[i].max()
            self.normalized_data_list.append([])
            for x in range(len(self.list_of_data[i])):
                self.normalized_data_list[i].append((self.list_of_data[i][x]-x_min) / (x_max - x_min))
    
    def get_train_and_test_data(self, percent):
        train_data = []
        test_data = []
        for x in self.list_of_data:
            train_data.append( x[:math.ceil(len(self.list_of_data[0])/percent)])
            test_data.append( x[math.floor(len(self.list_of_data)/percent):])
            
        print(len(self.list_of_data[0]))
        print(len(test_data[0]))
        print(len(train_data[0]))
        return train_data, test_data


    def get_train_and_test_data(self, percent, index):
        train_data = self.list_of_data[index][:math.ceil(len(self.list_of_data[0])/percent)]
        test_data = self.list_of_data[index][math.floor(len(self.list_of_data)/percent):]
        return train_data, test_data
    
    def get_train_and_test_data_normalized(self, percent):
        train_data = []
        test_data = []
        for x in self.normalized_data_list:
            train_data.append( x[:math.ceil(len(self.normalized_data_list[0])/percent)])
            test_data.append( x[math.floor(len(self.normalized_data_list)/percent):])
            
        return train_data, test_data


    def get_train_and_test_data_normalized(self, percent, index):
        train_data = self.normalized_data_list[index][:math.ceil(len(self.normalized_data_list[0])/percent)]
        test_data = self.normalized_data_list[index][math.floor(len(self.normalized_data_list)/percent):]
        return train_data, test_data
    