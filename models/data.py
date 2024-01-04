import os
import math
#klasa zczytująca dane ze wszystkich plików z podanego folderu
#przed uruchomieniem zmień ściezke
# ma 2 pola z danymi: 
#   data - macież, której indeksy to wierzchołki sieci np data[4][5] oznacza dane transwerowe z wierzchołka 4 do 5
#   list_of_data dane zapisane w liście w kolejności czytania plików (pewnie alfabetycznie)
class Data:
    data_path = 'D:\Seba\Studia\Semestr2\ML\Praktyka i testy\praktyka\data'
    count = 0
    data = []
    list_of_data = []
    
    
    def __init__(self, data_path):
        self.data = []
        self.list_of_data = []
        self.data_path = data_path
        for path in os.listdir(self.data_path):
            # check if current path is a file
            file_name = os.path.join(self.data_path, path)
            if os.path.isfile(file_name):
                self.count += 1
         
        self.data = [[ [] for i in range(self.count)] for j in range(self.count)]       
        
        for file in os.scandir(self.data_path):
            if(file.is_file()):
                data = []
                with open(file) as f:
                    
                    f.readline()    #read comment
                    start_node = int(f.readline().rstrip()) #read start node
                    end_node = int(f.readline().rstrip())   #read end node
                    
                    #read data
                    data = f.readlines()
                    
                #map data from strings to floats
                data = list(map(float, data))
                self.data[start_node][end_node] = data
                self.list_of_data.append(data)
    
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
    
    