import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tslearn.barycenters import dtw_barycenter_averaging
from minisom import MiniSom

# dla danych us26
# path_figures = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_26'

# dla danych int9
path_figures = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_9'

# dla encoderow
# path_figures = '..\imgs\pattern_swap\\change_three_quarters'

class SOMClustering(): 
    
    def __init__(self, data):
        self.data_series = data

        
    def train(self, sigma=0.3, learning_rate=0.5):
        self.som_x = self.som_y = math.ceil(math.sqrt(math.sqrt(len(self.data_series)))) + 1
        self.som = MiniSom(self.som_x, self.som_y, len(self.data_series[0]), sigma=sigma, learning_rate = learning_rate)
        self.som.random_weights_init(self.data_series)
        self.som.train(self.data_series, 50000)
        
    def plot_som_series_averaged_center(self):
        win_map = self.som.win_map(self.data_series)
        fig, axs = plt.subplots(self.som_x, self.som_y,figsize=(25,25))
        fig.suptitle('Clusters')
        for x in range(self.som_x):
            for y in range(self.som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    for series in win_map[cluster]:
                        axs[cluster].plot(series,c="gray",alpha=0.5) 
                    axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
                cluster_number = x*self.som_y+y+1
                axs[cluster].set_title(f"Cluster {cluster_number}")
        # 
        # plt.savefig(path_figures+'\All_Clusters_26.png', bbox_inches='tight') #zamiast wyświetlania to zapis do pliku
        plt.savefig(path_figures+'\All_Clusters_9.png', bbox_inches='tight') #zamiast wyświetlania to zapis do pliku
        # plt.show()
        plt.close()
        
    def plot_som_series_dba_center(self):
        win_map = self.som.win_map(self.data_series)
        for x in range(self.som_x):
            for y in range(self.som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    for series in win_map[cluster]:
                        plt.plot(series,c="gray",alpha=0.5) 
                    plt.plot(np.average(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red"))
                cluster_number = x*self.som_y+y+1
                plt.figure(f"Cluster {cluster_number}")
        plt.show()
        
    def get_clusters_average(self):
        win_map = self.som.win_map(self.data_series)
        cluster_average = []
        for x in range(self.som_x):
            for y in range(self.som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    cluster_average.append(np.average(np.vstack(win_map[cluster]),axis=0))
        return cluster_average
        
    def get_clusters_dba(self):
        win_map = self.som.win_map(self.data_series)
        cluster_average = []
        for x in range(self.som_x):
            for y in range(self.som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    cluster_average.append(np.average(dtw_barycenter_averaging(np.vstack(win_map[cluster]), axis=0)))
        return cluster_average
    
    
    
    def get_win_map(self):
        return self.som.win_map(self.data_series)
    
    def get_clusters_map(self):
        win_map = self.som.win_map(self.data_series)
        clusters_map = []
        for x in range(self.som_x):
            for y in range(self.som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    clusters_map.append(win_map[cluster])
                    
        return clusters_map
    