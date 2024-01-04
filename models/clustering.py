import matplotlib.pyplot as plt
import numpy as np
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Algorithms

from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

from somclustering import SOMClustering
from data import Data



class Clustering:

    def __init__(self, path):
        self.path_to_data = path
        self.init_data()
        self.nomalize_data()
        
    def init_data(self):
        self.data = Data(self.path_to_data)
        self.dataset = self.data.data
        self.data_list = np.array(self.data.list_of_data)
    #scaling data
     
    def nomalize_data(self):
        for i in range(len(self.data_list)):
            x_min = self.data_list[i].min()
            x_max = self.data_list[i].max()
            for x in range(len(self.data_list[i])):
                self.data_list[i][x] = (self.data_list[i][x]-x_min) / (x_max - x_min)
        
        
    def getSomCluster(self):
        return SOMClustering(self.data_list)
    


#przykład uzycia klastrów
path = 'D:\Seba\Studia\Semestr2\ML\dataset1'
cluster = Clustering(path)

somclusters = cluster.getSomCluster()
somclusters.train(0.3, 0.5)
somclusters.plot_som_series_averaged_center()
clusters_avg_lists = somclusters.get_clusters_average()

