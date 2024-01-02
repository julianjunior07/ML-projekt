from matplotlib import pyplot as plt
import numpy as np

measurements = []

#odczyt z pliku
def read_from_file():
    #odczyt formatu danych, node początkowy i końcowy
    file1 = open("data/1.txt")
    
    format = file1.readline()
    #print(format)
    
    node_origin = file1.readline()
    #print(node_origin)
    
    node_destination = file1.readline()
    #print(node_destination)
    
    #odczyt wartosci bitrate 
    for line in file1:
        measurements.append(line.rstrip())
    
    # #jesli chcemy tylko okreslona liczbe
    # for i in range(2000):
    #    line = next(file1).strip()
    #    measurements.append(line)
        
    #konwersja na floaty
    for i in range(0, len(measurements)):
        measurements[i] = float(measurements[i])
        
    #print(measurements)

    file1.close()
    
    #pokazanie wykresiku
    # plt.plot(measurements)
    # plt.xlabel('number')
    # plt.ylabel('bitrate')
    # plt.show()