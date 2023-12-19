
measurements = []
def read_from_file():
    file1 = open("data/1.txt")
    
    format = file1.readline()
    print(format)
    
    node_origin = file1.readline()
    print(node_origin)
    
    node_destination = file1.readline()
    print(node_destination)
    
    for line in file1:
        measurements.append(line.rstrip())
        
    #print(measurements)


    file1.close()