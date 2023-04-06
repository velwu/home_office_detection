import datetime
import csv
import numpy as np

def read_dmp_data(path):
    data = {}
    with open(path) as f:
        for uuid, start_time, hour, duration, visit_type, lat, lon in csv.reader(f):
            if uuid != 'id':
                if uuid not in data:
                    data[uuid] = []
                data[uuid].append([
                    datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'),
                    float(lat),
                    float(lon)])
    
    for uuid in list(data):
        data[uuid].sort()
    
    return data

def convert2matrix(footprint):
    matrix = [[None for j in range(192)] for i in range((footprint[-1][0].date() - footprint[0][0].date()).days+1)]

    for start_time, lat, lon in footprint:
        row = (start_time.date() - footprint[0][0].date()).days
        col = int((start_time - start_time.replace(hour=0, minute=0)).total_seconds()/(15*60))

        matrix[row][col] = lat
        matrix[row][col+96] = lon

    _, pervious_lat, pervious_lon = footprint[0]

    for row in range(len(matrix)):
        for col in range(96):
            if matrix[row][col] == None:
                matrix[row][col] = pervious_lat
                matrix[row][col+96] = pervious_lon
            else:
                pervious_lat = matrix[row][col]
                pervious_lon = matrix[row][col+96]
    
    return np.array(matrix, dtype=float)
        

        
        
            

    
            
