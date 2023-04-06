import datetime
import csv
import shapefile
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

class footprint_display:
    def __init__(self):
        shape = shapefile.Reader("resources/shapefiles/idn_admbnda_adm1_bps_20200401.shp")
        
        for shapeRecord in shape.shapeRecords():
            if shapeRecord.record[2] == 'Dki Jakarta':
                jakarta_shapes = shapeRecord.shape.__geo_interface__['coordinates']
            elif shapeRecord.record[2] == 'Jawa Barat':
                barat_shapes = shapeRecord.shape.__geo_interface__['coordinates']
            elif shapeRecord.record[2] == 'Banten':
                banten_shapes = shapeRecord.shape.__geo_interface__['coordinates']

        jawa_coord = []

        for shapes in [jakarta_shapes, barat_shapes, banten_shapes]:
            shapeCoords_length_list = [[shapeCoord[0], len(shapeCoord[0])] for shapeCoord in shapes]
            shapeCoords_length_list.sort(key=lambda x:x[1], reverse=True)
            shapeCoord = shapeCoords_length_list[0][0]
            jawa_coord.append([[i[0] for i in shapeCoord], [i[1] for i in shapeCoord]])

        
        
            

    
            
