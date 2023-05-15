import os
import shutil
import datetime
import csv
import shapefile
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ------------- create and clean necessary folders -------------------
if os.path.exists("display/temp"):
    shutil.rmtree("display/temp")

for path in ['display/footprint']:
    for root, folder, files in os.walk(path):
        for file in files:
            #if file.split(sep='_')[-1] != 'footprint.gif':
            if True:
                os.remove(os.path.join(root, file))

for path in [
    'display',
    'display/temp',
    'display/footprint']:
    if os.path.exists(path) == False:
        os.mkdir(path)
# --------------------------------------------------------------------

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
                    float(lon),
                    int(duration)
                ])
    
    for uuid in list(data):
        data[uuid].sort()
    
    return data

def read_home_work_data(path):
    df_hw = pd.read_csv(path)
    df_hw["id"] = df_hw["id"].astype(str)

    return df_hw

def footprint2matrix(footprint):
    matrix = [[None for j in range(288)] for i in range((footprint[-1][0].date() - footprint[0][0].date()).days+1)]

    for start_time, lat, lon,duration in footprint:
        row = (start_time.date() - footprint[0][0].date()).days
        col = int((start_time - start_time.replace(hour=0, minute=0)).total_seconds()/(15*60))

        matrix[row][col] = lat
        matrix[row][col+96] = lon
        matrix[row][col+192] = duration

    _, previous_lat, previous_lon, previous_dur = footprint[0]

    for row in range(len(matrix)):
        for col in range(96):
            if matrix[row][col] == None:
                matrix[row][col] = previous_lat
                matrix[row][col+96] = previous_lon
                matrix[row][col+192] = previous_dur
            else:
                previous_lat = matrix[row][col]
                previous_lon = matrix[row][col+96]
                previous_dur = matrix[row][col+192]
    
    return np.array(matrix, dtype=float)

def matrix2footprint(matrix):
    footprint = []

    if len(matrix.shape) == 1:
        matrix = matrix.reshape(1, -1)

    for row in range(matrix.shape[0]):
        for col in range(96):
            footprint.append([
                datetime.datetime(2023,4,1)+datetime.timedelta(days=row, minutes=col*15),
                matrix[row, col],
                matrix[row, col+96]
            ])
    
    return footprint

class footprint_display:
    def __init__(self):
        self.MAP_SIZE_COEFFICIENT = 0.5
        self.DISPLAY_TAIL_LEN = 20
        shape = shapefile.Reader("resource/shapefiles/idn_admbnda_adm1_bps_20200401.shp")
        
        self.border_list = []
        
        for shapeRecord in shape.shapeRecords():
            area = shapeRecord.shape.__geo_interface__['coordinates']
            for borders in area:
                if type(borders[0]) == tuple:
                    self.border_list.append([[i[0] for i in borders], [i[1] for i in borders]])
                else:
                    for border in borders:
                        self.border_list.append([[i[0] for i in border], [i[1] for i in border]])

        self.latlon_range = {
            'lat':{
                'max':max([max(border[1]) for border in self.border_list]), 
                'min':min([min(border[1]) for border in self.border_list])}, 
            'lon':{
                'max':max([max(border[0]) for border in self.border_list]),
                'min':min([min(border[0]) for border in self.border_list])}}
        self.latlon_range['area'] = (self.latlon_range['lat']['max']-self.latlon_range['lat']['min'])*\
                                    (self.latlon_range['lon']['max']-self.latlon_range['lon']['min'])
    
    def plot_map(self, latlon_list, group, img_name, centers=None, fix_map = True):
        file_path = os.path.join("display", "footprint", f"{img_name}.png")
        data = pd.DataFrame({
            'lat':[i[0] for i in latlon_list],
            'lon':[i[1] for i in latlon_list],
            'group':group
        })

        if fix_map:
            latlon_range = self.latlon_range
            MAP_SIZE = 1 * self.MAP_SIZE_COEFFICIENT
        else:
            latlon_range = {
                'lat':{'max':np.max(data['lat']), 'min':np.min(data['lat'])},
                'lon':{'max':np.max(data['lon']), 'min':np.min(data['lon'])},
                'area':(np.max(data['lat'])-np.min(data['lat']))*(np.max(data['lon'])-np.min(data['lon']))}
            MAP_SIZE = self.MAP_SIZE_COEFFICIENT*(self.latlon_range['area']/latlon_range['area'])**0.5
        
        plt.figure(figsize=(
            (latlon_range['lon']['max']-latlon_range['lon']['min'])*MAP_SIZE, 
            (latlon_range['lat']['max']-latlon_range['lat']['min'])*MAP_SIZE))

        for coor_X, coor_Y in self.border_list:
            plt.plot(coor_X, coor_Y)
        
        sns.scatterplot(data=data, x='lon', y='lat', hue='group')

        if centers:
            for center_lat, center_lon in centers:
                plt.scatter([center_lon], [center_lat], marker='s', s=60, c='k')
        
        plt.xlim([latlon_range['lon']['min'], latlon_range['lon']['max']])
        plt.ylim([latlon_range['lat']['min'], latlon_range['lat']['max']])
        plt.savefig(file_path)
        plt.close()


    def plot_gif(self, matrix, img_name, centers=None, fix_map=True, home_work_data=None):
        file_path = os.path.join("display", "footprint", f"{img_name}.gif")

        if os.path.exists(file_path) == False:
            os.mkdir(os.path.join("display", "temp", img_name))

            if len(matrix.shape) == 1:
                matrix = matrix.reshape(1, -1)
            
            lat_matrix, lon_matrix, dur_matrix = matrix[:, :96], matrix[:, 96:192], matrix[:, 192:]

            if fix_map:
                latlon_range = self.latlon_range
                MAP_SIZE = 1 * self.MAP_SIZE_COEFFICIENT
            else:
                latlon_range = {
                    'lat':{'max':np.max(lat_matrix), 'min':np.min(lat_matrix)},
                    'lon':{'max':np.max(lon_matrix), 'min':np.min(lon_matrix)},
                    'area':(np.max(lat_matrix)-np.min(lat_matrix))*(np.max(lon_matrix)-np.min(lon_matrix))}
                MAP_SIZE = self.MAP_SIZE_COEFFICIENT*(self.latlon_range['area']/latlon_range['area'])**0.5

            for col in range(0, 96):
                plt.figure(figsize=(
                    (latlon_range['lon']['max']-latlon_range['lon']['min'])*MAP_SIZE, 
                    (latlon_range['lat']['max']-latlon_range['lat']['min'])*MAP_SIZE))

                for coor_X, coor_Y in self.border_list:
                    plt.plot(coor_X, coor_Y)

                for row in range(matrix.shape[0]):
                    list_lon = lon_matrix[row, max(0, col-self.DISPLAY_TAIL_LEN):col].tolist()
                    list_lat = lat_matrix[row, max(0, col-self.DISPLAY_TAIL_LEN):col].tolist()
                    list_dur = dur_matrix[row, max(0, col-self.DISPLAY_TAIL_LEN):col].tolist()

                    plt.scatter(list_lon,list_lat,s=[x // 50 for x in list_dur],zorder=1,)
                    plt.plot(list_lon,list_lat,zorder=2,)

                # if len(home_work_data) > 0:
                poi_point_size = 500
                poi_shape = "*"
                home_point_x = home_work_data['home_lon']
                home_point_y = home_work_data['home_lat']
                home_color = 'green'
                work_point_x = home_work_data['work_lon']
                work_point_y = home_work_data['work_lat']
                work_color = 'darkblue'

                x_limit_min = [latlon_range['lon']['min']]
                x_limit_max = [latlon_range['lon']['max']]
                y_limit_min = [latlon_range['lat']['min']]
                y_limit_max = [latlon_range['lat']['max']]
                
                if centers:
                    for center_lat, center_lon in centers:
                        x_limit_min.append(center_lon)
                        x_limit_max.append(center_lon)
                        y_limit_min.append(center_lat)
                        y_limit_max.append(center_lat)
                        enricher_pca_dist = str(haversine(home_point_y, home_point_x, center_lat, center_lon))
                        plt.scatter([center_lon], [center_lat], marker=poi_shape, s=poi_point_size, c='k', zorder=200)
                else:
                    enricher_pca_dist = "N/A"

                if (0 not in [home_point_x, home_point_y]):
                    x_limit_min.append(home_point_x)
                    x_limit_max.append(home_point_x)
                    y_limit_min.append(home_point_y)
                    y_limit_max.append(home_point_y)
                    plt.scatter(home_point_x, home_point_y, s=poi_point_size, c=home_color, marker=poi_shape,zorder=101)
                if (0 not in [work_point_x, work_point_y]):
                    x_limit_min.append(work_point_x)
                    x_limit_max.append(work_point_x)
                    y_limit_min.append(work_point_y)
                    y_limit_max.append(work_point_y)
                    plt.scatter(work_point_x, work_point_y, s=poi_point_size, c=work_color, marker=poi_shape,zorder=100)
                if (0 not in [home_point_x, home_point_y, work_point_x, work_point_y]):
                    home_office_dist = str(haversine(home_point_y, home_point_x, work_point_y, work_point_x))
                else:
                    home_office_dist = "N/A"
                
                x_padding = 0.25
                y_padding = 0.12
                plot_borders = {
                    "lon": {"min": min(x_limit_min) - x_padding, "max": max(x_limit_max) + x_padding},
                    "lat": {"min": min(y_limit_min) - y_padding, "max": max(y_limit_max) + y_padding}
                }

                plt.xlim([plot_borders['lon']['min'], plot_borders['lon']['max']])
                plt.ylim([plot_borders['lat']['min'], plot_borders['lat']['max']])
                plt.text(
                    x=plot_borders['lon']['min'],
                    y=plot_borders['lat']['min'],
                    s="TIME: "+(datetime.datetime(2023,4,1)+datetime.timedelta(minutes=col*15)).strftime("%H:%M:%S"),
                    fontdict={'size':20, 'color':'darkgreen'},
                    bbox=dict(boxstyle='round', facecolor='#00FF00', alpha=0.69))

                plt.text(
                    x=plot_borders['lon']['min'],
                    y=plot_borders['lat']['max'],
                    s="\n".join(["Home-Office Distance (km): " + home_office_dist, 
                                 "Enricher-PCA Distance (km): " + enricher_pca_dist]),
                    fontdict={'size':20, 'color':'white'},
                    bbox=dict(boxstyle='round', facecolor='#000000', alpha=0.69),
                    horizontalalignment='left'
                )
                plt.savefig(os.path.join("display", "temp", img_name, f"{col}.png"))
                plt.close()

            images = []
            for root_img, folder_img, files_img in os.walk(os.path.join("display", "temp", img_name)):
                for file_img in files_img:
                    images.append([int(file_img.replace(".png", "")), imageio.v3.imread(os.path.join(root_img,file_img))])

            images.sort()
            images = [i[1] for i in images]
            imageio.mimsave(file_path, images, duration=0.1)
            shutil.rmtree(os.path.join("display", "temp", img_name))
        
def weight_plot(weight_data, uuid, note):
    plt.plot()
    sns.scatterplot(data=weight_data,x='x',y='y',hue='label')
    plt.text(
        min(weight_data['x']),
        min(weight_data['y']),
        note,
        fontdict={'size':10, 'color':'red'})
    plt.savefig(os.path.join("display", "footprint", f"{uuid}_weight.png"))
    plt.close()

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Earth's radius in km
    earth_radius = 6371

    # Calculate the distance in km
    distance = earth_radius * c
    return distance
