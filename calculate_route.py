import numpy as np
from manhattan_bresenham import draw_manhattan_line as dml
from manhattan_bresenham import calculate_position_at_time_t
import matplotlib.pyplot as plt
from skyfield.api import load, PlanetaryConstants
import random
import tifffile
import time
from manhattan_bresenham import calculate_line
from datetime import datetime
import json
import rasterio
from pyproj import Transformer, CRS
from datetime import timedelta


is_shadow = []
traversal_time = 180 # 3 minutes to cross 150cm which is the resolution of one cell in the dem
costs = []	

# Load the DEM and horizon map
# Change the paths of these two files to the paths of the DEM and horizon map you want to use
shadowmap = tifffile.imread('C:/Users/user/Downloads/lunar-planner-prototyping/test/cropped_dem_with_nans.tif')
horizonmap = np.load('E:/combined_horizonmap/combined_horizonmap.npy', allow_pickle=True).item()

#Initialize seeds for reproducibility
random.seed(4)
np.random.seed(4)

# Randomly generate points of interests
# Modify this array if you want to assign specific points yourself instead of doing it randomly
# ex: pois2 = [(x,y), (x2,y2), (x3,y3), ....]
pois2 = [(random.randint(750, 4250), random.randint(750, 4250)) for _ in range(10)]

eph = load('de421.bsp')
moon, sun = eph['moon'], eph['sun']
# Create a Time object for the specific time
ts = load.timescale()

mission_start_time = ts.utc(2024, 2, 1, 00, 0)


pc = PlanetaryConstants()
pc.read_text(load('moon_080317.tf'))
pc.read_text(load('pck00008.tpc'))
pc.read_binary(load('moon_pa_de421_1900-2050.bpc'))

frame = pc.build_frame_named('MOON_ME_DE421')


crs = None
xy = None
dataset = None

# keep this the same path as the DEM, used for coordinate to latitute and longitude conversion
dataset = rasterio.open('C:/Users/user/Downloads/lunar-planner-prototyping/test/cropped_dem_with_nans.tif')

# center of DEM cell index to latlon
xy = dataset.xy(dataset.width // 2 , dataset.height // 2) #height, width
crs = CRS(dataset.crs)

transformer = Transformer.from_crs(crs, crs.geodetic_crs)
print(dataset)
def transformToLatLon(x, y):
    """
    Convert raster dataset coordinates to latitude and longitude.

    Parameters:
    x (float): X-coordinate in the dataset's coordinate system.
    y (float): Y-coordinate in the dataset's coordinate system.

    Returns:
    tuple: A tuple (latitude, longitude).
    """

    lon, lat = transformer.transform(x, y)
    return (lat, lon)
output = []
latlon = transformToLatLon(*xy)
aristarchus = moon + pc.build_latlon_degrees(frame, latlon[0], latlon[1])
output.append({'DEM center lat': latlon[0], 'DEM center lon': latlon[1]})



def find_random_closest_point(start, pois, timetamp_at_start, N=4,):
    """
    Finds the closest point among a set of points of interest (POIs) using a path simulation method.

    Parameters:
    start (tuple): The starting coordinates (x, y).
    pois (list): A list of tuples representing the cell indices of points of interest.
    timestamp_at_start (datetime): The starting time for the simulation.
    N (int, optional): The number of closest points to consider. Defaults to 4.

    Returns:
    tuple: A tuple containing the index of the next POI, the cost to reach it, the path to it, and timestamps for each path point.
    """

    lines = [calculate_line(start, poi) for poi in pois]
    #print('length ', sum(len(sub_array) for sub_array in lines) )
    #lines = [dml(start, poi) for poi in pois]

    path_idx = 0


    # create arrays size of lines
    found = np.zeros(len(lines))
    costs = np.zeros(len(lines))
    indices = np.zeros(len(lines))
    path_timestamps = [[] for _ in range(len(lines))]

    id = 0
    time_at_path_point = timetamp_at_start

    while True:



        apparent = aristarchus.at(time_at_path_point).observe(sun).apparent()
        alt, az, distance = apparent.altaz()

        for line_idx in range(len(lines)):
            if not found[line_idx]:

                # path_timestamps[line_idx].append(time_at_path_point)
    
                try:
                    row, col = lines[line_idx][int(indices[line_idx]) + 1] # check target cell
                    shadow = alt.degrees < horizonmap[str(int(az.degrees))][int(row), int(col)]

                    print(f"Debug: At {row}, {col} - Shadow: {shadow}, Current Index: {indices[line_idx]}, cost: {costs[line_idx]} path idx {path_idx}")

                except IndexError as e: # out of line - found
                    found[line_idx] = True
                    continue

                if not shadow: 
                    indices[line_idx] += 1
                    path_timestamps[line_idx].append(time_at_path_point)
                    

                costs[line_idx] += traversal_time # TODO: NB! this is not entirely correct, should be calculated based on distance. Last distance might be different.
                                                        # last linspace point distance to POI may be less than 1

        if np.sum(found) >= N:
            break

        # path_idx += 1
        time_at_path_point = time_at_path_point + timedelta(minutes=5) # add 5 minutes

    next_node_index = np.random.choice(np.where(found==True)[0])
    return next_node_index, costs[next_node_index], lines[next_node_index], path_timestamps[next_node_index]

def find_route_random(start, pois, mission_start_time, N=4):
    """
    Constructs a random route through a set of POIs starting from a given start point.

    Parameters:
    start (tuple): The starting coordinates (x, y).
    pois (list): List of tuples representing the coordinates of points of interest.
    mission_start_time (datetime): The starting time for the mission.
    N (int, optional): Number of iterations to perform in the search. Defaults to 4.

    Returns:
    tuple: Returns the complete route, total cost, paths between POIs, and timestamps for each path.
    """

    route = [start]
    final_cost = 0
    timestaps_of_paths = []
    paths_between_pois = []

    poi_start_time = mission_start_time
    #print('poi type ', type(poi_start_time))

    while len(route) < len(pois):
        not_visited = [poi for poi in pois if poi not in route]
        next_node_index, cost, path_to_poi, timestamps_of_path = find_random_closest_point(route[-1], not_visited, poi_start_time, N=1)#max(int(len(not_visited)/3), 2)) # max necessary to avoid freezing in closest point while loop
        
        timestamp_at_arrival = timestamps_of_path[-1]
        
        # Maybe we want to add some time here to perform some action at the poi
        # time_spent_at_poi = 0
                                                                                                         # N becomes smaller than np.sum(found) and never breaks as not_visited                                                                                                # becomes smaller. 
        route.append(not_visited[next_node_index])
        #print('back to utc time ', timestamp.utc_iso())
        paths_between_pois.append(path_to_poi)
        final_cost += cost
        timestaps_of_paths.append(timestamps_of_path)
        poi_start_time = timestamp_at_arrival # + time_spent_at_poi

        #print('has duplicates ', len(route) != len(set(route)))
    return route, final_cost, paths_between_pois, timestaps_of_paths
        
def find_route_stochastic(start, pois, mission_start_time,  N=4, return_routes=3):
    """
    Generates multiple stochastic routes and selects the best ones based on cost.

    Parameters:
    start (tuple): The starting coordinates (x, y).
    pois (list): List of tuples representing the coordinates of points of interest.
    mission_start_time (datetime): The starting time for the mission.
    N (int): Number of stochastic iterations to perform, defaults to 4.
    return_routes (int): Number of best routes to return, defaults to 3.

    Returns:
    list: A list of the best route(s) (the order of POIs) as tuples sorted by cost containing the paths between POIs, the timestamp for each point in the paths.
    """

    results = []
    for i in range(N): # can pool this
        print(f'{i}. iteration of stochastic route')
        #output.clear()
        route, cost, poi_paths, path_timestamps = find_route_random(start, pois, mission_start_time, N=1)
        results.append((route, cost, poi_paths, path_timestamps))
        #print('timestamps', timestamps)
    sorted_results = sorted(results, key=lambda x: x[1]) 
    return sorted_results[:min([len(pois), return_routes])]


best_cost = float('inf')
best_results = None
best_start = None
start_time = time.time()
for start in pois2:
    results = find_route_stochastic(start,pois2, mission_start_time, N=1,return_routes=1)#find_route_stochastic_pooled(start, pois2, N=3, closest_N=1)
    cost = results[0][1] 
    if cost < best_cost:
        best_cost = cost
        best_results = results
        best_start = start


#best_results = find_route_stochastic((4300,4600),pois2, N=1)
end_time = time.time()
print(f'route calculation took {end_time - start_time} seconds')
print(' best results path ', best_results[0][0], 'weight ', best_results[0][1])#, 'timestamps ', best_results[0][3])
print('route contains all elements in pois array ', set(best_results[0][0]) == set(pois2))
print('length POIs ', len(best_results[0][0]))
print('length points between POIs ', sum(len(sub_array) for sub_array in best_results[0][2]))
print('length timestamps ', sum(len(sub_array) for sub_array in best_results[0][3]) )


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(shadowmap, cmap='gray')

def plot_cycle(only_route):
    n = len(only_route)
    for i in range(n-1):
        
        start = only_route[i]
        end = only_route[(i + 1)]
        line_points = calculate_line(start, end)#dml(start, end)
        x_coords, y_coords = zip(*line_points)
        ax.plot(x_coords, y_coords)
        ax.text(start[0], start[1], str(i), fontsize=12, ha='right', va='bottom')
        ax.plot(start[0], start[1], marker='o')
        #plt.show()
    last_point = only_route[-1]
    ax.text(last_point[0], last_point[1], str(n - 1), fontsize=12, ha='right', va='bottom')
    #ax.plot(last_point[0], last_point[1])

idx = 0
id = 0
print(len(best_results[0][3][0]))
flat_array = np.concatenate(best_results[0][3])
print(len(flat_array))
print(len(best_results[0][2]))

count = 0
for arr in best_results[0][2]:
    # Extract x and y coordinates from each inner array
    
    x_coords = [point[0] for point in arr]
    y_coords = [point[1] for point in arr]
    
    # Plot the points
    ax.scatter(x_coords, y_coords)
    id = 0
    for i in range(len(x_coords)):
        xy = dataset.xy(x_coords[i], y_coords[i])
        latlon = transformToLatLon(*xy)
        # if i == 0:
        #     output.append({'lat': latlon[0], 'lon': latlon[1], 'arrival': best_results[0][3][idx][i].utc_iso(), 'id': 'poi'})
        # else:
        #     output.append({'lat': latlon[0], 'lon': latlon[1], 'arrival': best_results[0][3][idx][i].utc_iso(), 'id': id})
        count += 1
        try:
            output.append({'x': x_coords[i], 'y': y_coords[i], 'arrival': best_results[0][3][idx][i].utc_iso(), 'id': count})
            #ax.text(x_coords[i], y_coords[i], f'{best_results[0][3][idx][i].utc_iso()}', fontsize=8, ha='center', va='bottom')
            id += 1
        except Exception as e:
            print('error ',e)
            print('id', id)
            print('idx ', idx)
    if idx < len(best_results[0][2]):
        idx += 1
        

with open('output.json', 'w') as json_file:
    json.dump(output, json_file, indent=4)

plot_cycle(best_results[0][0])
plt.show()

