import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, PlanetaryConstants
import random
import time
from scipy.spatial import distance
from datetime import datetime, timedelta
import json
import rasterio
from pyproj import Transformer, CRS
from skyfield import almanac
import multiprocessing # <-- IMPORT MULTIPROCESSING
from functools import partial # <-- To help pass arguments to the worker

# --- VERBOSE LOGGING SETUP ---
def log_debug(message, indent_level=0):
    """Prints a message with indentation for hierarchical logging."""
    pass
    #indent = '    ' * indent_level
    #print(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} {indent}{message}")

# --- Simple Line Calculation Function ---
def calculate_line(start, end):
    arr_points = []
    for i in range(1):
        arr_points.append((start[0], start[1]))
        for j in range(i+1, 2):
            #start, end = points[i], points[j]
            dist = distance.euclidean(start, end)
            segments = int(dist)
            dx = (end[0] - start[0]) / dist
            dy = (end[1] - start[1]) / dist
            # Generate linspace points with a consistent distance of 1 between consecutive points
            for k in range(segments):
                point1 = (start[0] + k * dx, start[1] + k * dy)
                point2 = (start[0] + (k + 1) * dx, start[1] + (k + 1) * dy)
                
                dist = distance.euclidean(point1, point2)
                #print(point2)
                #print(f"Distance between point {k+1} and point {k+2}: {dist:.2f}")
                arr_points.append((point2[0], point2[1]))
                #arr_points.append((point2[0], point2[1]))
            arr_points.append((end[0], end[1]))
    #print(len(arr_points))
    return np.array(arr_points)

# --- Configuration and Constants ---
TIME_PER_CELL = timedelta(seconds=180)

# --- Setup: Data Loading and Skyfield Initialization ---
print("Loading data and initializing Skyfield...")

print("Loading 35GB horizon map dictionary... (This may take a moment)")
horizon_maps = np.load('./test/combined_horizonmap/combined_horizonmap.npy', allow_pickle=True).item()
print("Horizon map dictionary loaded successfully.")

with rasterio.open('./test/cropped_dem_with_nans.tif') as dem_dataset:
    dem_for_plotting = dem_dataset.read(1)

random.seed(4)
np.random.seed(4)

eph = load('de421.bsp')
moon, sun = eph['moon'], eph['sun']
ts = load.timescale()
pc = PlanetaryConstants()
pc.read_text(load('moon_080317.tf'))
pc.read_text(load('pck00008.tpc'))
pc.read_binary(load('moon_pa_de421_1900-2050.bpc'))
frame = pc.build_frame_named('MOON_ME_DE421')

# --- Coordinate Transformation Setup ---
dataset = rasterio.open('./test/cropped_dem_with_nans.tif')
dem_center_xy = dataset.xy(dataset.height // 2, dataset.width // 2)
crs = CRS(dataset.crs)
transformer_to_latlon = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)

dem_center_lon, dem_center_lat = transformer_to_latlon.transform(*dem_center_xy)
observer_location = moon + pc.build_latlon_degrees(frame, dem_center_lat, dem_center_lon)
print(f"DEM Center Latitude: {dem_center_lat}, Longitude: {dem_center_lon}")

# --- Point of Interest (POI) Generation ---
pois = [(random.randint(750, 4250), random.randint(750, 4250)) for _ in range(10)]
mission_start_time = ts.utc(2024, 2, 1, 0, 0)

# --- ROBUST CORE FUNCTIONS ---

def calculate_wait_time(point_coords, current_time, observer, sun_obj, horizon_data, indent=0):
    log_debug(f"-> Entr CWT for point {np.round(point_coords, 1)} at time {current_time.utc_strftime('%Y-%m-%d %H:%M')}", indent)
    row, col = int(point_coords[1]), int(point_coords[0])

    alt_now, az_now, _ = observer.at(current_time).observe(sun_obj).apparent().altaz()
    if alt_now.degrees > 0:
        az_key = str(int(round(az_now.degrees / 10.0)) * 10 % 360)
        if az_key in horizon_data:
            required_altitude = horizon_data[az_key][row, col]
            if alt_now.degrees >= required_altitude:
                log_debug(f"   Point is illuminated. Wait is 0.", indent)
                return timedelta(seconds=0)

    log_debug(f"   Point in shadow. Starting robust search...", indent)
    
    coarse_step = timedelta(hours=6)
    time_cursor = current_time
    log_debug(f"   Starting coarse search with {coarse_step.total_seconds()/3600}hr steps...", indent)
    for i in range(200):
        alt_cursor, _, _ = observer.at(time_cursor).observe(sun_obj).apparent().altaz()
        if alt_cursor.degrees > 0:
            log_debug(f"   Coarse search found sunrise window around {time_cursor.utc_strftime('%Y-%m-%d %H:%M')}", indent)
            break
        time_cursor += coarse_step
    else:
        log_debug(f"   Coarse search failed to find sunrise. Returning huge wait.", indent)
        return timedelta(days=999)

    log_debug(f"   Refining illumination time with binary search...", indent)
    low_bound = time_cursor - coarse_step
    high_bound = time_cursor

    for _ in range(25):
        mid_point = low_bound + (high_bound - low_bound) / 2
        mid_alt, mid_az, _ = observer.at(mid_point).observe(sun_obj).apparent().altaz()
        
        is_lit = False
        if mid_alt.degrees > 0:
            az_key = str(int(round(mid_az.degrees / 10.0)) * 10 % 360)
            required_altitude = horizon_data[az_key][row, col]
            if mid_alt.degrees >= required_altitude:
                is_lit = True

        if is_lit:
            high_bound = mid_point
        else:
            low_bound = mid_point

    # --- BUG FIX AREA ---
    # The result of subtracting two Time objects is a float (in days).
    # We must convert it to a timedelta object.
    wait_duration_days = high_bound - current_time
    wait_duration_timedelta = timedelta(days=wait_duration_days)
    
    log_debug(f"   Binary search complete. Total wait: {wait_duration_timedelta}", indent)
    return wait_duration_timedelta

def get_path_cost_and_data(start_poi, end_poi, departure_time, indent=0):
    log_debug(f"-> Eval path from {start_poi} to {end_poi}", indent)
    path_points = calculate_line(start_poi, end_poi)
    path_timestamps = []
    accumulated_wait_time = timedelta(seconds=0)
    check_interval = max(1, len(path_points) // 10)

    for i in range(len(path_points)):
        point = path_points[i]
        travel_time_to_point = i * TIME_PER_CELL
        arrival_time_at_point = departure_time + travel_time_to_point + accumulated_wait_time
        if i % check_interval == 0 or i == len(path_points) - 1:
            log_debug(f"   Chkpt {i}/{len(path_points)-1} at {arrival_time_at_point.utc_strftime('%Y-%m-%d %H:%M')}", indent)
            wait_at_point = calculate_wait_time(point, arrival_time_at_point, observer_location, sun, horizon_maps, indent=indent+1)
            accumulated_wait_time += wait_at_point
        final_time_at_point = departure_time + travel_time_to_point + accumulated_wait_time
        path_timestamps.append(final_time_at_point)
    
    base_travel_time = (len(path_points) -1) * TIME_PER_CELL
    total_path_duration = base_travel_time + accumulated_wait_time
    log_debug(f"   Path cost: {total_path_duration}", indent)
    return total_path_duration.total_seconds(), departure_time + total_path_duration, path_points, path_timestamps

# The rest of the script is unchanged and should now work correctly.

def find_route_random(start_poi, all_pois, start_time, indent=0):
    log_debug(f"-> Building route from {start_poi}", indent)
    route, total_cost_seconds, all_paths, all_timestamps = [start_poi], 0, [], []
    current_location, current_time = start_poi, start_time

    while len(route) < len(all_pois):
        log_debug(f"   [Loop] Visited {len(route)}/{len(all_pois)}. Time: {current_time.utc_strftime('%Y-%m-%d')}", indent)
        not_visited = [p for p in all_pois if p not in route]
        candidate_results = []
        for cand_poi in not_visited:
            cost, arr, path, time_data = get_path_cost_and_data(current_location, cand_poi, current_time, indent=indent+1)
            candidate_results.append({'poi': cand_poi, 'cost': cost, 'arrival': arr, 'path': path, 'timestamps': time_data})
        
        if not candidate_results: 
            log_debug("   No viable candidates!", indent)
            break
        best_candidate = min(candidate_results, key=lambda x: x['cost'])
        log_debug(f"   Best next is {best_candidate['poi']} with cost {timedelta(seconds=best_candidate['cost'])}", indent)
        
        current_location, current_time = best_candidate['poi'], best_candidate['arrival']
        route.append(best_candidate['poi'])
        total_cost_seconds += best_candidate['cost']
        all_paths.append(best_candidate['path'])
        all_timestamps.append(best_candidate['timestamps'])
    
    log_debug(f"   Finished route. Total duration: {timedelta(seconds=total_cost_seconds)}", indent)
    return route, total_cost_seconds, all_paths, all_timestamps

def find_route_stochastic_sequential(all_pois, start_time):
    best_overall_cost, best_overall_result = float('inf'), None
    for i, start_poi in enumerate(all_pois):
        print(f"\n===== CALCULATING FULL ROUTE STARTING FROM POI {i}: {start_poi} =====")
        result = find_route_random(start_poi, all_pois, start_time, indent=1)
        if result and result[1] < best_overall_cost:
            log_debug(f"New best route found with cost {timedelta(seconds=result[1])}", 0)
            best_overall_cost, best_overall_result = result[1], result
    return [best_overall_result]

def build_cost_matrix(pois, departure_time):
    """
    Pre-calculates the travel duration between all pairs of POIs.
    This is the most time-consuming part, but it's done only once.
    """
    n = len(pois)
    cost_matrix = np.full((n, n), np.inf)
    
    print("Building cost matrix... (This may take a while for many POIs)")
    for i in range(n):
        for j in range(n):
            if i == j:
                cost_matrix[i, j] = 0
                continue
            
            # We use a fixed departure time for this approximation.
            # This is the key simplification that makes this approach fast.
            cost_seconds, _, _, _ = get_path_cost_and_data(pois[i], pois[j], departure_time)
            cost_matrix[i, j] = cost_seconds
            print(f"  Cost from POI {i} to {j}: {timedelta(seconds=cost_seconds)}")
            
    return cost_matrix

def calculate_tour_cost(tour, cost_matrix):
    """Calculates the total cost of a tour using the pre-computed matrix."""
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += cost_matrix[tour[i], tour[i+1]]
    # Add cost to return to start if it's a round trip, otherwise not.
    # For this problem, it's a path, not a cycle.
    return total_cost

def solve_tsp_2opt(cost_matrix):
    """
    Finds a good solution to the TSP using the 2-opt heuristic.
    This is extremely fast because it only uses matrix lookups.
    """
    num_pois = len(cost_matrix)
    # Start with an initial random tour
    current_tour = list(range(num_pois))
    random.shuffle(current_tour)
    
    improvement = True
    while improvement:
        improvement = False
        best_cost = calculate_tour_cost(current_tour, cost_matrix)
        
        for i in range(1, num_pois - 2):
            for j in range(i + 1, num_pois):
                # Create a new tour by reversing the segment between i and j
                new_tour = current_tour[:i] + current_tour[i:j][::-1] + current_tour[j:]
                new_cost = calculate_tour_cost(new_tour, cost_matrix)
                
                if new_cost < best_cost:
                    current_tour = new_tour
                    best_cost = new_cost
                    improvement = True
                    # Break to restart the loops with the new improved tour
                    break
            if improvement:
                break
                
    return current_tour, best_cost

def refine_final_route(poi_objects, best_order_indices, start_time):
    """
    Takes the best order from 2-opt and does one final, precise calculation
    to get the true time-dependent costs and timestamps.
    """
    print("Refining final route with precise time-dependent calculations...")
    final_route_pois = [poi_objects[i] for i in best_order_indices]
    
    total_cost_seconds = 0
    all_paths = []
    all_timestamps = []
    current_time = start_time

    for i in range(len(final_route_pois) - 1):
        start_poi = final_route_pois[i]
        end_poi = final_route_pois[i+1]
        
        cost, arrival, path_data, time_data = get_path_cost_and_data(start_poi, end_poi, current_time)
        
        current_time = arrival
        total_cost_seconds += cost
        all_paths.append(path_data)
        all_timestamps.append(time_data)
        
    return final_route_pois, total_cost_seconds, all_paths, all_timestamps

# --- Main Execution Logic (Now uses the new architecture) ---
if __name__ == '__main__':
    start_exec_time = time.time()
    
    # STAGE 1: Build the approximate cost matrix
    # Using mission_start_time as the typical departure time for all segments
    cost_matrix = build_cost_matrix(pois, mission_start_time)
    
    # STAGE 2: Run the fast 2-opt solver on the matrix
    print("\nSolving TSP with 2-opt heuristic...")
    best_order, estimated_cost = solve_tsp_2opt(cost_matrix)
    print(f"2-opt found best order: {best_order} with estimated cost: {timedelta(seconds=estimated_cost)}")
    
    # STAGE 3: Refine the route to get exact times
    final_route, final_cost, final_paths, final_timestamps = refine_final_route(pois, best_order, mission_start_time)
    
    end_exec_time = time.time()
    
    best_results = (final_route, final_cost, final_paths, final_timestamps)

    # ... (The rest of the script for plotting and JSON output remains the same) ...
    # It will now receive the 'best_results' tuple in the correct format.
    print(f'\n--- Route Calculation Complete ---')
    print(f'Total execution time: {end_exec_time - start_exec_time:.2f} seconds')
    print(f'Best starting POI: {best_results[0][0]}')
    print(f'Total mission duration (cost): {timedelta(seconds=best_results[1])}')
    print(f'Route visits all POIs: {len(set(best_results[0])) == len(pois)}')

    # --- Plotting and Output Generation ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(dem_for_plotting, cmap='gray')
    ax.set_title("Optimal Mission Route")

    route_pois, paths = best_results[0], best_results[2]
    for i, poi_node in enumerate(route_pois):
        ax.plot(poi_node[0], poi_node[1], 'o', markersize=8, color='red')
        ax.text(poi_node[0] + 15, poi_node[1] + 15, str(i), color='white', fontsize=14, fontweight='bold')
        if i > 0:
            x_coords, y_coords = zip(*paths[i-1])
            ax.plot(x_coords, y_coords, alpha=0.8, color='cyan')
    
    output_data = []
    waypoint_id = 0
    all_path_timestamps = best_results[3]
    for i in range(len(paths)):
        for j in range(len(paths[i])):
            point, timestamp = paths[i][j], all_path_timestamps[i][j]
            lon, lat = transformer_to_latlon.transform(*dataset.xy(point[1], point[0]))
            output_data.append({'name': waypoint_id, 'latitude': lat, 'longitude': lon, 'arrival_utc': timestamp.utc_iso()})
            waypoint_id += 1
    
    with open('mission_plan_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    print("Generated mission_plan_output.json")

    plt.show()