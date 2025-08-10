# This is used to generate shadow maps for the dem at specific times using ray tracing
# UNFINISHED, do not use! refer to horizon.py for a working example

import tifffile
import numpy as np
#import cupy as cp
from skyfield.api import load, PlanetaryConstants
import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer, CRS
import numba
from numba import cuda
import math
import time
import os
import datetime
# cuda_driver_path = os.getenv('CUDA_PATH')
# print(cuda_driver_path)
# print(numba.cuda.detect())
# Change path for different DEM
tif_img = tifffile.imread("lunar-planner-prototyping\data\LRO_NAC_DEM_Apollo_15_26N004E_150cmp.tif")
#print(tif_img)
# Mask nodata values to fix binary apollo 15 DEM displaying
nodata_mask = tif_img == -3.4028226550889045e+38
tif_img[nodata_mask] = np.nan
# #print(np.max(nodata_mask))
# plt.figure(figsize=(10, 5))
# plt.imshow(tif_img, cmap='gray')

# plt.colorbar()
# plt.show()
# exit()
crs = None
xy = None
dataset = None
with rasterio.open("lunar-planner-prototyping\data\LRO_NAC_DEM_Apollo_15_26N004E_150cmp.tif") as dataset:
    #print(dataset.height, dataset.width)
    print(dataset.xy(dataset.height, dataset.width))
    xy = dataset.xy(10500 , 3500) #height, width
    #xy = dataset.xy(dataset.height//2, dataset.width//2) #height, width
    # print("Additional Metadata:")
    # for key, value in dataset.meta.items():
    #     print(f"  {key}: {value}")
    crs = CRS(dataset.crs)

transformer = Transformer.from_crs(crs, crs.geodetic_crs)
print(dataset)
def transformToLatLon(x, y):
    
    lon, lat = transformer.transform(x, y)
    return (lat, lon)

#print("lat lon is: ", transformToLatLon(*xy))
latlon = transformToLatLon(*xy)
print(latlon)
exit()
# Define the region you want to display (start, top-left corner end bottom right corner)
start_row, end_row = 8000, 13001  # Define the row range #8000 13001 #8000, 19001
start_col, end_col = 1000, 6001  # Define the column range #1000, 6001 #1000, 12001

# Crop the image to the specified region
elevation_matrix = tif_img[start_row:end_row, start_col:end_col]
elevation_matrix_contiguous = np.ascontiguousarray(elevation_matrix)


# Load the ephemeris data for the Moon and the Sun
eph = load('de421.bsp')
moon, sun = eph['moon'], eph['sun']
# Create a Time object for the specific time
ts = load.timescale()

t = ts.utc(2024, 2, 1, 00, 0)
print("Shadow map calculated for:", t.utc_iso())


pc = PlanetaryConstants()
pc.read_text(load('moon_080317.tf'))
pc.read_text(load('pck00008.tpc'))
pc.read_binary(load('moon_pa_de421_1900-2050.bpc'))

frame = pc.build_frame_named('MOON_ME_DE421')

aristarchus = moon + pc.build_latlon_degrees(frame, latlon[0], latlon[1])

# What's the position of Sun, viewed from a latlon on the Moon?

apparent = aristarchus.at(t).observe(sun).apparent()
alt, az, distance = apparent.altaz()
print(alt.degrees, 'degrees above the horizon')
print(az.degrees, 'degrees around the horizon from north')

sun_angle = np.array(apparent.position.au)
sun_angle = sun_angle / np.linalg.norm(sun_angle)
#print("sun direction vector ", sun_angle)

# convert spherical to cartesian coordinates
x = math.cos(alt.radians) * math.cos(az.radians)
y = math.cos(alt.radians) * math.sin(az.radians)
z = math.sin(alt.radians)
sun_angle_vector = np.array([x,y,z])
sun_angle_vector /= np.linalg.norm(sun_angle_vector)
#print("sun angle vect ", sun_angle_vector)
shadow_map = np.zeros_like(elevation_matrix)

angle_map = np.zeros_like(elevation_matrix)
alt_radians = alt.radians
ray_x_list = []
ray_y_list = []

@cuda.jit
def calculate_shadow_map(elevation_matrix, sun_angle, shadow_map, angle_map):
    row, col = cuda.grid(2)

    if row < elevation_matrix.shape[0] and col < elevation_matrix.shape[1]:
        x, y = row, col
        z = elevation_matrix[x, y]
      
        for step in range(1, 5200):  # keep this at least the size of the end_row - start_row size (test)
            # Calculate the coordinates along the ray
            ray_x_rel, ray_y_rel = step * sun_angle[0], step * sun_angle[1]
            ray_x, ray_y = x + ray_x_rel, y +  ray_y_rel
            ray_triangle_z = z+step * sun_angle[2]#z+math.sqrt((ray_x_rel**2 + ray_y_rel**2)) * math.tan(alt_radians)

            # Check if the ray coordinates are within the bounds of the elevation matrix
            if 0 <= ray_x < elevation_matrix.shape[0] and 0 <= ray_y < elevation_matrix.shape[1]:
                # Check if the elevation along the ray is lower than the current cell's elevation
                if (ray_triangle_z) > (elevation_matrix[int(ray_x), int(ray_y)]):
                    # ray elevation at this cell lower than origin, keep checking until out of matrix
                    continue
                    
                else:
                    # we in shadow
                    shadow_map[x, y] = 0
                    return
            else:
                # ray reached out of the matrix dimensions, so it is unobstructed therefore the pixel is illuminated
                shadow_map[x,y] = 1

#Copy to GPU
elevation_matrix_gpu = cuda.to_device(elevation_matrix_contiguous)
shadow_map_gpu = cuda.device_array_like(shadow_map)
angle_map_gpu = cuda.device_array_like(angle_map)
# Invoke CUDA kernel
# https://numba.pydata.org/numba-doc/latest/cuda/kernels.html
threadsperblock = (16, 16)
blockspergrid_x = (elevation_matrix.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (elevation_matrix.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

start_time = time.time()
#calculate_shadow_map[blockspergrid, threadsperblock](elevation_matrix_gpu, sun_angle_vector, shadow_map_gpu, angle_map_gpu)
# Transfer result back to CPU
end = time.time()
#print(end - start_time , "seconds")
shadow_map = shadow_map_gpu.copy_to_host()

# CPU
def is_illuminated(x, y, z):
    # Check if the point is illuminated based on the change in elevation along the ray
    
    for step in range(1, 5000):  
        # Calculate the coordinates along the ray
        # Actual ray length
        ray_x_rel, ray_y_rel = step * sun_angle_vector[0], step * sun_angle_vector[1]
        # Ray length from x,y coordinate
        ray_x, ray_y = x + ray_x_rel, y +  ray_y_rel
        
        ray_x_list.append(ray_x)
        ray_y_list.append(ray_y)
        continue
        
        # ray_triangle_z = z + np.sqrt(ray_x_rel**2 + ray_y_rel**2) * math.tan(alt.radians)
        # #print("ray coords: ", ray_x, ray_y)
        # #print("ray vector ", np.sqrt(ray_x_rel**2 + ray_y_rel**2))
        # #print("z: ", z, " ray_z ", ray_triangle_z)
        # # Check if the ray coordinates are within the bounds of the elevation matrix
        # if 0 <= ray_x < elevation_matrix.shape[0] and 0 <= ray_y < elevation_matrix.shape[1]:
        #     # Check if the elevation along the ray is lower than the current pixel's elevation
        #     if (ray_triangle_z) > (elevation_matrix[int(ray_x), int(ray_y)]):
        #         # test the current pixel of the ray height if it is in shadow, if it isnt continue the loop
        #         continue
        #     else:
        #         #print("here")
        #         return 0
        # else:
        #     # If the ray goes beyond the bounds, consider it as illuminated
        #     return 1
    

#is_illuminated(2500,2500, elevation_matrix[2500,2500])

# fig, axs = plt.subplots(1, 2, figsize=(10, 6))
#plt.plot(ray_x_list, ray_y_list, 'r-')

# CPU
# # Iterate over the pixels in the elevation matrix
# for row in range(elevation_matrix.shape[0]):
#     for col in range(elevation_matrix.shape[1]):
#         # Check if the point is illuminated or in shadow
#         shadow_map[row, col] = is_illuminated(row, col, elevation_matrix[row, col])

# Plot the results
# im1= axs[0].imshow(elevation_matrix, cmap="gray")
# axs[0].set_title('Original Elevation Matrix')
# cbar1 = fig.colorbar(im1, ax=axs[0], label='Elevation')
# axs[1].imshow(shadow_map, cmap='gray')
# axs[1].set_title('Shadow Map ' + t.utc_iso())
# plt.tight_layout()
#plt.show()

for i in range(361):
    apparent = aristarchus.at(t).observe(sun).apparent()
    alt, az, distance = apparent.altaz()
    # print(alt.degrees, 'degrees above the horizon')
    # print(az, 'degrees around the horizon from north')
    print("calculating shadowmap for time>> ", t.utc_iso())
    x = math.cos(alt.radians) * math.cos(az.radians)
    y = math.cos(alt.radians) * math.sin(az.radians)
    z = math.sin(alt.radians)
    sun_angle_vector = np.array([x,y,z])
    sun_angle_vector /= np.linalg.norm(sun_angle_vector)
    #print("sun angle vect ", sun_angle_vector)
    shadow_map = np.zeros_like(elevation_matrix, dtype=np.uint8)
    
    shadow_map_gpu = cuda.device_array_like(shadow_map)
    calculate_shadow_map[blockspergrid, threadsperblock](elevation_matrix_gpu, sun_angle_vector, shadow_map_gpu, angle_map_gpu)
    shadow_map = shadow_map_gpu.copy_to_host()
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    # im1= axs[0].imshow(elevation_matrix, cmap="gray")
    # axs[0].set_title('Original Elevation Matrix')
    # cbar1 = fig.colorbar(im1, ax=axs[0], label='Elevation')
    axs.imshow(shadow_map, cmap='gray')
    axs.set_title('Shadow Map ' + t.utc_iso())
    cur_time = str(t.utc_iso())
    cur_time = cur_time.replace(':', '-')
    plt.tight_layout()
    #np.save(f"C:/Users/user/Downloads/lunar-planner-prototyping/planner_proto/shadowmaps/{cur_time}.npy", shadow_map)
    plt.savefig(f"C:/Users/user/Downloads/lunar-planner-prototyping/planner_proto/shadowmaps/{cur_time}.png")
    plt.close()
    t += 0.25/3
