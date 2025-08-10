import numpy as np
import os
import rasterio

# Path to the horizonmaps folder
folder_path = "C:/Users/Uku/Desktop/lunar-planner-prototyping/test/horizonmaps"

# Get a list of all horizonmap files in the folder
horizonmap_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]


#horizonmap_files = horizonmap_files[30] # take 30 files only for testing
# Initialize an empty dictionary to store the horizonmaps
horizon_maps = {}

# Load each horizonmap file and store it in the dictionary
for file in horizonmap_files:
    horizonmap_path = os.path.join(folder_path, file)
    with rasterio.open(horizonmap_path) as src:
        horizonmap = src.read(1)  # Read the first band of the raster
        horizonmap = np.transpose(horizonmap)
    azimuth = file.split("horizon_angles")[1].split(".")[0]  # Extract azimuth from the filename
    #print(azimuth)
    horizon_maps[azimuth] = horizonmap

#horizonmap[azimuth][row, col]
#print(horizon_maps[54][225,4628])
#print(type(horizon_maps))
print(horizon_maps.keys())
#print(horizon_maps[54][4628,225])
# Save the horizon_maps dictionary to a file
output_file = "C:/Users/Uku/Desktop/lunar-planner-prototyping/test/combined_horizonmap/combined_horizonmap.npy"
np.save(output_file, horizon_maps)

# Example query on the combined horizonmap
# in_shadow = sun_angle < combined_horizonmap[azimuth][row, col]