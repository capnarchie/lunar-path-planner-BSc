import whitebox
import rasterio
from rasterio.windows import Window
import tifffile
import matplotlib.pyplot as plt
from skyfield.api import load, PlanetaryConstants

# Initialize the WhiteboxTools interface
wbt = whitebox.WhiteboxTools()

# Load the DEM, requires absolute path
dem_file = "C:/Users/ukuil/Desktop/lunar-planner-prototyping/test/cropped_dem_with_nans.tif"

for i in range(0, 361,10):
    #This loop calculates the horizon angles for a specific location on the Moon's surface at different times
    # Define the output file for horizon angles, requires absolute path
    output_file = f"C:/Users/ukuil/Desktop/lunar-planner-prototyping/test/horizonmaps/horizon_angles{i}.tif"
    # Run the horizon angle tool
    wbt.horizon_angle(dem_file, output_file, azimuth=i)

print("Horizon angles calculation complete.")