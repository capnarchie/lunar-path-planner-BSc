import whitebox
import rasterio
from rasterio.windows import Window
import tifffile
import matplotlib.pyplot as plt
from skyfield.api import load, PlanetaryConstants

# Initialize the WhiteboxTools interface
wbt = whitebox.WhiteboxTools()

# Load the DEM
dem_file = "C:/Users/Uku/Desktop/lunar-planner-prototyping/test/cropped_dem_with_nans.tif"

# eph = load('de421.bsp')
# moon, sun = eph['moon'], eph['sun']
# # Create a Time object for the specific time
# ts = load.timescale()

# t = ts.utc(2024, 2, 1, 00, 0)
# pc = PlanetaryConstants()
# pc.read_text(load('moon_080317.tf'))
# pc.read_text(load('pck00008.tpc'))
# pc.read_binary(load('moon_pa_de421_1900-2050.bpc'))

# frame = pc.build_frame_named('MOON_ME_DE421')

# aristarchus = moon + pc.build_latlon_degrees(frame, 26.244675768670426, 3.219823194482537)

for i in range(0, 361):
    #This loop calculates the horizon angles for a specific location on the Moon's surface at different times
    # Define the output file for horizon angles
    output_file = f"./test/horizonmaps/horizon_angles{i}.tif"
    # Run the horizon angle tool
    wbt.horizon_angle(dem_file, output_file, azimuth=i)

print("Horizon angles calculation complete.")