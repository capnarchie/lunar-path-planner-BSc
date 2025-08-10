# Lunar Planner Prototyping

Research and development work of lunar route planner backend
Testing and development done using Python 3.12

# Lunar route Visualizer

## How to Use

Lunar route planner is a tool that helps you find the optimal route between multiple points on a lunar map. Follow these steps to use it:
## Prerequisites
    To install the prerequisites required to use the scripts in this repository use
    $pip install -r requirements.txt 

If you already have a DEM and a horizonmap you only need to run the application in the test folder
**Run the Application:**

    Execute the calculate_route.py script in terminal to launch the application:

    python3 ./test/calculate_route.py

**Changing the Points of Interest:**

    To change the points of interest locations on the DEM one can do so by changing the cell indices on the POI array in the script calculate_route.py located

**Changing the DEM**

    Swapping the DEM is possible by changing the file path for the DEM that will be loaded into memory in calculate_route.py script

## Generating lookup table for shadow lookup
    To generate the lookuptable it is necessary to create horizon maps and combine them into a lookup table.
    To do this execute the horizon.py script after having given the script the path to your DEM
    After that provide the folder path that horizon.py created to the combine_horizonmaps.py script and execute it.
    This will result in a single lookup table that calculate_route.py will use for shadow lookups

