import tifffile
import numpy as np
import networkx as nx
from planner_proto.utils.visualization import visualize_graph
from planner_proto.utils.visualization import visualize_quadtree_graph
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Delaunay

# Load the TIFF image
tif_img = tifffile.imread("lunar-planner-prototyping\data\WAC_GLD100_E300N0450_100M.tif")

# Define the region you want to display (e.g., top-left corner)
start_row, end_row = 13000, 16001  # Define the row range
start_col, end_col = 13000, 16001  # Define the column range

# Calculate the size of the region
region_height = end_row - start_row
region_width = end_col - start_col
print(region_width)

# Crop the image to the specified region
cropped_img = tif_img[start_row:end_row, start_col:end_col]

# Calculate cost grid
min_value = np.min(cropped_img)
max_value = np.max(cropped_img)
baseline = np.percentile(cropped_img, 50)  # 50th percentile (median)

cost_grid = np.abs(cropped_img - baseline)

# These two values are random-ish right now but based off of the max and min value of the cost grid
high_cost_threshold = 110
low_cost_threshold = 50
downsample_factor = 60 # ensure modulo = 0 with map width and height also ensure modulo = 0 with with last index of the array
                        # e.g shape 3000x3000 does not include 3000 index as last element but 2999 so 2999 % 375 != 0
                        #                                                      you would need shape 3001x3001 instead TODO?
low_cost_value = 10
#cost_grid[cost_grid < low_cost_threshold] = low_cost_value

# Set values above high_cost_threshold to a high cost value
high_cost_value = 100
#cost_grid[cost_grid > high_cost_threshold] = high_cost_value                        
downsampled_cost_grid = cost_grid[::downsample_factor, ::downsample_factor]

print("downsampled shape", downsampled_cost_grid)

# Create a graph
G = nx.Graph()

# Define the resolution and speed of the robot
resolution = 150 # cm per pixel
speed = 5 # cm per second

# Define the origin node
origin = (0, 0)
distance = 150
nodes_visited = 0
initial_battery = 100
total_time = 0

battery_weight = 5
time_weight = 2
elevation_weight = 2
# for i in range(0, (end_row - start_row), downsample_factor):
#     for j in range(0, (end_col - start_col), downsample_factor):
#         node = (i, j)
#         cost = downsampled_cost_grid[i // downsample_factor, j // downsample_factor]
#         G.add_node(node, cost=cost)

#         if i > 0:
#             G.add_edge(node, (i - downsample_factor, j), weight=cost)
#         if i < end_row - start_row - downsample_factor:
#             G.add_edge(node, (i + downsample_factor, j), weight=cost)
#         if j > 0:
#             G.add_edge(node, (i, j - downsample_factor), weight=cost)
#         if j < end_col - start_col - downsample_factor:
#             G.add_edge(node, (i, j + downsample_factor), weight=cost)

print("min", min_value)
print("max", max_value)
print("baseline", baseline)

# Visualize the graph
fig, ax = visualize_graph(G, cropped_img, start_row, start_col, end_row, end_col, downsample_factor)

class QuadTree:
    def __init__(self, boundary, capacity):
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.subdivided = False
        self.quadrants = [None] * 4
        self.centers = [None] * 4
        self.elevation_limit = 10
        self.elevation = 0


    def subdivide(self):
        x, y, w, h = self.boundary
        cx, cy = x + w / 2, y + h / 2  # Center of the current quadrant

        nw = (x, y + h / 2, w / 2, h / 2)
        ne = (x + w / 2, y + h / 2, w / 2, h / 2)
        sw = (x, y, w / 2, h / 2)
        se = (x + w / 2, y, w / 2, h / 2)

        self.quadrants[0] = QuadTree(nw, self.capacity)
        self.quadrants[1] = QuadTree(ne, self.capacity)
        self.quadrants[2] = QuadTree(sw, self.capacity)
        self.quadrants[3] = QuadTree(se, self.capacity)

        self.centers[0] = (cx - w / 4, cy + h / 4)  # Center of NW quadrant
        self.centers[1] = (cx + w / 4, cy + h / 4)  # Center of NE quadrant
        self.centers[2] = (cx - w / 4, cy - h / 4)  # Center of SW quadrant
        self.centers[3] = (cx + w / 4, cy - h / 4)  # Center of SE quadrant
        
        #print("quadrant elevation------> ", self.elevation)
        self.subdivided = True


    def calculate_average_elevation(self):

        elevations = [
            downsampled_cost_grid[x // downsample_factor, y // downsample_factor]
            for x, y in self.points
        ]

        min_elevation = min(elevations)
        max_elevation = max(elevations)

        elevation_difference = max_elevation - min_elevation
        self.elevation = elevation_difference
        if self.subdivided:
            for quadrant in self.quadrants:
                if quadrant is not None:
                    quadrant.calculate_average_elevation()


    def insert(self, point):
        if not self.boundary_contains_point(point):
            return False
        
        self.points.append(point)

        if self.elevation < self.elevation_limit:
            self.calculate_average_elevation()
            return True
        else:
            if not self.subdivided:
                self.subdivide()
            
            for quadrant in self.quadrants:
                if quadrant.insert(point):
                    return True
                
    def boundary_contains_point(self, point):
        x, y, w, h = self.boundary
        px, py = point
        
        return x <= px <= x + w and y <= py <= y + h
    
def draw_quadtree_centers(ax, quadtree):
    
    if quadtree.centers[0] is not None:
        ax.plot(quadtree.centers[0][1], quadtree.centers[0][0], marker='o', markersize=0.5, color='red')  # NW quadrant center
    if quadtree.centers[1] is not None:
        ax.plot(quadtree.centers[1][1], quadtree.centers[1][0], marker='o', markersize=0.5, color='red')  # NE quadrant center
    if quadtree.centers[2] is not None:
        ax.plot(quadtree.centers[2][1], quadtree.centers[2][0], marker='o', markersize=0.5, color='red')  # SW quadrant center
    if quadtree.centers[3] is not None:
        ax.plot(quadtree.centers[3][1], quadtree.centers[3][0], marker='o', markersize=0.5, color='red')  # SE quadrant center

    if quadtree.subdivided:
        for quadrant in quadtree.quadrants:
            if quadrant is not None:
                draw_quadtree_centers(ax, quadrant)

def draw_quadtree(ax, quadtree, depth=0):
    if quadtree.subdivided:
        for quadrant in quadtree.quadrants:
            draw_quadtree(ax, quadrant, depth + 1)

    x, y, w, h = quadtree.boundary
    rect = patches.Rectangle((y, x), w, h, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

def draw_graph_and_quadtree(ax, graph, quadtree):

    draw_quadtree(ax, quadtree)
    draw_quadtree_centers(ax, quadtree)
    
# Create a QuadTree
boundary = (0, 0, end_row-start_row-1, end_col - start_col-1)
capacity = 3
quadtree = QuadTree(boundary, capacity)

#Insert nodes of the graph into the QuadTree
for i in range(0, (end_row - start_row), downsample_factor):
    for j in range(0, (end_col - start_col), downsample_factor):
        quadtree.insert((i,j))
# for node in G.nodes:
#     quadtree.insert(node)


center_graph = nx.Graph()

#fig1, ax1 = plt.subplots(figsize=(8, 8))
# Draw the graph and the quadtree on the same axis
draw_graph_and_quadtree(ax, G, quadtree)
pos = {}
collected_nodes = []
def recursive_traversal(quadtree):
    #non_none_centers = [center for center in quadtree.centers if center is not None]
    non_none_centers = [(center[1], center[0]) for center in quadtree.centers if center is not None]
    for center in non_none_centers:
        center_graph.add_node(center)
        #what the flip numpy and matplotlib NB! switch x and y places for visualizing
        pos[center] = (center[0], center[1])
        collected_nodes.append(center)

    if quadtree.subdivided:
        for quadrant in quadtree.quadrants:
            if quadrant is not None:
                recursive_traversal(quadrant)
                
recursive_traversal(quadtree)


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
# Convert collected_nodes to numpy array for Delaunay triangulation
nodes_array = np.array(collected_nodes)
#print("nodes", center_graph.nodes, len(collected_nodes))
# Delaunay triangulation using scipy.spatial.Delaunay
tri = Delaunay(nodes_array)
# Add edges from Delaunay triangulation to the graph
#print("delaunay ", tuple(nodes_array[tri.simplices[0]]))
#print("len ", len(tri.simplices[0]))
for simplex in tri.simplices:
    # Each simplex is a set of 3 nodes representing a triangle
    # In this situation a simplex is made of 3 node indexes that are from graph nodes(tuple) array
    for i in range(len(simplex)):
        # Loop over all 3 simplex points (nodes)
        for j in range(i + 1, len(simplex)):
            # Extract the coordinates of the two nodes from the simplex
            node_i = tuple(nodes_array[simplex[i]])
            node_j = tuple(nodes_array[simplex[j]])
            # Add edge between them
            center_graph.add_edge(node_i, node_j, weight=2)


fig1, ax1 = plt.subplots(figsize=(8, 8))
#print("new list ",inverted_center_graph)
#print("pos nodes ", pos, len(pos))
# if set(center_graph.nodes) == set(pos):
#     print("equal")
nx.draw(center_graph, pos, with_labels=False, node_size=2.5, font_size=0)
edge_labels = nx.get_edge_attributes(center_graph, 'weight')
#nx.draw_networkx_edge_labels(center_graph, pos, edge_labels=edge_labels, font_color='red', font_size=7)
# ax1.invert_yaxis()
# ax1.invert_xaxis()
visualize_quadtree_graph(center_graph, fig1, ax1)
#fig1, ax1 = visualize_graph(center_graph, cropped_img, start_row, start_col, end_row, end_col, downsample_factor)
plt.show()

