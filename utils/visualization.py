import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib.widgets import Slider

# # Create a figure for the slider
# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.25)

# # Initial variable value
# initial_value = 0.5

# # Add a slider for changing the variable value
# ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# slider = Slider(ax_slider, 'Value', 0.1, 1.0, valinit=initial_value)



start_node = None
end_node = None
path = None
initial_battery = 100 # percentage
speed = 5
resolution = 150
total_time = 0
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance_pixels = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    distance_meters = distance_pixels * 1.5
    return distance_meters

def visualize_graph(graph, image, start_row, start_col, end_row, end_col, downsample_factor):
    global start_node, end_node, path, initial_battery, total_time
    # Create a larger figure and axis for the grayscale image
    fig, ax = plt.subplots(figsize=(8, 8))
    max_cost = np.max(image)

    # Scale the actual image to fit the available space
    img = ax.imshow(image, cmap='viridis', origin='lower')
    # Add a colorbar
    #cbar = plt.colorbar(img)

# Set a label for the colorbar
    #cbar.set_label('Elevation')
    #ax.axis('on')

    # # Draw the grid and label each node with coordinates and cost
    # for i in range(0, end_row - start_row, downsample_factor):
    #     for j in range(0, end_col - start_col, downsample_factor):
    #         ax.plot(j, i, 'ro', markersize=6)
    #         cost = graph.nodes[(i, j)]['cost']
    #         ax.text(j, i - 20, f'Cost: {cost:.2f}', fontsize=6, color='black', ha='left', va='top')

    def on_click(event):
        global start_node, end_node, path, initial_battery, total_time
        if event.inaxes is not None:
            x, y = int(event.xdata), int(event.ydata)
            print(y, x)
            node = (y, x)

            if start_node is None:
                start_node = find_nearest_node(graph, node)
                ax.plot(start_node[1], start_node[0], 'go', markersize=10)
                plt.draw()
                print(f"Start node selected: {start_node}")
                print("start elevation: ", graph.nodes[start_node]['cost'])
                
            elif end_node is None:
                end_node = find_nearest_node(graph, node)
                ax.plot(end_node[1], end_node[0], 'bo', markersize=10)
                plt.draw()
                print(f"End node selected: {end_node}")
                print("end elevation: ", graph.nodes[end_node]['cost'])
                if start_node and end_node:


                    # # Calculate the battery and time costs for the nodes along the path
                    # nodes_visited = 0
                    # for i in range(0, (end_row - start_row), downsample_factor):
                    # #     #print("i", i)
                    #     for j in range(0, (end_col - start_col), downsample_factor):
                    #         node = (i,j)
                    #         cost = graph.nodes[node]['cost']
                    #         distance = 150#euclidean_distance(node, path[i - 1]) / resolution # Distance in pixels
                    #         battery_drain = 0.1 * nodes_visited # 0.25% per second
                    #         battery_level = initial_battery - battery_drain
                    #         nodes_visited += 1
                    #         time = distance / speed
                    #         total_time += time
                    #         #battery_level = initial_battery - battery
                    #          #cost = (cost/max_cost) * -1 + (battery_level/100) * .8
                    #          # Add the battery and time attributes to the node
                    # #         graph.nodes[node]['cost'] = cost
                    # #         #graph.nodes[node]['battery'] = battery
                    # #         graph.nodes[node]['time'] = total_time
                    # #         graph.nodes[node]['battery_level'] = battery_level

                    #          # Add the battery and time labels to the node
                    #         ax.text(node[1], node[0] - 80, f'Battery: {battery_level:.2f}%', fontsize=6, color='black', ha='left', va='top')
                    #         ax.text(node[1], node[0] - 120, f'Time: {total_time:.2f}s', fontsize=6, color='black', ha='left', va='top')
                    #         ax.text(node[1], node[0] - 160, f'Cost: {cost:.2f}', fontsize=6, color='black', ha='left', va='top')
                        
                    #     # Update the initial battery level for the next node
                    # initial_battery = battery_level

                    # Perform A* search
                    def astar_search(graph, start, end):
                        path = nx.astar_path(graph, start, end)
                        total_length_meters=0
                        for i in range(len(path) - 1):
                            total_length_meters += euclidean_distance(path[i], path[i + 1])
                            #print(total_length_meters)
                        return path, total_length_meters
                    path, total_length_meters = astar_search(graph, start_node, end_node)
                    print(f"Path calculated: {path}")
                    print(f"Total path length (meters): {total_length_meters:.2f}")

                    path_x, path_y = zip(*path)
                    ax.plot(path_y, path_x, 'g-')
                    

    fig.canvas.mpl_connect('button_press_event', on_click)
    return fig, ax
    #plt.show()


def find_nearest_node(graph, point):
    min_distance = np.inf
    nearest_node = None

    for node in graph.nodes():
        node_x, node_y = node
        distance = np.sqrt((node_x - point[0]) ** 2 + (node_y - point[1]) ** 2)

        if distance < min_distance:
            min_distance = distance
            nearest_node = node

    return nearest_node


def visualize_quadtree_graph(center_graph, fig1, ax1):
    # for i, node in enumerate(center_graph.nodes):
    #     ax1.plot(*node, 'ro', markersize=2)
    # Function to handle mouse click events
    def euclidean(node, clicked_node):
        return np.linalg.norm(np.array(node) - np.array(clicked_node))

    def on_click(event):
        global start_node, end_node
        min_distance = float('inf')
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            clicked_node = (x, y)


            #closest_node = min(center_graph.nodes(), key=lambda node: euclidean(node, clicked_node))
            for node in center_graph.nodes():
                distance = euclidean(node, clicked_node)
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node
            #closest_node = min(center_graph.nodes(), key=lambda node: np.linalg.norm(np.array(node) - np.array(clicked_node)))


            if start_node is None:
                start_node = closest_node
                ax1.plot(start_node[0], start_node[1], 'go', markersize=10)
                plt.draw()
                print("Start Node:", start_node)
            elif end_node is None:
                end_node = closest_node
                ax1.plot(end_node[0], end_node[1], 'bo', markersize=10)
                plt.draw()
                print("End Node:", end_node)

                def astar_search(center_graph, start, end):

                    path = nx.astar_path(center_graph, start, end)
                    total_length_meters=0
                    for i in range(len(path) - 1):
                        #total_length_meters += euclidean_distance(path[i], path[i + 1])
                        #print(total_length_meters)
                        return path#, #total_length_meters
                path = astar_search(center_graph, start_node, end_node)
                print(f"Path calculated: {path}")
                #print(f"Total path length (meters): {total_length_meters:.2f}")
                # flip coords
                #path_x, path_y = zip(*path)

                path_x = [node[0] for node in path]
                path_y = [node[1] for node in path]
                ax1.plot(path_x, path_y, 'bo-')

    # Connect the function to the figure's button press event
    fig1.canvas.mpl_connect('button_press_event', on_click)