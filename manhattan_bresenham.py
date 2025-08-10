import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import random
import tifffile


# random.seed(3)
# pois2 = [(random.randint(2500, 2505), random.randint(2500, 2507)) for _ in range(3)]
# def draw_manhattan_line(start, end):
#     """
#     Parameters:
#     - start: Tuple[int, int], Starting coordinates of the line.
#     - end: Tuple[int, int], Ending coordinates of the line.
    
#     Returns:
#     - A list of tuples representing the points on the path.
#     """
#     x0,y0 = start
#     x1, y1 = end
#     points = []
#     xDist = abs(x1 - x0)
#     yDist = -abs(y1 - y0)
#     xStep = 1 if x0 < x1 else -1
#     yStep = 1 if y0 < y1 else -1
#     error = xDist + yDist
 
#     points.append((x0, y0))
 
#     while x0 != x1 or y0 != y1:
#         if 2 * error - yDist > xDist - 2 * error:
#             error += yDist
#             x0 += xStep
#         else:
#             error += xDist
#             y0 += yStep
        
#         points.append((x0, y0))
 
#     return points

# Generate points on a line from one of these points to all the other points
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


# for i, point in enumerate(pois2):
#     # place the text abvoe the point with an offset
#     plt.text(point[0], point[1]+0.1, f'POI {i}', fontsize=12, ha='center', va='bottom')
    
#     plt.scatter(point[0], point[1], color='red')
    
# for i in range(len(pois2)-1):
#     dist = distance.euclidean((pois2[i][0], pois2[i][1]),(pois2[i+1][0], pois2[i+1][1]))
#     print(f"Distance between point {i} and point {i+1}: {dist:.2f}")
# plt.axis('equal')
# plt.show()
