import os
import sys

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.spatial.transform import Rotation as R

depth_load = np.load(sys.argv[1])
depth = depth_load[0][0]

# Rotation's angle (radians)
phi = math.radians(90)

# Rotation matrix in x axis
# r = R.from_matrix([[1,             0,             0],
#                    [0, math.cos(phi), -math.sin(phi)],
#                    [0, math.sin(phi),  math.cos(phi)]])

# Rotation matrix in y axis
r = R.from_matrix([[math.cos(phi), 0, math.sin(phi)],
                   [0,             1,             0],
                   [0, -math.sin(phi),  math.cos(phi)]])

# Rotation matrix in z axis
# r = R.from_matrix([[math.cos(phi), -math.sin(phi), 0],
#                    [math.sin(phi),  math.cos(phi), 0],
#                    [0,             0,             1]])

rows,cols = depth.shape

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection="3d")

STEP = 3
points_array = np.ndarray(shape=(rows*cols,3))

i = 0
for x in range(0, depth.shape[0]):
    for y in range(0, depth.shape[1]):
        points_array[i] = r.apply([x,y,depth[x,y]])
        i += 1

ax.scatter(
    points_array[:, 1],
    points_array[:, 2],
    points_array[:, 0],
    s=3,
    c=points_array[:, 2],
    cmap="plasma_r"
)

ax.view_init(45,235)

ax.set_xlabel(" Y ")
ax.set_ylabel(" Z ")
ax.set_zlabel(" X ")

ax.invert_zaxis()
plt.show()
