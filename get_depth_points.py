import os
import sys

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

depth = np.load(sys.argv[1])
depth1 = depth[0][0]

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection="3d")

STEP = 3

for x in range(0, depth1.shape[0], STEP):
    for y in range(0, depth1.shape[1], STEP):
        ax.scatter(
            [y] * 3,
            depth1[x,y] * 3,
            [x] * 3,
            s=3,
            c = [0,0,1]
        )
    ax.view_init(45,235)

ax.set_xlabel(" X ")
ax.set_ylabel(" Y ")
ax.set_zlabel(" Z ")

ax.invert_zaxis()
plt.show()