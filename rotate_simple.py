import os
import sys

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import ndimage, misc

fig = plt.figure(figsize=(10, 3))
img = cv2.imread('monoUWNet/results/00032065._disp.jpeg')
img_45 = ndimage.rotate(img, 1, axes=(1,2), reshape=False)

cv2.imshow('image', img_45)
# ax1.imshow(img, cmap='gray')
# ax1.set_axis_off()
# ax2.imshow(img_45, cmap='gray')
# ax2.set_axis_off()
# ax3.imshow(full_img_45, cmap='gray')
# ax3.set_axis_off()
# fig.set_layout_engine('tight')
# plt.show()


# Maintain output window utill
# user presses a key
cv2.waitKey(0)       
 
# Destroying present windows on screen
cv2.destroyAllWindows()
