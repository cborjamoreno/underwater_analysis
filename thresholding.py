import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

from skimage import io, color, filters, morphology

# Load the image
sample = '00011499'
image = cv2.imread('monoUWNet/samples/'+sample+'.tiff')

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

plt.show()

# # Thresholding using Otsu's method
# threshold_value = filters.threshold_otsu(gray_image)
# binary_image = gray_image > threshold_value

# # Visualize the results
# plt.imshow(binary_image)
# plt.show()


# # Morphological operations
# binary_image = morphology.binary_closing(binary_image, morphology.disk(3))


# Thresholding using Otsu's method
fig, ax = filters.try_all_threshold(gray_image, figsize=(10,8), verbose=False)

plt.show()

