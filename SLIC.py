#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:49:14 2023

@author: cbm
"""

import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

sample = '00034302'

img = cv2.imread('monoUWNet/samples/'+sample+'.tiff')

hsv_image= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

height,width,channels = hsv_image.shape
num_iterations = 500
prior = 5
double_step = True 
num_superpixels = 800
num_levels = 20
num_histogram_bins = 5


seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
color_img = np.zeros((height,width,3), np.uint8)
color_img[:] = (0, 0, 255)

# Iterate until segmentation is done
seeds.iterate(img, num_iterations)

# Get the labels for each pixel
labels = seeds.getLabels()

superpixel_mask = seeds.getLabelContourMask(False)

# stitch foreground & background together
mask_inv = cv2.bitwise_not(superpixel_mask)
result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
result_fg = cv2.bitwise_and(color_img, color_img, mask=superpixel_mask)
result = cv2.add(result_bg, result_fg)

# Create an empty image to draw the superpixel segments
output = np.zeros_like(img)

# Define HSV blue range
lower_blue = np.array([97,170,70])
upper_blue= np.array([130,255,255])

# Define HSV white range
lower_white = np.array([0,0,150])
upper_white = np.array([180,255,255])

light_purple = (255, 213, 184)

dark_blue = (122, 1, 1)


# Loop over each superpixel and color it in either light blue or light yellow
for i in range(num_superpixels):
    mask = labels == i
    if (np.count_nonzero((hsv_image[mask] >= lower_blue).all(axis=1) & (hsv_image[mask] <= upper_blue).all(axis=1)) >= len(hsv_image[mask])) or (np.count_nonzero((hsv_image[mask] >= lower_white).all(axis=1) & (hsv_image[mask] <= upper_white).all(axis=1)) >= len(hsv_image[mask])):
        output[mask] = light_purple
    else:
        output[mask] = dark_blue


cv2.imshow('superpixels', result)
cv2.waitKey(0)
cv2.imwrite('results/segmentation/superpixel_'+sample+'.png', result)
# Show image
cv2.imshow('Superpixel segmentation', output)
cv2.waitKey(0)
cv2.imwrite('results/segmentation/segmentation_'+sample+'.png', output)
cv2.destroyAllWindows()




# labels output: use the last x bits to determine the color
# num_label_bits = 2
# labels &= (1<<num_label_bits)-1
# labels *= 1<<(16-num_label_bits)
# labels = (labels * 255).round().astype(np.uint8)


# mask = seeds.getLabelContourMask(False)


# # stitch foreground & background together
# mask_inv = cv2.bitwise_not(mask)
# result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
# result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
# result = cv2.add(result_bg, result_fg)


# cv2.namedWindow('mask',0)
# cv2.namedWindow('result_bg',0)
# cv2.namedWindow('result_fg',0)
# cv2.namedWindow('result',0)


# cv2.imshow('mask',mask_inv)
# cv2.imshow('result_bg',result_bg)
# cv2.imshow('result_fg',result_fg)
# cv2.imshow('result',result)


# cv2.imwrite('mask.jpg',mask_inv)
# cv2.imwrite('result_bg.jpg',result_bg)
# cv2.imwrite('result_fg.jpg',result_fg)
# cv2.imwrite('result.jpg',result)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# cv2.imwrite("lsc_mask.tif", slic_mask) #holes visible when zooming in