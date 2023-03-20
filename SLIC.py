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

img = cv2.imread('monoUWNet/samples/00032065.tiff')

converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


height,width,channels = converted_img.shape
num_iterations = 100
prior = 5
double_step = True
num_superpixels = 150
num_levels = 20
num_histogram_bins = 5


seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
color_img = np.zeros((height,width,3), np.uint8)
color_img[:] = (0, 0, 255)
seeds.iterate(converted_img, num_iterations)


# retrieve the segmentation result
labels = seeds.getLabels()


# labels output: use the last x bits to determine the color
num_label_bits = 2
labels &= (1<<num_label_bits)-1
labels *= 1<<(16-num_label_bits)


mask = seeds.getLabelContourMask(False)


# stitch foreground & background together
mask_inv = cv2.bitwise_not(mask)
result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
result = cv2.add(result_bg, result_fg)


cv2.namedWindow('mask',0)
cv2.namedWindow('result_bg',0)
cv2.namedWindow('result_fg',0)
cv2.namedWindow('result',0)


cv2.imshow('mask',mask_inv)
cv2.imshow('result_bg',result_bg)
cv2.imshow('result_fg',result_fg)
cv2.imshow('result',result)


cv2.imwrite('mask.jpg',mask_inv)
cv2.imwrite('result_bg.jpg',result_bg)
cv2.imwrite('result_fg.jpg',result_fg)
cv2.imwrite('result.jpg',result)


cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
# cv2.imwrite("lsc_mask.tif", slic_mask) #holes visible when zooming in