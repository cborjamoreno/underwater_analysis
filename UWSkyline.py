#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:45:20 2023

@author: cbm
"""

"""
Spyder Editor


"""


import os, sys
import glob
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import torch
import random
import time

from mpl_toolkits.mplot3d import Axes3D
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def parse_args():
    """
    Argument parsing
    
    """
    
    parser = argparse.ArgumentParser(
        description='Underwater skyline segmentation')

    parser.add_argument('-points', '--pointset_path', type=str,
                        help='path to a test pointset or folder of pointsets', required=True)
    parser.add_argument('-out', '--output_path', type=str,
                        help='dir path to save output figures', required=False)
    parser.add_argument('--SAM', 
                        help='path of image to use SAM on it', type=str, required=False)
    parser.add_argument('--colorSeg', 
                        help='Segment rest of the objects in scene, not only water and non-water', action='store_true', required=False)
    parser.add_argument('--pc_color', 
                        help='Point cloud colors', choices=['depth', 'binary', 'objects'], required=False, default='objects')
    
    return parser.parse_args()


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.5)))

def getUWSkyline_threshold(points):
    """Get the segmentation between sea's floor and water using depth estimation

    Parameters
    ----------
    points : array_like, shape (N,3)
        Array containing the set of points in space

    Returns
    -------
    img : array_like, shape (N,2)
        Image with segmentation between sea's floor and water

    """
    
    light_purple = (213, 184, 255)
    dark_blue = (1, 1, 122)

    nrows,ncols = points.shape
    
    img = np.zeros((nrows,ncols,3), dtype=np.int32)
    
    for i in range(nrows):
        for j in range(ncols):
            if points[i,j] > 0.55:
                img[i,j] = light_purple
            else:
                img[i,j] = dark_blue
    
    return img

def getUWSkyline_SAM(points, sample, SAM_path, colorSeg=False):
    """Get the segmentation between sea's floor and water using SAM segmentation

    Parameters
    ----------
    points : array_like, shape (N,3)
        Array containing the set of points in space
    sample : str
        Sample id
    SAM_path : str
        Path of image to use SAM on it
    colorSeg : boolean
        If True, segment objects in scene apart from water and non-water segmentation

    Returns
    -------
    img : array_like, shape (N,2)
        Image with segmentation between sea's floor and water

    """
    light_purple = (213, 184, 255)
    dark_blue = (1, 1, 122)
    
    nrows_pts, ncols_pts = points.shape
    
    img = cv2.imread(SAM_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    nrows_img, ncols_img, _ = img.shape
    
    

    # Segmentation with depth
    thresh = getUWSkyline_threshold(points)
    thresh = cv2.resize(thresh, (ncols_img,nrows_img), interpolation = cv2.INTER_LINEAR_EXACT)
    
    

    # Calculate water pixels percent in thresh
    water = 0
    
    for i in range(nrows_img):
        for j in range(ncols_img):
            if thresh[i,j,:].tolist() == list(light_purple):
                water += 1
    water_perc_depth = (water/(ncols_pts*nrows_pts)) * 100

    # Get water pixels in thresh
    indices = np.where(np.all(thresh == light_purple, axis=-1))
    water_thresh = list(zip(indices[0], indices[1]))
    
    # Get SAM segmentation
    sam = sam_model_registry["vit_h"](checkpoint="vit_h.pth")
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32
    )
    mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(img)
    masks = mask_generator_2.generate(img)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
    # fig = plt.figure()
    # plt.imshow(thresh)
    # plt.axis('off')
    # plt.show()
    
    # fig = plt.figure(figsize=(20,20))
    # plt.imshow(img)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()
    # fig.savefig('SemanticSegmentationUnderwaterImagery/results/SAM/'+sample+'SAM.png')
    

    # Select biggest segment as water
    segment_areas = [sub['area'] for sub in masks]
    water_segment_index = segment_areas.index(max(segment_areas))
    water_segment = masks[water_segment_index]
    
    
    intersection = 0
    union = 0
    if colorSeg:
    
        # Merge all masks to obtain actual water segment
        water_segment_merged = np.zeros(water_segment['segmentation'].shape, dtype=np.int32)
        for i in range(water_segment_merged.shape[0]):
            for j in range(water_segment_merged.shape[1]):
                for m in range(len(masks)):
                    if masks[m]['segmentation'][i,j]:
                        water_segment_merged[i,j] = m+1
        
        # Check if selected water segment has a good "intersect over union" value
        for i in range(water_segment_merged.shape[0]):
            for j in range(water_segment_merged.shape[1]):
                if water_segment_merged[i,j] == water_segment_index+1:
                    if thresh[i,j,:].tolist() == list(light_purple):
                        intersection += 1
                    union += 1
    else:
    
        masks_aux = masks.copy()
        del masks_aux[water_segment_index]
        # Merge all masks to obtain actual water segment
        water_segment_merged = water_segment['segmentation'].copy()
        for i in range(water_segment_merged.shape[0]):
            for j in range(water_segment_merged.shape[1]):
                for m in masks_aux:
                    if m['segmentation'][i,j]:
                        water_segment_merged[i,j] = False
        
        # Check if selected water segment has a good "intersect over union" value
        for i in range(water_segment_merged.shape[0]):
            for j in range(water_segment_merged.shape[1]):
                if water_segment_merged[i,j]:
                    if thresh[i,j,:].tolist() == list(light_purple):
                        intersection += 1
                    union += 1
            

    while (intersection/union) < 0.4:
        del segment_areas[water_segment_index]
        if len(segment_areas) == 0:
            print('SE ELIGE DEPTH')
            return thresh
        water_segment_index = segment_areas.index(max(segment_areas))
        water_segment = masks[water_segment_index]
        
        if colorSeg:
            
            # Merge all masks to obtain actual water segment
            water_segment_merged = np.zeros(water_segment['segmentation'].shape, dtype=np.int32)
            for i in range(water_segment_merged.shape[0]):
                for j in range(water_segment_merged.shape[1]):
                    for m in range(len(masks)):
                        if masks[m]['segmentation'][i,j]:
                            water_segment_merged[i,j] = m+1
            
            # Check if selected water segment has a good "intersect over union" value
            intersection = 0
            union = 0
            for i in range(water_segment_merged.shape[0]):
                for j in range(water_segment_merged.shape[1]):
                    if water_segment_merged[i,j] == water_segment_index+1:
                        if thresh[i,j,:].tolist() == list(light_purple):
                            intersection += 1
                        union += 1
                        
        else:
                        
            masks_aux = masks.copy()
            del masks_aux[water_segment_index]
            # Merge all masks to obtain actual water segment
            water_segment_merged = water_segment['segmentation'].copy()
            for i in range(water_segment_merged.shape[0]):
                for j in range(water_segment_merged.shape[1]):
                    for m in masks_aux:
                        if m['segmentation'][i,j]:
                            water_segment_merged[i,j] = False
            
            # Check if selected water segment has a good "intersect over union" value
            intersection = 0
            union = 0
            for i in range(water_segment_merged.shape[0]):
                for j in range(water_segment_merged.shape[1]):
                    if water_segment_merged[i,j]:
                        if thresh[i,j,:].tolist() == list(light_purple):
                            intersection += 1
                        union += 1
        
    union += (water - intersection)
    print('Intersection over union =',intersection/union)
    
    if colorSeg:
        
        # Defining random colors
        colors = []
        for i in range(len(masks)):
            for j in range(3):
                colors.append(list(np.random.choice(range(255),size=3)))
        
        # Create segmentation mask
        for i in range(water_segment_merged.shape[0]):
            for j in range(water_segment_merged.shape[1]):
                if water_segment_merged[i,j] == water_segment_index+1:
                    img[i,j] = light_purple
                else:  
                    img[i,j] = colors[water_segment_merged[i,j]]
    else:
        
        # Create binary mask
        for i in range(water_segment_merged.shape[0]):
            for j in range(water_segment_merged.shape[1]):
                if water_segment_merged[i,j]:
                    img[i,j] = light_purple
                else:  
                    img[i,j] = dark_blue
        
                
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    
    # Apply erode and dilate to reduce noise
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    
    return img

def applyMask(mask, points, coloring):
    """Apply skyline mask to pointcloud

    Parameters
    ----------
    mask : array_like, shape (nrows, ncols, 3)
        Image that represent de skyline mask
    points : array_like, shape (N,3)
        Array containing the set of points in space

    Returns
    -------
    point_array : array_like 
        Resulting array after deleting water pixels

    """
    
    light_purple = (213, 184, 255)
    dark_blue = (1, 1, 122)
    
    nrows,ncols = points.shape
    
    points_mask = points.copy()
    
    delete_counter = 0
    
    for i in range(nrows):
        for j in range(ncols):
            if mask[i,j,:].tolist() == list(light_purple):
                points_mask[i,j] = 1
                delete_counter += 1
                
    useful = nrows*ncols - delete_counter
                
    point_array = np.zeros(shape=(useful,3))
    colors = np.array(np.zeros(shape=(useful,3)))
    
    
    i = 0
    for x in range(nrows):
        for y in range(ncols):
            if points_mask[x,y] < 0.9:
                point_array[i] = [x,y,points_mask[x,y]]
                if coloring == 'objects':
                    colors[i] = [val/255.0 for val in mask[x,y,:].tolist()]
                elif coloring == 'binary':
                    colors[i] = [val/255.0 for val in list(dark_blue)]
                i += 1
    
    if coloring == 'depth':
        colors = point_array[:, 2]
    
    return point_array, colors
                



def main(args):
    """
    

    """
    
    # FINDING INPUT POINTS
    if os.path.isfile(args.pointset_path):
        # Only use a single pointset
        paths = [args.pointset_path]

        # Checking SAM path
        if args.SAM:
            SAM_path = args.SAM

    elif os.path.isdir(args.pointset_path):
        # Searching folder for pointsets
        paths = glob.glob(os.path.join(args.pointset_path,'**/*.{}'.format('npy')), recursive=True)

        # Checking SAM path
        if args.SAM:
            SAM_path = args.SAM
            if SAM_path[len(SAM_path)-1] != '/':
                # Append '/' to SAM path
                SAM_path += '/'
    else:
        raise Exception("Can not find args.pointset_path: {}".format(args.pointset_path))
        
    # Checking output path
    if args.output_path:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        
        out_path = args.output_path
        if out_path[len(out_path)-1] != '/':
            # Append '/' to output path
            out_path += '/'
    
    for path in paths:
        
        depth_load = np.load(path)
        depth = depth_load[0][0]

        nrows,ncols = depth.shape

        sample = os.path.splitext(path)
        sample = os.path.splitext(sample[0])
        sample = (sample[0].split('/'))[-1]

        print('Generating mask for sample:', sample)


        start = time.time()
        if args.SAM:
            if args.colorSeg:
                colorSeg = True
            else:
                colorSeg = False
                
            if os.path.isfile(args.pointset_path):
                mask = getUWSkyline_SAM(depth, sample, str(SAM_path), colorSeg)
            else:
                mask = getUWSkyline_SAM(depth, sample, str(SAM_path+sample+'.jpg'), colorSeg)
                
        else:
            mask = getUWSkyline_threshold(depth)
        end = time.time()
        
        print('Execution time:',end-start,'s')



        dim = (ncols,nrows)
        resized = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
        pc_mask, colors = applyMask(resized, depth, args.pc_color)
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        
        # =============
        # First subplot
        # =============
        # set up the axes for the first plot
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(resized)
        
        # ==============
        # Second subplot
        # ==============
        # set up the axes for the second plot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        
        
        if args.pc_color == 'objects' or args.pc_color == 'binary':
            ax.scatter(
                pc_mask[:, 1],
                pc_mask[:, 2],
                pc_mask[:, 0],
                s=0.03,
                c=colors
            )
        else:
            cmap="jet_r"
            ax.scatter(
                pc_mask[:, 1],
                pc_mask[:, 2],
                pc_mask[:, 0],
                s=0.03,
                c=colors,
                cmap=cmap
            )
        ax.view_init(0,270)
        ax.dist = 7

        ax.set_xlim3d(0, ncols)
        ax.set_ylim3d(np.amin(pc_mask[:,2]), np.amax(pc_mask[:,2]))
        ax.set_zlim3d(0, nrows)

        ax.invert_zaxis()
        plt.show()

        
    
        if args.output_path:
            fig.savefig(out_path+sample+"_segmentation_and_pc.png", bbox_inches='tight')

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
