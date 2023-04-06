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
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import torch
import seaborn as sns
from scipy.spatial.transform import Rotation as R

def parse_args():
    """
    Argument parsing
    
    """
    
    parser = argparse.ArgumentParser(
        description='Pointcloud managing and image reprojection')

    parser.add_argument('-points', '--pointset_path', type=str,
                        help='path to a test pointset or folder of pointsets', required=True)
    parser.add_argument('-out', '--output_path', type=str,
                        help='dir path to save output figures', required=False)
    return parser.parse_args()


def getUWSkyline(points):
    """Get the segmentation between sea's floor and water using depth estimation

    Parameters
    ----------
    points : array_like, shape (N,3)
        Array containing the set of points in space

    Returns
    -------
    rotated_array : array_like, shape (N,2)
        Image with segmentation between sea's floor and water using depth estimation

    """
    
    light_purple = (213, 184, 255)
    dark_blue = (1, 1, 122)
    
    nrows,ncols = points.shape
    
    img = np.zeros((nrows,ncols,3), dtype=np.int32)
    
    for i in range(nrows):
        for j in range(ncols):
            if points[i,j] > 0.6:
                img[i,j] = light_purple
            else:
                img[i,j] = dark_blue
    
    return img



def main(args):
    """
    

    """
    
    # FINDING INPUT POINTS
    if os.path.isfile(args.pointset_path):
        # Only use a single pointset
        paths = [args.pointset_path]
    elif os.path.isdir(args.pointset_path):
        # Searching folder for pointsets
        paths = glob.glob(os.path.join(args.pointset_path,'**/*.{}'.format('tiff.npy')), recursive=True)
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

        # Cutting y
        # depth = depth[200:,:]

        # Cutting depth
        # for x in range(0, rows):
        #     for y in range(0, cols):
        #         if depth[x,y] > 0.6:
        #             depth[x,y] = 0.6
        
        fig = plt.figure(figsize=(15, 10))
        
        img = getUWSkyline(depth)
        
        plt.imshow(img)
        plt.show()

        
       
    
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
