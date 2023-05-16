#!/usr/bin/env python3

""" 
depth_estimation.py: Módulo de estimación de profundidad. Permite estimar la profundidad de una imagen y mostrar el colormap de la estimación.
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import cv2
import seaborn as sns

from scipy.spatial.transform import Rotation as R
from torchvision import transforms
from .monoUWNet import networks
from .monoUWNet.layers import disp_to_depth
from .monoUWNet.my_utils import *

LIGHT_PURPLE = (213, 184, 255)
DARK_BLUE = (1, 1, 122)

def estimate(image_path):
    """ Function to estimate depth for a single image

    Parameters
    ----------
    image_path : str
        Path to the image to estimate depth on

    Returns
    -------
    depth : numpy array, shape (N,3)
        Estimated depth for each pixel of image_path
    
    """

    if torch.cuda.is_available():
        #device = torch.device("cuda")
        device = "cuda" 
    else:
        device = "cpu"

    model_folder = 'DepthEstimation/monoUWNet/20220908_FLC_all_wo_rhf_FLC_4DS_tiny_sky/models/weights_last'

    print("   Loading model from",model_folder)
    encoder_path = os.path.join(model_folder, "encoder.pth")
    depth_decoder_path = os.path.join(model_folder, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.test_hr_encoder.hrnet18(False)
    encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    para_sum_encoder = sum(p.numel() for p in encoder.parameters())
    
    print("   Loading pretrained decoder")
    depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    para_sum_decoder = sum(p.numel() for p in depth_decoder.parameters())
    depth_decoder.to(device)
    depth_decoder.eval()
    para_sum = para_sum_decoder + para_sum_encoder
    print("encoder has {} parameters".format(para_sum_encoder))
    print("depth_decoder has {} parameters".format(para_sum_decoder))
    print("encoder and depth_ decoder have  total {} parameters".format(para_sum))

    # PREDICTING ON IMAGE IN TURN
    with torch.no_grad():

        # Load image and preprocess
        input_image = pil.open(image_path).convert('RGB')
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        # Saving numpy file
        _ , depth = disp_to_depth(disp, 0.1, 100)
        
        return depth[0,0].squeeze().cpu().numpy()

def showColorMap(depth):
    """ Function to show the colormapped depth image

    Parameters
    ----------
    depth : numpy array, shape (N,3)
        Depth estimation
    
    """
    
    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    cv2.imshow('colormap depth image', colormapped_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotatePoints(points, axis, angle, degrees=False):
    """Apply an specific angle rotation about an axis to a set of points.

    Parameters
    ----------
    points : array_like, shape (N,3)
        Array containing the set of points in space
    axis : string
        Specifies the axis for rotation. Up to 3 characters belonging to the
        set {'x', 'y', 'z'}.
    angle : float
        Angle for rotation. Euler angle specified in radians
        ('degrees' isFalse) or degrees ('degrees' is True).
    degrees : bool, optional
        If True, then the 'angle' is assumed to be in degrees.
        Default is False.

    Returns
    -------
    rotated_array : array_like, shape (N,3)
        Array containing the set of points after rotation

    """
    r = R.from_euler(axis, angle, degrees)
    rotated_array = np.ndarray(shape=points.shape)
    
    i = 0
    for p in points:
        rotated_array[i] = r.apply(p)
        i += 1
    
    return rotated_array

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
    nrows,ncols = points.shape
    
    points_mask = points.copy()
    
    delete_counter = 0
    
    for i in range(nrows):
        for j in range(ncols):
            if mask[i,j,:].tolist() == list(LIGHT_PURPLE):
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
                if coloring == 'OBJECTS':
                    colors[i] = [val/255.0 for val in mask[x,y,:].tolist()]
                elif coloring == 'BINARY':
                    colors[i] = [val/255.0 for val in list(DARK_BLUE)]
                i += 1
    
    if coloring == 'DEPTH':
        colors = point_array[:, 2]
    
    return point_array, colors



def showPointcloud(depth, mask, coloring):
    """Shows pointcloud from depth points after apply the segmentation mask 'mask' to delete water points with the specify coloring type. 

    Parameters
    ----------
    depth : array_like, shape (N,3)
        Array containing the set of points in space
    mask : array_like, shape (nrows, ncols, 3)
        Segmentation mask
    coloring : str, {'OBJECTS', 'BINARY', 'DEPTH'}
        Points coloring type.
         - OBJECTS: each point (x,y,z) is colored with the color
           specified in mask[x,y,:].
         - BINARY: each point is colored with RGB DARK_BLUE = (1, 1, 122).
         - DEPTH: each point (x,y,z) is colored with colormap 'jet_r' taking depth[x,y,z] value.
    """

    if mask is None:
        print('ERROR: mask is empty. Try to use another mask')
        return

    # Resize mask
    nrows,ncols = depth.shape
    mask_resized = cv2.resize(mask, (ncols,nrows), interpolation = cv2.INTER_AREA)

    # Apply mask to pointcloud to delete water points
    pc_mask, colors = applyMask(mask_resized, depth, coloring)

    # Plot pointcloud
    fig = plt.figure().add_subplot(projection='3d')
    fig.set_title('3D pointcloud')
    
    if coloring == 'OBJECTS' or coloring == 'BINARY':
        fig.scatter(
            pc_mask[:, 1],
            pc_mask[:, 2],
            pc_mask[:, 0],
            s=0.03,
            c=colors
        )
    else:
        cmap="jet_r"
        fig.scatter(
            pc_mask[:, 1],
            pc_mask[:, 2],
            pc_mask[:, 0],
            s=0.03,
            c=colors,
            cmap=cmap
        )
    fig.view_init(0,270)

    fig.set_xlim3d(0, ncols)
    fig.set_ylim3d(np.amin(pc_mask[:,2]), np.amax(pc_mask[:,2]))
    fig.set_zlim3d(0, nrows)

    fig.invert_zaxis()
    plt.show()



def showOverheadReproyection(mask, depth):

    def points_to_image_torch(xs, ys, zs, sensor_size=(192,640)):
        xt, yt, zt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(zs)
        zt = zt.float()
        img = torch.zeros(sensor_size)
        img.index_put_((xt, yt), zt)
        return img
    
    nrows,ncols = depth.shape

    # Resize mask
    nrows,ncols = depth.shape
    mask_resized = cv2.resize(mask, (ncols,nrows), interpolation = cv2.INTER_AREA)

    # Apply mask to pointcloud to delete water points
    pc_mask, _ = applyMask(mask_resized, depth, 'depth')
    
    pc_mask[:, 2] = pc_mask[:, 2]*nrows/np.max(pc_mask[:,2])
    pc_mask = rotatePoints(pc_mask, axis='y', angle=-90, degrees=True)

    reprojection = points_to_image_torch(pc_mask[:, 0].astype(int), pc_mask[:, 1].astype(int), pc_mask[:, 2], (nrows,ncols))
                    
    reprojection = reprojection.squeeze().cpu().numpy()

    # Saving colormapped depth image
    vmax = np.percentile(reprojection, 95)
    cmap = mpl.cm.get_cmap("jet_r").copy()
    cmap.set_under(color='black')

    reprojection_plot = sns.heatmap(reprojection, vmin=0.000001, vmax=vmax, cmap=cmap)
    plt.show()