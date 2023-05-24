#!/usr/bin/env python3

""" 
pointcloud.py: Módulo de nubes de puntos. Permite generar y manejar nubes de puntos y reproyectarlas a una persepctiva cenital
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import cv2
import seaborn as sns

from scipy.spatial.transform import Rotation as R

LIGHT_PURPLE = (213, 184, 255)
DARK_BLUE = (1, 1, 122)

def rotatePoints(points, axis, angle, degrees=True):
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
        If False, then the 'angle' is assumed to be in radians.
        Default is True.

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


def showPointcloud(depth, rotation_axis='y', rotation_angle=0, degrees=True):
    """Shows pointcloud from depth points applying rotation. 

    Parameters
    ----------
    depth : array_like, shape (N,3)
        Array containing the set of points in space
    axis : string
        Specifies the axis for rotation. Up to 3 characters belonging to the
        set {'x', 'y', 'z'}.
    angle : float
        Angle for rotation. Euler angle specified in radians
        ('degrees' is False) or degrees ('degrees' is True).
    degrees : bool, optional
        If False, then the 'angle' is assumed to be in radians.
        Default is True.
    """

    nrows,ncols = depth.shape

    # Plot pointcloud
    fig = plt.figure().add_subplot(projection='3d')
    # fig.set_title('3D pointcloud')

    point_array = np.ndarray(shape=(nrows*ncols,3))

    i = 0
    for x in range(0, nrows):
        for y in range(0, ncols):
            point_array[i] = [x,y,depth[x,y]]
            i += 1
            
    if rotation_angle != 0:
        # Applying rotation
        point_array = rotatePoints(point_array, rotation_axis, rotation_angle, degrees)
        if rotation_axis == 'y':
            c=point_array[:, 0],
            cmap="jet"
        elif rotation_axis == 'x':
            c=point_array[:, 1],
            cmap="jet"
        else:
            c=point_array[:, 2],
            cmap="jet_r"
    else:
        c=point_array[:, 2],
        cmap="jet_r"

    fig.scatter(
        point_array[:, 1],
        point_array[:, 2],
        point_array[:, 0],
        s=0.01,
        c=c,
        cmap=cmap
    )
    fig.view_init(15,235)

    fig.set_xlabel(" Y ")
    fig.set_ylabel(" Z ")
    fig.set_zlabel(" X ")

    fig.invert_zaxis()

    cmap = mpl.cm.jet_r
    norm = mpl.colors.Normalize(vmin=np.amin(depth[:,2]), vmax=np.amax(depth[:,2]))

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),label='depth estimation value')

    plt.show()

def showPointcloudWithMask(depth, mask, coloring):
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
    
    print('MAXIMO', np.amax(depth[:,2]))

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
            c=colors,
            vmin=np.amin(depth[:,2]),
            vmax = np.amax(depth[:,2])
        )
    else:
        cmap="jet_r"
        fig.scatter(
            pc_mask[:, 1],
            pc_mask[:, 2],
            pc_mask[:, 0],
            s=0.03,
            c=colors,
            cmap=cmap,
            vmin=np.amin(depth[:,2]),
            vmax = np.amax(depth[:,2])
        )
    fig.view_init(15,235)

    fig.set_xlim3d(0, ncols)
    fig.set_ylim3d(np.amin(depth[:,2]), np.amax(depth[:,2]))
    fig.set_zlim3d(0, nrows)

    fig.set_xlabel(" Y ")
    fig.set_ylabel(" Z ")
    fig.set_zlabel(" X ")

    fig.invert_zaxis()
    plt.show()



def showOverheadReproyection(image_path, depth, mask=None):
    """Shows overhead reproyection of depth pointcloud 

    Parameters
    ----------
    depth : array_like, shape (N,3)
        Array containing the set of points in space
    mask : array_like, shape (nrows, ncols, 3), optional
        Segmentation mask
    """

    def points_to_image_torch(xs, ys, zs, sensor_size=(192,640)):
        xt, yt, zt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(zs)
        zt = zt.float()
        img = torch.zeros(sensor_size)
        img.index_put_((xt, yt), zt)
        return img
    
    nrows,ncols = depth.shape

    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image
    img_resized = cv2.resize(img, (ncols,nrows), interpolation = cv2.INTER_AREA)
    depth_rescaled = np.copy(depth)

    if mask is not None:
        # Resize mask
        mask_resized = cv2.resize(mask, (ncols,nrows), interpolation = cv2.INTER_AREA)

        # Apply mask to pointcloud to delete water points
        pc_mask, _ = applyMask(mask_resized, depth, 'depth')
        
        pc_mask[:, 2] = pc_mask[:, 2]*nrows/np.max(pc_mask[:,2])
        points_rotated = rotatePoints(pc_mask, axis='y', angle=-90, degrees=True)
    
    else:

        point_array_rescaled = np.ndarray(shape=(nrows*ncols,3))
        depth_rescaled[:,:] = (depth_rescaled[:,:]*nrows) / np.max(depth)

        i = 0
        for x in range(0, nrows):
            for y in range(0, ncols):
                point_array_rescaled[i] = [x,y,depth_rescaled[x,y]]
                i += 1
        points_rotated = rotatePoints(point_array_rescaled, axis='y', angle=-90, 
        degrees=True)


    reprojection = points_to_image_torch(points_rotated[:, 0].astype(int), points_rotated[:, 1].astype(int), points_rotated[:, 2], (nrows,ncols))
                    
    reprojection = reprojection.squeeze().cpu().numpy()
    reprojection_colored = np.zeros((reprojection.shape[0],reprojection.shape[1],3))


    r = R.from_euler('y',90,True)
    i = 0

    for i in range(len(points_rotated)):
        x = int(round(np.max(depth_rescaled))) + int(round(points_rotated[i,0]))
        y = int(round(points_rotated[i,1]))
        if reprojection[x,y] > 0:
            p = r.apply(points_rotated[i])
            x_ori = int(round(p[0]))
            y_ori = int(round(p[1]))
            reprojection_colored[x,y,:] = img_resized[x_ori,y_ori]

    # Resize image
    reprojection_colored = cv2.resize(reprojection_colored, (img.shape[1],img.shape[0]), interpolation = cv2.INTER_AREA)
    depth_rescaled = np.copy(depth)

    cv2.imshow('final',cv2.cvtColor(reprojection_colored.astype(np.uint8), cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()