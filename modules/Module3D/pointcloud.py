#!/usr/bin/env python3

""" 
pointcloud.py: Módulo de nubes de puntos. Permite generar y rotar nubes de puntos y reproyectarlas a una persepctiva cenital
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


def showPointcloud(depth, rotation_axis='y', rotation_angle=0, degrees=True, img=None):
    """Shows pointcloud from depth points applying rotation. 

    Parameters
    ----------
    depth : array_like, shape (nrows,ncols)
        Depth estimation for image_path image.
    rotation_axis : string
        Specifies the axis for rotation. Up to 3 characters belonging to the
        set {'x', 'y', 'z'}.
    rotation_angle : float
        Angle for rotation. Euler angle specified in radians
        ('degrees' is False) or degrees ('degrees' is True).
    degrees : bool, optional
        If False, then the 'angle' is assumed to be in radians.
        Default is True.
    img : array, shape (nrows,ncols,3)
        If passed, each point will be colored as img.
    """

    nrows,ncols = depth.shape

    # Plot pointcloud
    fig = plt.figure().add_subplot(projection='3d')
    fig.set_title('3D pointcloud')

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

    if img is not None:
        img_resized = cv2.resize(img, (ncols,nrows), interpolation = cv2.INTER_AREA)
        colors = list(img_resized.reshape(img_resized.shape[0]*img_resized.shape[1],3))
        colors = [[c[0]/255,c[1]/255,c[2]/255] for c in colors]

        fig.scatter(
            point_array[:, 1],
            point_array[:, 2],
            point_array[:, 0],
            s=0.01,
            c=colors
        )
    else:
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
    if img is None:
        cmap = mpl.cm.jet_r
        norm = mpl.colors.Normalize(vmin=np.amin(depth[:,2]), vmax=np.amax(depth[:,2]))

        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),label='depth estimation value')

    plt.show()



def showOverheadReproyection(image_path, depth):
    """Shows overhead reproyection of image_path using depth information 

    Parameters
    ----------
    image_path : str
        Image path
    depth : array_like, shape (nrows,ncols)
        Depth estimation for image_path image.
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