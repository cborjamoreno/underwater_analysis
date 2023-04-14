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
    parser.add_argument('-reproject', '--reprojection', action='store_true',
                        help='reproject pointcloud to image', required=False)
    
    rotation = parser.add_mutually_exclusive_group(required=False)
    rotation.add_argument('-ax', '--rotation_axis', type=str, default='y', choices=['x','y','z'],
                        help='axis for rotation', required=False)
    rotation.add_argument('-angle', '--rotation_angle', type=float, default=0,
                        help='angle for rotation. Degrees.', required=False)
    return parser.parse_args()


def rotatePoints(point_array, axis, angle, degrees=False):
    """Apply an specific angle rotation about an axis to a set of points.

    Parameters
    ----------
    point_array : array_like, shape (N,3)
        Array containing the set of points to be rotated
    axis : string
        Specifies the axis for rotation. Up to 3 characters belonging to the
        set {'x', 'y', 'z'}.
    angle : float
        Angle for rotation. Euler angle specified in radians
        (`degrees` isFalse) or degrees (`degrees` is True).
    degrees : bool, optional
        If True, then the given angle is assumed to be in degrees.
        Default is False.

    Returns
    -------
    rotated_array : array_like, shape (N,3)
        Array containing the set of points after rotation

    """
    
    r = R.from_euler(axis, angle, degrees)
    rotated_array = np.ndarray(shape=point_array.shape)
    
    i = 0
    for p in point_array:
        rotated_array[i] = r.apply(p)
        i += 1
    
    return rotated_array


def points_to_image_torch(xs, ys, zs, sensor_size=(192,640)):
    xt, yt, zt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(zs)
    zt = zt.float()
    img = torch.zeros(sensor_size)
    img.index_put_((xt, yt), zt)
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

        rows,cols = depth.shape

        # Cutting depth
        for x in range(0, rows):
            for y in range(0, cols):
                if depth[x,y] > 0.55:
                    depth[x,y] = 0.7

        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection="3d")
        
        point_array = np.ndarray(shape=(rows*cols,3))
        
        #Rescalling z
        if args.reprojection:
            point_array_rescaled = np.ndarray(shape=(rows*cols,3))
            depth_rescaled = np.copy(depth)
            depth_rescaled[:,:] = (depth_rescaled[:,:]*rows) / np.max(depth_rescaled)
            i = 0
            for x in range(0, rows):
                for y in range(0, cols):
                    point_array_rescaled[i] = [x,y,depth_rescaled[x,y]]
                    i += 1

        
        i = 0
        for x in range(0, rows):
            for y in range(0, cols):
                point_array[i] = [x,y,depth[x,y]]
                i += 1
                
        if args.rotation_angle != 0:
            # Applying rotation
            point_array = rotatePoints(point_array, args.rotation_axis, args.rotation_angle, True)
            if args.reprojection:
                point_array_rescaled = rotatePoints(point_array_rescaled, args.rotation_axis, args.rotation_angle, True)
            c=point_array[:, 0],
            cmap="jet"
        else:
            c=point_array[:, 2],
            cmap="jet_r"

        ax.scatter(
            point_array[:, 1],
            point_array[:, 2],
            point_array[:, 0],
            s=0.03,
            c=c,
            cmap=cmap
        )
        ax.view_init(0,270)

        ax.set_xlabel(" Y ")
        ax.set_ylabel(" Z ")
        ax.set_zlabel(" X ")

        ax.invert_zaxis()
        plt.show()
        
        if args.output_path:
            sample = os.path.splitext(path)
            sample = os.path.splitext(sample[0])
            sample = (sample[0].split('/'))[2]
            
            if args.rotation_angle != 0:
                fig.savefig(out_path+sample+"_rotated_pointcloud.png", bbox_inches='tight')
            else:
                fig.savefig(out_path+sample+"_pointcloud.png", bbox_inches='tight')


        if args.reprojection:
            
            reprojection = points_to_image_torch(point_array_rescaled[:, 0].astype(int),
                                                 point_array_rescaled[:, 1].astype(int),
                                                 point_array_rescaled[:, 2], (rows,cols))
                    
            reprojection = reprojection.squeeze().cpu().numpy()
    
            # Saving colormapped depth image
            vmax = np.percentile(reprojection, 95)
            cmap = mpl.cm.get_cmap("jet_r").copy()
            cmap.set_under(color='black')
    
            reprojection_plot = sns.heatmap(reprojection, vmin=0.000001, vmax=vmax, cmap=cmap)
            fig = reprojection_plot.get_figure()
    
            if args.output_path:
                sample = os.path.splitext(path)
                sample = os.path.splitext(sample[0])
                sample = (sample[0].split('/'))[2]
                
                if args.rotation_angle != 0:
                    fig.savefig(out_path+sample+"_rotated.png", bbox_inches='tight')
                else:
                    fig.savefig(out_path+sample+".png", bbox_inches='tight')
    
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
