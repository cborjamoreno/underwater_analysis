#!/usr/bin/env python3

""" 
demo.py: Este script hace una segmentación de la masa de agua de la imagen pasada por parámetro, separándola de los elementos de interés de la imagen. También estima la profundidad de la imagen y crea una nube de puntos 3D de la escena 
"""


import argparse
import time

from modules.Module3D.depth_estimation import *
from modules.Module3D.pointcloud import *
from modules.Segmentation.segmentation import *
from modules.Final.final import *

def parse_args():
    """
    Argument parsing
    
    """
    
    parser = argparse.ArgumentParser(
        description='Demo')

    parser.add_argument('-p', '--path', type=str,
                        help='Input image path', required=True)
    parser.add_argument('--colormap', action='store_true',
                        help='Show colormap')
    parser.add_argument('--pointcloud', nargs=2,
                        metavar=('rotation_axis','rotation_amgle (degrees)'),
                        help='Show pointcloud with a rotation')
    parser.add_argument('--reprojection', action='store_true',
                        help='Show overhead reprojection')
    parser.add_argument('--superpixels_seg', action='store_true',
                        help='Show superpixels segmentation')
    parser.add_argument('--depth_seg', action='store_true',
                        help='Show depth based segmentation')
    parser.add_argument('--floating_seg', action='store_true',
                        help='Show floating segmentation')
    parser.add_argument('--SAM', action='store_true',
                        help='Show SAM result')
    parser.add_argument('--pc_color', 
                        help='Point cloud colors.', choices=['DEPTH', 'FLOATING'], default='FLOATING')
    parser.add_argument('-eval', '--evalPath', type=str,
                        help='Evaluation segmented mask path. Water pixels must be labeled as RGB (0,0,0) in mask located in \'evalPath\'.')
    
    return parser.parse_args()

def main(args):
    image_path = args.path
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    if args.colormap or args.pointcloud or args.reprojection:
        print('Estimating depth for image:'+image_path+'...')
        depth = estimate(image_path)
        print('-> Done!\n')
        if args.colormap:
            print('Showing colormap estimation...')
            showColorMap(depth, image_path)
            print('-> Done!\n')
        if args.pointcloud:
            print('Showing 3D pointcloud')
            showPointcloud(depth,rotation_axis=args.pointcloud[0],rotation_angle=args.pointcloud[1],degrees=True,img=img)
        if args.reprojection:
            print('Showing overhead reproyection...')
            print('Press any key to close the window.')
            showOverheadReproyection(image_path,depth)
            print('-> Done!\n')

    if args.superpixels_seg:
        print('Generating binary superpixels segmentation...')
        showBinarySegmentationSuperpixels(image_path)
        print('-> Done!\n')
    if args.depth_seg:
        print('Generating binary depth segmentation...')
        showBinarySegmentationDepth(image_path)
        print('-> Done!\n')
    if args.SAM:
        print('Generating SAM result...')
        masks = segmentationSAM(img)
        showSAM(img,masks)
        print('-> Done!\n')

    print('Generating binary and object segmentation...')
    binary_mask, color_mask = segmentationFinal(image_path,args.pc_color)
    print('-> Done!\n')
    
    if args.floating_seg:
        print('Generating floating objects segmentation...')
        showFloatingSegmentation(binary_mask)
        print('-> Done!\n')

    if args.evalPath:
        print('Evaluating...')
        TP, FP, TN, FN = evaluate(args.evalPath, binary_mask)

        print('Precision =',TP/(TP+FP))
        print('Recall =',TP/(TP+FN))
        print('-> Done!\n')

if __name__ == "__main__":
    args = parse_args()
    main(args)