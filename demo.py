#!/usr/bin/env python3

""" 
demo.py: Este script hace una segmentación de la masa de agua de la imagen pasada por parámetro, separándola de los elementos de interés de la imagen. También estima la profundidad de la imagen y crea una nube de puntos 3D de la escena 
"""


import argparse

from Module3D.depth_estimation import *
from Module3D.pointcloud import *
from Segmentation.segmentation import *

def parse_args():
    """
    Argument parsing
    
    """
    
    parser = argparse.ArgumentParser(
        description='Demo')

    parser.add_argument('-p', '--path', type=str,
                        help='Input image path', required=True)
    parser.add_argument('--pc_color', 
                        help='Point cloud colors.', choices=['DEPTH', 'BINARY', 'OBJECTS'], default='OBJECTS')
    parser.add_argument('-eval', '--evalPath', type=str,
                        help='Evaluation segmented mask path. Water pixels must be labeled as RGB (0,0,0) in mask located in \'evalPath\'.')
    
    return parser.parse_args()

def main(args):
    image_path = args.path

    print('Estimating depth for image:'+image_path+'...')
    depth = estimate(image_path)
    print('-> Done!\n')

    # print('Showing colormap estimation...')
    # showColorMap(depth)
    # print('-> Done!\n')

    print('Showing 3D pointcloud')
    showPointcloud(depth,rotation_axis='y',rotation_angle=0)

    # print('Generating binary and object segmentation...')
    # binary_mask, color_mask = segmentationFinal(depth, image_path)
    # print('-> Done!\n')

    # print('Showing segmentations...')
    # showSegmentation(depth, binary_mask, color_mask)
    # print('-> Done!\n')
    
    # print('Showing 3D pointcloud with '+args.pc_color+' coloring type...')
    # showPointcloudWithMask(depth, color_mask, args.pc_color)
    # print('-> Done!\n')

    print('Showing overhead reproyection...')
    showOverheadReproyection(depth)
    print('-> Done!\n')

    # if args.evalPath:
    #     print('Evaluating...')
    #     TP, FP, TN, FN = evaluate(args.evalPath, binary_mask)

    #     print('Precision =',TP/(TP+FP))
    #     print('Recall =',TP/(TP+FN))
    #     print('-> Done!\n')

if __name__ == "__main__":
    args = parse_args()
    main(args)