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
    parser.add_argument('--pc_color', 
                        help='Point cloud colors.', choices=['DEPTH', 'FLOATING'], default='DEPTH')
    parser.add_argument('-eval', '--evalPath', type=str,
                        help='Evaluation segmented mask path. Water pixels must be labeled as RGB (0,0,0) in mask located in \'evalPath\'.')
    
    return parser.parse_args()

def main(args):
    image_path = args.path
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    # print('Generating binary superpixels segmentation...')
    # binary_mask = showBinarySegmentationSuperpixels(image_path)
    # print('-> Done!\n')

    # print('Generating binary depth segmentation...')
    # binary_mask = showBinarySegmentationDepth(image_path)
    # print('-> Done!\n')

    # print('Estimating depth for image:'+image_path+'...')
    # depth = estimate(image_path)
    # print('-> Done!\n')
    # print('Showing colormap estimation...')
    # showColorMap(depth, image_path)
    # print('-> Done!\n')
    # print('Showing 3D pointcloud')
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # showPointcloud(depth,rotation_axis='y',rotation_angle=0)
    # print('Showing 3D pointcloud')
    # showPointcloud(depth,rotation_axis='y',rotation_angle=-90,degrees=True,img=img)
    
    print('Generating binary and object segmentation...')
    binary_mask, color_mask = segmentationFinal(image_path,args.pc_color)
    print('-> Done!\n')


    # print('Generating floating objects segmentation...')
    # showFloatingSegmentation(binary_mask)
    # print('-> Done!\n')

    # masks = segmentationSAM(img)
    # showSAM(img,masks)

    # print('Showing 3D pointcloud with '+args.pc_color+' coloring type...')
    # showPointcloudWithMask(depth, color_mask, 'OBJECTS')
    # print('-> Done!\n')
    # print('Showing 3D pointcloud with '+args.pc_color+' coloring type...')
    # showPointcloudWithMask(depth, floating_mask, 'OBJECTS')
    # print('-> Done!\n')

    # print('Showing overhead reproyection...')
    # showOverheadReproyection(image_path,depth)
    # print('-> Done!\n')


    if args.evalPath:
        print('Evaluating...')
        TP, FP, TN, FN = evaluate(args.evalPath, binary_mask)

        print('Precision =',TP/(TP+FP))
        print('Recall =',TP/(TP+FN))
        print('-> Done!\n')


    

if __name__ == "__main__":
    args = parse_args()
    main(args)