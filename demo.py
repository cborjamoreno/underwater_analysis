#!/usr/bin/env python3

""" 
demo.py: Este script hace una segmentación de la masa de agua de la imagen pasada por parámetro, separándola de los elementos de interés de la imagen. También estima la profundidad de la imagen y crea una nube de puntos 3D de la escena 
"""


import argparse

from DepthEstimation.monoUWNet.depth_estimation import estimate, showColorMap
from Segmentation.segmentation import segmentationSAM, showSegmentations, binarySegmentationSuperpixels

def parse_args():
    """
    Argument parsing
    
    """
    
    parser = argparse.ArgumentParser(
        description='Demo')

    parser.add_argument('-p', '--path', type=str,
                        help='Input image path', required=True)
    parser.add_argument('--pc_color', 
                        help='Point cloud colors.', choices=['depth', 'binary', 'objects'], default='objects')
    parser.add_argument('-eval', '--evalPath', type=str,
                        help='Labeled image path')
    
    return parser.parse_args()

def main(args):
    image_path = args.path

    depth = estimate(image_path)
    # showColorMap(depth)

    binary_mask, color_mask = segmentationSAM(depth, image_path)
    showSegmentations(binary_mask, depth, args.pc_color, color_mask)

    #TODO: evaluar con algoritmo de evaluacion

if __name__ == "__main__":
    args = parse_args()
    main(args)