#!/usr/bin/env python3

""" 
demo_folder.py: Este script hace una segmentación de la masa de agua de las imagenes de la ruta por parámetro, separándola de los elementos de interés de la imagen. También estima la profundidad de la imagen y crea una nube de puntos 3D de la escena 
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
        description='Demo folder')

    parser.add_argument('-p', '--path', type=str,
                        help='Input path', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='Output path')
    parser.add_argument('--colormap', action='store_true',
                        help='Show colormap')
    parser.add_argument('--pointcloud', nargs='*', default=None,
                    metavar=('rotation_axis','rotation_angle (degrees)'),
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
    directory_path = args.path
    if args.pointcloud == []:
        args.pointcloud = ['x', '0']
    for filename in os.listdir(directory_path):
        #check if the file d_r_47_.jpg is in the directory
        if filename == 'd_r_47_.jpg':
            print('AAAAAAAAAAAAAAAd_r_47_.jpg is in the directory')
        if filename.endswith(".jpg") or filename.endswith(".png"):  # add more conditions if there are other image formats
            image_path = os.path.join(directory_path, filename)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if args.output:
                output_path = os.path.join(args.output, filename.split('.')[0])
            else:
                output_path = None

            print(f'Processing {filename}...')

            if args.colormap or args.pointcloud or args.reprojection:
                print('Estimating depth for image:'+image_path+'...')
                depth = estimate(image_path)
                print('-> Done!\n')
                if args.colormap:
                    print('Showing colormap estimation...')
                    showColorMap(depth, image_path, output_path+'/'+filename.split('.')[0]+'_colormap.png' if output_path else None)
                    print('-> Done!\n')
                if args.pointcloud:
                    print('Showing 3D pointcloud')
                    showPointcloud(depth,rotation_axis=args.pointcloud[0],rotation_angle=args.pointcloud[1],degrees=True,img=img)
                if args.reprojection:
                    print('Showing overhead reproyection...')
                    print('Press any key to close the window.')
                    showOverheadReproyection(image_path,depth, output_path+'/'+filename.split('.')[0]+'_reprojection.png' if output_path else None)
                    print('-> Done!\n')

            if args.superpixels_seg:
                print('Generating binary superpixels segmentation...')
                showBinarySegmentationSuperpixels(image_path, output_path+'/'+filename.split('.')[0]+'_superpixels.png' if output_path else None)
                print('-> Done!\n')
            if args.depth_seg:
                print('Generating binary depth segmentation...')
                showBinarySegmentationDepth(image_path, output_path+'/'+filename.split('.')[0]+'_depth.png' if output_path else None)
                print('-> Done!\n')
            if args.SAM:
                print('Generating SAM result...')
                masks = segmentationSAM(img)
                showSAM(img,masks,output_path+'/'+filename.split('.')[0]+'_SAM.png' if output_path else None)
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

                print(f'Finished processing {filename}\n')

if __name__ == "__main__":
    args = parse_args()
    main(args)