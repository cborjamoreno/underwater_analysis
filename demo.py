#!/usr/bin/env python3

""" 
demo_folder.py: Este script hace una segmentación de la masa de agua de las imagenes de la ruta por parámetro, separándola de los elementos de interés de la imagen. También estima la profundidad de la imagen y crea una nube de puntos 3D de la escena 
"""


import argparse
import time
import pandas as pd

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
    parser.add_argument('--SAM', action='store_true',
                        help='Show SAM result')
    parser.add_argument('--pc_color', 
                        help='Point cloud colors.', choices=['DEPTH', 'FLOATING'], default='FLOATING')
    parser.add_argument('-eval', '--evalPath', type=str,
                        help='Evaluation segmented mask path. Water pixels must be labeled as RGB (0,0,0) in mask located in \'evalPath\'.')
    
    return parser.parse_args()

def main(args):
    if args.pointcloud == []:
        args.pointcloud = ['x', '0']
    
    times = [0, 0, 0]
    masks = [None, None, None]
    if args.evalPath:
        algorithmsToEvaluate = [args.superpixels_seg, args.depth_seg, True]
        algorithmsNames = ['superpixel', 'depth', 'depthSAM']
        pandas = [pd.DataFrame(columns=['File', 'Precision', 'Recall', 'Execution time (s)']) for _ in range(3)]
        eval_files = os.listdir(args.evalPath)

    # if args.path is a directory, process all images in the directory:
    if os.path.isdir(args.path):
        filenames = os.listdir(args.path)
        directory_path = args.path
    else:
        filenames = [args.path.split('/')[-1]]
        directory_path = os.path.dirname(args.path)
            
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):  # add more conditions if there are other image formats
            image_path = os.path.join(directory_path, filename)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if args.output:
                output_path = os.path.join(args.output, filename.split('.')[0])
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
            else:
                output_path = None

            print(f'Processing {filename}...')

            if args.colormap or args.pointcloud or args.reprojection:
                print('Estimating depth for image:'+image_path+'...')
                depth = estimate(image_path)
                print('-> Done!\n')
                if args.colormap:
                    print('Showing colormap estimation...')
                    showColorMap(depth, image_path, output_path+'/'+filename.split('.')[0]+'colormap.png' if output_path else None)
                    print('-> Done!\n')
                if args.pointcloud:
                    # Convert degrees to float
                    angle = float(args.pointcloud[1]) * np.pi / 180
                    print('Showing 3D pointcloud')
                    showPointcloud(depth,rotation_axis=args.pointcloud[0],rotation_angle=angle,degrees=True,img=img)
                if args.reprojection:
                    print('Showing overhead reproyection...')
                    print('Press any key to close the window.')
                    showOverheadReproyection(image_path,depth, output_path+'/'+filename.split('.')[0]+'reprojection.png' if output_path else None)
                    print('-> Done!\n')

            if args.superpixels_seg:
                print('Generating binary superpixels segmentation...')
                times[0] = time.time()
                masks[0] = showBinarySegmentationSuperpixels(image_path, output_path+'/'+filename.split('.')[0]+'superpixels.png' if output_path else None)
                # show masks[0]
                # cv2.imshow('Superpixels', masks[0])

                times[0] = time.time() - times[0]
                print('-> Done!\n')
            if args.depth_seg:
                print('Generating binary depth segmentation...')
                times[1] = time.time()
                masks[1] = showBinarySegmentationDepth(image_path, output_path+'/'+filename.split('.')[0]+'depth.png' if output_path else None)
                times[1] = time.time() - times[1]
                print('-> Done!\n')
            if args.SAM:
                print('Generating SAM result...')
                SAM_masks = segmentationSAM(img)
                showSAM(img,SAM_masks,output_path+'/'+filename.split('.')[0]+'_SAM.png' if output_path else None)
                print('-> Done!\n')

            print('Generating binary and object segmentation...')
            output_paths = [None, None, None]
            if output_path:
                output_paths.append(output_path+'/'+filename.split('.')[0]+'three_class.png' if output_path else None)
                output_paths.append(output_path+'/'+filename.split('.')[0]+'objects.png' if output_path else None)
                output_paths.append(output_path+'/'+filename.split('.')[0]+'pointcloud.png' if output_path else None)
                
            times[2] = time.time()
            binary_mask, three_mask, color_mask = segmentationFinal(image_path,args.pc_color,output_paths)
            masks[2] = binary_mask
            times[2] = time.time() - times[2]

            print('-> Done!\n')

            if args.evalPath:
                print('Evaluating...')

                # print('Precision =',TP/(TP+FP))
                # print('Recall =',TP/(TP+FN))

                eval_file = eval_files[eval_files.index(filename.split('.')[0]+'.bmp')]
                eval_path = os.path.join(args.evalPath, eval_file)

                # print('masks:', masks)

                for i, eval in enumerate(algorithmsToEvaluate):
                    if eval:
                        TP, FP, TN, FN = evaluate(eval_path, masks[i])
                        prec = round(TP/(TP+FP), 2) if TP else 0
                        rec = round(TP/(TP+FN), 2) if TP else 0
                        new_row = pd.DataFrame([[filename, prec, rec, round(times[i],2)]], columns=pandas[i].columns)
                        pandas[i] = pd.concat([pandas[i], new_row], ignore_index=True)

                # # Save precision, recall and execution time of every image to a file with pandas
                # results = pd.DataFrame(columns=['Precision', 'Recall', 'Execution time'])

                # results.to_csv('evaluation.csv', index=False)

                print('-> Done!\n')

                print(f'Finished processing {filename}\n')
    
    if args.evalPath:
        # Add mean values to pandas
        for i, eval in enumerate(algorithmsToEvaluate):
            if eval:
                # Add mean row
                mean = pd.DataFrame([['Mean', round(pandas[i]['Precision'].mean(), 2), round(pandas[i]['Recall'].mean(), 2), round(pandas[i]['Execution time (s)'].mean(), 2)]], columns=pandas[i].columns)
                pandas[i] = pd.concat([pandas[i], mean], ignore_index=True)
                # Add median row
                median = pd.DataFrame([['Median', round(pandas[i]['Precision'].median(), 2), round(pandas[i]['Recall'].median(), 2), round(pandas[i]['Execution time (s)'].median(), 2)]], columns=pandas[i].columns)
                pandas[i] = pd.concat([pandas[i], median], ignore_index=True)
                # Save pandas to file
                pandas[i].to_csv('evaluation_'+algorithmsNames[i]+'.csv', index=False)
                

if __name__ == "__main__":
    args = parse_args()
    main(args)