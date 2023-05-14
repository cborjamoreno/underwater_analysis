import cv2
import numpy as np
import argparse
import os,sys
import glob

def parse_args():
    """
    Argument parsing
    
    """
    
    parser = argparse.ArgumentParser(
        description='Segmentation between floor, water and floating objects')

    parser.add_argument('-p', '--path', type=str,
                        help='binary image path', required=True)
    parser.add_argument('-out', '--output_path', type=str,
                        help='dir path to save output figures', required=False)
    
    return parser.parse_args()

def main(args):


    # FINDING INPUT POINTS
    if os.path.isfile(args.path):
        # Only use a single pointset
        paths = [args.path]
    elif os.path.isdir(args.path):
        # Searching folder for pointsets
        paths = glob.glob(os.path.join(args.path,'**/*.{}'.format('png')), recursive=True)
    else:
        raise Exception("Can not find args.path: {}".format(args.path))
        
    # Checking output path
    if args.output_path:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        
        out_path = args.output_path
        if out_path[len(out_path)-1] != '/':
            # Append '/' to output path
            out_path += '/'

    for path in paths:

        # Load image
        img = cv2.imread(path)

        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find Canny edges
        edged = cv2.Canny(gray, 30, 200)
        
        # Finding Contours
        contours, hierarchy = cv2.findContours(edged, 
            cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        result = img.copy()

        for i in range(len(contours)):

            # Calculate contour's area
            area = cv2.contourArea(contours[i])

            # Check if contour is closed
            if hierarchy[0][i][2] > 0 and area < 40000.0:

                # Draw and fill contour 
                cv2.drawContours(result, [contours[i]], 0, (90,128,0), -1)

        # Show result
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        sample = os.path.splitext(path)
        sample = os.path.splitext(sample[0])
        sample = (sample[0].split('/'))[2]

        if args.output_path:
            cv2.imwrite(out_path+sample+"_segmentation.png", result)


if __name__ == "__main__":
    args = parse_args()
    main(args)