import cv2 as cv
import numpy as np
import argparse

def parse_args():
    """
    Argument parsing
    
    """
    
    parser = argparse.ArgumentParser(
        description='Segmentation between floor, water and floating objects')

    parser.add_argument('--seg_mask', type=str,
                        help='Path of Semantic Segmentation of Underwater Imagery\'s dataset segmentation mask', required=True)
    parser.add_argument('--mask', type=str,
                        help='Binary segmentation mask', required=True)
    
    return parser.parse_args()

def main(args):

    # Load masks
    seg_mask = cv.imread(args.seg_mask)
    seg_mask = cv.cvtColor(seg_mask, cv.COLOR_BGR2RGB)
    mask = cv.imread(args.mask)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    
    # cv.imshow('seg',seg_mask)
    # cv.imshow('mask',mask)
    # cv.waitKey(0)
    

    c = (0,0,0) # Black
    indices = np.where(np.all(seg_mask == c, axis=-1))
    water_GT = list(zip(indices[0], indices[1]))
    
    indices = np.where(np.any(seg_mask != c, axis=-1))
    not_water_GT = list(zip(indices[0], indices[1]))

    light_purple = (213, 184, 255)
    dark_blue = (1, 1, 122)

    TP = 0
    FP = 0
    FN = 0
    TN = 0



    for p in water_GT:
        if mask[p[0],p[1],:].tolist() == list(light_purple):
            TP +=1
        else:
            FN +=1
    
    for p in not_water_GT:
        if mask[p[0],p[1],:].tolist() == list(dark_blue):
            TN +=1
        else:
            FP +=1


    print('precision =',TP/(TP+FP))
    print('recall =',TP/(TP+FN))
    
    


if __name__ == "__main__":
    args = parse_args()
    main(args)