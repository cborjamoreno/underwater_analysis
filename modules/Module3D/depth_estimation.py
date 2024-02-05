#!/usr/bin/env python3

""" 
depth_estimation.py: Módulo de estimación de profundidad. Permite estimar la profundidad de una imagen y mostrar el colormap de la estimación.
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import cv2

from torchvision import transforms
from modules.Module3D.monoUWNet import networks
from modules.Module3D.monoUWNet.layers import disp_to_depth
from modules.Module3D.monoUWNet.my_utils import *

LIGHT_PURPLE = (213, 184, 255)
DARK_BLUE = (1, 1, 122)

def estimate(image_path):
    """ Function to estimate depth for a single image

    Parameters
    ----------
    image_path : str
        Path to the image to estimate depth on

    Returns
    -------
    depth : numpy array, shape (N,3)
        Estimated depth for each pixel of image_path
    
    """

    if torch.cuda.is_available():
        #device = torch.device("cuda")
        device = "cuda" 
    else:
        device = "cpu"

    model_folder = 'modules/Module3D/monoUWNet/20220908_FLC_all_wo_rhf_FLC_4DS_tiny_sky/models/weights_last'

    print("   Loading model from",model_folder)
    encoder_path = os.path.join(model_folder, "encoder.pth")
    depth_decoder_path = os.path.join(model_folder, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.test_hr_encoder.hrnet18(False)
    encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    para_sum_encoder = sum(p.numel() for p in encoder.parameters())
    
    print("   Loading pretrained decoder")
    depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    para_sum_decoder = sum(p.numel() for p in depth_decoder.parameters())
    depth_decoder.to(device)
    depth_decoder.eval()
    para_sum = para_sum_decoder + para_sum_encoder
    print("encoder has {} parameters".format(para_sum_encoder))
    print("depth_decoder has {} parameters".format(para_sum_decoder))
    print("encoder and depth_ decoder have  total {} parameters".format(para_sum))

    # PREDICTING ON IMAGE IN TURN
    with torch.no_grad():

        # Load image and preprocess
        input_image = pil.open(image_path).convert('RGB')
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        # Saving numpy file
        _ , depth = disp_to_depth(disp, 0.1, 100)
        
        return depth[0,0].squeeze().cpu().numpy()

def showColorMap(depth, image_path, output_path=None):
    """ Function to show the colormapped depth image

    Parameters
    ----------
    depth : numpy array, shape (nrows,ncols)
        Depth estimation
    image_path: str
        Image path
    """
    
    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='jet_r')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    print('Press any key to close')

    img = cv2.imread(image_path)
    nrows,ncols,_ = img.shape

    # Resize colormap
    colormapped_im_resized = cv2.resize(colormapped_im, (ncols,nrows), interpolation = cv2.INTER_AREA)

    cmap = mpl.cm.jet_r
    norm = mpl.colors.Normalize(vmin=np.amin(depth[:,2]), vmax=np.amax(depth[:,2]))

    plt.imshow(colormapped_im_resized)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),label='depth estimation value')
    plt.grid(False)
    plt.axis('off')
    if output_path:
        plt.savefig(output_path)
    plt.show()

        