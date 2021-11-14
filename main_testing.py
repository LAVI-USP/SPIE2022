"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import pydicom
import os
import argparse
import pathlib

from tqdm import tqdm

# Own codes
from libs.models import Gen
from libs.utilities import load_model, makedir
from libs.dataset import scale, de_scale


#%%
def img2rois(img_ld):
    
    h, w = img_ld.shape
    
    # How many Rois fit in the image?
    n_h = h % 64
    n_w = w % 64
    
    if n_h == 0:
        h_pad = h
    else:
        h_pad = (h//2 + 1) * 64
        
    if n_w == 0:
        w_pad = w
    else:
        w_pad = (w//2 + 1) * 64
    
    # Calculate how much padding is necessary and sum 64 for the frontiers
    padding = (((h_pad - h)//2 + 64, (h_pad - h)//2 + 64),
               ((w_pad - w)//2 + 64, (w_pad - w)//2 + 64))
    
    # Pad the image
    img_ld_pad = np.pad(img_ld, padding, mode='reflect')
            
    n_h = h_pad // 64 
    n_w = w_pad // 64 
    
    # Allocate memory to speed up the for loop
    rois = np.empty((n_h*n_w, 1, 192, 192), dtype='float32')

    nRoi = 0
    # Get the ROIs
    for i in range(n_h):
        for j in range(n_w):
            rois[nRoi, 0, :, :] = img_ld_pad[i*64: (i+3)*64, j*64:(j+3)*64]
            nRoi += 1
            
    return rois, img_ld_pad.shape
            
def rois2img(rst_rois, original_shape, padded_shape):

    
    rst_img = np.empty((padded_shape))
    
    n_h = (padded_shape[0] // 64) - 2
    n_w = (padded_shape[1] // 64) - 2
      
    nRoi = 0
    # Reconstruct image format
    for i in range(n_h):
        for j in range(n_w):
            rst_img[(i+1)*64:(i+2)*64, (j+1)*64:(j+2)*64] = rst_rois[nRoi,0,64:128,64:128]
            nRoi += 1
        
    org_h, org_w = original_shape
    pad_h, pad_w = padded_shape
    
    # How much to crop?
    start_w = (pad_w - org_w) // 2
    start_h = (pad_h - org_h) // 2
    
    # Crop image
    rst_img = rst_img[start_h:start_h+org_h, start_w:start_w+org_w]
        
    return rst_img
         
def model_forward(model, img_ld, red_factor, batch_size):
    
    # Change model to eval
    model.eval()

    # Normalize image
    img_ld = scale(img_ld, red_factor=red_factor, vmin=50)    

    # Extract ROIs
    rois, padded_shape = img2rois(img_ld)
    
    # Allocate memory to speed up the for loop 
    rst_rois = np.empty_like(rois)
    
    for x in range(0,rois.shape[0],batch_size):
        
        # Get the batch and send to GPU
        batch = torch.from_numpy(rois[x:x+batch_size]).to(device)
        
        # Forward through the model
        with torch.no_grad():
            batch = model(batch)
        
        # Get from GPU
        rst_rois[x:x+batch_size] = batch.to('cpu').numpy()

    # Contruct the image
    rst_img = rois2img(rst_rois, img_ld.shape, padded_shape)
    
    # Normalize image (Inv)
    rst_img = de_scale(rst_img)
    
    return rst_img

def test(model, path_data, path2write, red_factor, mAsLowDose, batch_size):
    
    path_data_ld = path_data + '31_' + str(30)
    
    file_names = list(pathlib.Path(path_data_ld).glob('**/*.dcm'))

    for file_name in tqdm(file_names):
        
        file_name = str(file_name) 
        
        # Read dicom image
        dcmH = pydicom.dcmread(file_name)

        # Read dicom image pixels
        img_ld = dcmH.pixel_array.astype('float32')
     
        # Forward through model
        rst_img = model_forward(model, img_ld, red_factor, batch_size)

        folder_name = path2write + 'DBT_DL_ResNet' + file_name.split('/')[-2] 
        file2write_name = 'DL_ResNet' + file_name.split('/') [-1]
        
        # Create output dir (if needed)
        makedir(folder_name)
        
        # Copy the restored data to the original dicom header
        dcmH.PixelData = np.uint16(rst_img)
        
        # Write dicom
        pydicom.dcmwrite(os.path.join(folder_name,file2write_name),
                         dcmH, 
                         write_like_original=True) 

    return

#%%

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description='Restore low-dose mamography')
    ap.add_argument("--rlz", type=int, required=True, 
                    help="Realization number")
    ap.add_argument("--dts", type=int, required=True, 
                    help="Dataset size")
    ap.add_argument("--typ", type=str, required=True, 
                    help="Loss type")

    
    args = vars(ap.parse_args())
    
    rlz = args['rlz']
    dts = args['dts']
    typ = args['typ']
        
    # Noise scale factor
    mAsFullDose = 60
    mAsLowDose = 30
    
    batch_size = 50
    
    red_factor = mAsLowDose / mAsFullDose
    
    path_data = "Imgs"
    path_models = "final_models/rlz_{}/{}_{}/".format(rlz,dts,typ)
    path2write = "outputs"
    
    makedir(path2write)
    
    path_final_generator = path_models + "generator_DBT_VCT-{}mAs.pth".format(mAsLowDose)
    
    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    generator = Gen()
    
    # Send it to device (GPU if exist)
    generator = generator.to(device)
    
    # Load gen pre-trained model parameters (if exist)
    _ = load_model(generator,path_final_model=path_final_generator)
    
    print("Running test on {}.".format(device))
    
    test(generator, path_data, path2write, red_factor, mAsLowDose, batch_size)
    
