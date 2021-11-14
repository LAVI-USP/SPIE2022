"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import numpy as np
import pydicom as dicom
import h5py
import random
from pathlib import Path

#%%

def get_img_bounds(img):
    '''Get image bounds of the segmented breast from the GT'''
    
    # Segment the breast
    mask = img < 4000
    
    # Height bounds
    mask_h = np.sum(mask, 1) > 0
    res = np.where(mask_h == True)
    h_min, h_max = res[0][0], res[0][-1]
    
    # Weight bounds
    mask_w = np.sum(mask, 0) > 0
    res = np.where(mask_w == True)
    w_min, w_max = res[0][0], res[0][-1]
        
    return w_min, h_min, w_max, h_max


def extract_rois(img_ld, img_fd, img_gt):
    '''Extract low-dose and full-dose rois'''
    
    # Check if images are the same size
    assert img_ld.shape == img_fd.shape == img_gt.shape, "image sizes differ"
    
    global trow_away
    
    # Get image bounds of the segmented breast from the GT
    w_min, h_min, w_max, h_max = get_img_bounds(img_gt)
    
    # Crop all images
    img_ld = img_ld[h_min:h_max, w_min:w_max]
    img_fd = img_fd[h_min:h_max, w_min:w_max]
    img_gt = img_gt[h_min:h_max, w_min:w_max]
    
    # Get updated image shape
    w, h = img_ld.shape
    
    rois = []
    
    # Non-overlaping roi extraction
    for i in range(0, w-64, 64):
        for j in range(0, h-64, 64):
            
            # Extract roi
            roi_tuple = (img_ld[i:i+64, j:j+64], img_fd[i:i+64, j:j+64])
            roi_gt = img_gt[i:i+64, j:j+64]
            
            # Am I geting at least one pixel from the breast?
            if np.sum(roi_gt > 4000) != 64*64:
                rois.append(roi_tuple)
            else:
                trow_away += 1                

    return rois


def process_each_folder(folder_name, num_proj=15):
    '''Process DBT folder to extract low-dose and full-dose rois'''
        
    noisy_path = path2read + '/noisy/' + folder_name.split('/')[-1]
    
    rlz = 1
    
    global nt_imgs
    
    rois = []
    
    # Loop on each projection
    for proj in range(num_proj):
        
        # GT with corrected mean value
        gt_file_name = noisy_path + '-CM/_{}.dcm'.format(proj)
        
        # Low-dose image
        ld_file_name = noisy_path + '-{}mAs-rlz{}/_{}.dcm'.format(lowDosemAs,rlz,proj)

        # Full-dose image
        fd_file_name = noisy_path + '-{}mAs-rlz{}/_{}.dcm'.format(fullDosemAs,rlz,proj)
    
        img_ld = dicom.read_file(ld_file_name).pixel_array
        img_fd = dicom.read_file(fd_file_name).pixel_array
        img_gt = dicom.read_file(gt_file_name).pixel_array
    
        rois += extract_rois(img_ld, img_fd, img_gt)
                    
    return rois

#%%

if __name__ == '__main__':
    
    path2read = 'D:/Rod/VCT_Penn_2020/Hologic-projs'
    path2write = '../data/'
    
    folder_names = [str(item) for item in Path(path2read).glob("*-proj") if Path(item).is_dir()]
    
    random.shuffle(folder_names)
    
    fullDosemAs = 60 

    lowDosemAs = 30 
    
    nROIs_total = 128000 
    
    np.random.seed(0)
    
    trow_away = 0
    flag_final = 0
    nROIs = 0
    
    # Create h5 file
    f = h5py.File('{}DBT_VCT_training_{}mAs.h5'.format(path2write, lowDosemAs), 'a')
    
    # Loop on each DBT folder (projections)
    for idX, folder_name in enumerate(folder_names):
        
        # Get low-dose and full-dose rois
        rois = process_each_folder(folder_name)        
                
        data = np.stack([x[0] for x in rois])
        target = np.stack([x[1] for x in rois])
        
        data = np.expand_dims(data, axis=1) 
        target = np.expand_dims(target, axis=1) 
        
        nROIs += data.shape[0]
        
        # Did I reach the expected size (nROIs_total)?
        if  nROIs >= nROIs_total:
            flag_final = 1
            diff = nROIs_total - nROIs
            data = data[:diff,:,:,:]
            target = target[:diff,:,:,:]
                            
        if idX == 0:
            f.create_dataset('data', data=data, chunks=True, maxshape=(None,1,64,64))
            f.create_dataset('target', data=target, chunks=True, maxshape=(None,1,64,64)) 
        else:
            f['data'].resize((f['data'].shape[0] + data.shape[0]), axis=0)
            f['data'][-data.shape[0]:] = data
            
            f['target'].resize((f['target'].shape[0] + target.shape[0]), axis=0)
            f['target'][-target.shape[0]:] = target
            
        print("Iter {} and 'data' chunk has shape:{} and 'target':{}".format(idX,f['data'].shape,f['target'].shape))

        if flag_final:
            break

    f.close()       
     
    
    