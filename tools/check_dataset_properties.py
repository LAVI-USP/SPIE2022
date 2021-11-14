"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import torch
import sys

from tqdm import tqdm 

sys.path.insert(0, '../')

from libs.dataset import VCTDataset


if __name__ == '__main__':
        
    # Noise scale factor
    mAsFullDose = 90
    mAsLowDose = 45
    
    red_factor = mAsLowDose / mAsFullDose
    
    path_data = "../data/"
    
    dataset_path = '{}DBT_VCT_training_{}mAs.h5'.format(path_data,mAsLowDose)
    
    # Create dataset helper
    train_set = VCTDataset(dataset_path, red_factor, normalize=False)
    
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=1000, 
                                              shuffle=True,
                                              pin_memory=True)
    
    data_min = 2**16
    data_max = 0
    target_min = 2**16
    target_max = 0
    
    for step, (data, target) in enumerate(tqdm(train_loader)):
        
        data_min_batch = data.min()
        data_max_batch = data.max()
        
        target_min_batch = target.min()
        target_max_batch = target.max()
        
        if(data_min_batch < data_min):
            data_min = data_min_batch
        if(data_max_batch > data_max):
            data_max = data_max_batch
            
        if(target_min_batch < target_min):
            target_min = target_min_batch
        if(target_max_batch > target_max):
            target_max = target_max_batch
            
    print(data_min, data_max) 
    print(target_min, target_max) 