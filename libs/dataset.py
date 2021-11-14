"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import torch
import h5py


def de_scale(data, vmax):
    
    data = data * vmax 
        
    return data

def scale(data, vmin, vmax, red_factor=1.):
    
    data -= vmin
    data /= red_factor
    data += vmin
    data /= vmax
    
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
          
    return data

class VCTDataset(torch.utils.data.Dataset):
  """ Virtual Clinical Trial for DBT dataset."""
  def __init__(self, h5_file_name, red_factor, vmin, vmax, normalize=True):
    """
    Args:
      h5_file_name (string): Path to the h5 file.
      red_factor: Reduction factor of data
    """
    self.h5_file_name = h5_file_name
    self.red_factor = red_factor
    self.vmin = vmin
    self.vmax = vmax

    self.h5_file = h5py.File(self.h5_file_name, 'r')

    self.data = self.h5_file['data']
    self.target = self.h5_file['target']
    
    self.normalize = normalize
    

  def __len__(self):
      
    return self.data.shape[0]

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    data = self.data[idx,:,:]
    target = self.target[idx,:,:]
    
    # To torch tensor
    data = torch.from_numpy(data.astype(float)).type(torch.FloatTensor)
    target = torch.from_numpy(target.astype(float)).type(torch.FloatTensor)
    
    # Normalize 0-1 data
    if self.normalize:
        data = scale(data, self.vmin, self.vmax, red_factor=self.red_factor)
        target = scale(target, self.vmin, self.vmax)
    

    return data, target

