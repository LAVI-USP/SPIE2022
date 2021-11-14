"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import os 
import torch
import matplotlib.pyplot as plt

def load_model(model, optimizer=None, scheduler=None, path_final_model='', path_pretrained_model=''):
    """Load pre-trained model, resume training or initialize from scratch."""
    
    epoch = 0
      
    # Resume training
    if os.path.isfile(path_final_model):
          
      checkpoint = torch.load(path_final_model)
      model.load_state_dict(checkpoint['model_state_dict'])
      if optimizer != None:
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      if scheduler != None:
          scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      epoch = checkpoint['epoch'] + 1
      
      print('Loading model {} from epoch {}.'.format(path_final_model, epoch-1))
      
    # Loading pre-trained model
    elif os.path.isfile(path_pretrained_model):
          
      # Load a pre trained network 
      checkpoint = torch.load(path_pretrained_model)
      model.load_state_dict(checkpoint['model_state_dict'])
      
      print('Initializing from scratch \nLoading pre-trained {}.'.format(path_pretrained_model))
      
    # Initializing from scratch
    else:
      print('I couldnt find any model, I am just initializing from scratch.')
      
    return epoch


def image_grid(ld_img, hd_img, rt_img):
    """Return a 1x3 grid of the images as a matplotlib figure."""
    
    # Get from GPU
    ld_img = ld_img.to('cpu')
    hd_img = hd_img.to('cpu')
    rt_img = rt_img.to('cpu').detach()
    
    # Create a figure to contain the plot.
    figure = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(torch.squeeze(ld_img),'gray')
    plt.title("Low dose"); plt.grid(False)
    
    plt.subplot(1,3,2)
    plt.imshow(torch.squeeze(hd_img),'gray')
    plt.title("Full dose"); plt.grid(False)
    
    plt.subplot(1,3,3)
    plt.imshow(torch.squeeze(rt_img),'gray')
    plt.title("Restored dose"); plt.grid(False)
      
    return figure

def makedir(path2create):
    """Create directory if it does not exists."""
 
    error = 1
    
    if not os.path.exists(path2create):
        os.makedirs(path2create)
        error = 0
    
    return error

def readDicom(dir2Read, imgSize):
    """Read Dicom function."""
      
    # List dicom files
    dcmFiles = list(pathlib.Path(dir2Read).glob('*.dcm'))

    dcmImg = np.empty([imgSize[0],imgSize[1],len(dcmFiles)])

    if not dcmFiles:    
        raise ValueError('No DICOM files found in the specified path.')

    for dcm in dcmFiles:
        
        dcmH = pydicom.dcmread(str(dcm))
        
        ind = int(str(dcm).split('/')[-1].split('_')[-1].split('.')[0])
        
        dcmImg[:,:,ind] = dcmH.pixel_array[130:-130,50:-50].astype('float32')  

    return dcmImg

