"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import matplotlib.pyplot as plt
import torch
import time
import os
import argparse


from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Own codes
from libs.models import Gen, Vgg16
from libs.utilities import load_model, image_grid, makedir
from libs.dataset import VCTDataset

import libs.pytorch_ssim

#%%

def train(generator, vgg, gOptimizer, epoch, train_loader, device, summarywriter):

    # Enable trainning
    generator.train()

    for step, (data, target) in enumerate(tqdm(train_loader)):

        data = data.to(device)
        target = target.to(device)
        
        # Zero all grads            
        gOptimizer.zero_grad()
            
        # Generate a batch of new images
        gen_data = generator(data)      
      
        # PL
        features_y = vgg(gen_data)
        features_x = vgg(target)
        
        pl4_loss = torch.mean((features_y.relu4_3 - features_x.relu4_3)**2)
        
        loss = pl4_loss 
        
        ### Backpropagation ###
        # Calculate all grads
        loss.backward()
        
        # Update weights and biases based on the calc grads 
        gOptimizer.step()
        
        # ---------------------
        
        # Write Gen Loss to tensorboard
        summarywriter.add_scalar('Gen_Loss/train', 
                                 loss.item(), 
                                 epoch * len(train_loader) + step)
        
        
        # Print images to tensorboard
        if step % 20 == 0:
            summarywriter.add_figure('Plot/train', 
                                     image_grid(data[0,0,:,:], 
                                                target[0,0,:,:], 
                                                gen_data[0,0,:,:]),
                                     epoch * len(train_loader) + step,
                                     close=True)
            # Write Gen SSIM to tensorboard
            summarywriter.add_scalar('Gen_SSIM/train', 
                                     ssim(gen_data, target).item(), 
                                     epoch * len(train_loader) + step)
        
#%%

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description='Restore low-dose mamography')
    ap.add_argument("--rlz", type=int, required=True, 
                    help="Realization number")
    ap.add_argument("--dts", type=int, required=True, 
                    help="Dataset size")

    
    args = vars(ap.parse_args())
    
    rlz = args['rlz']
    dts = args['dts']
        
    # Noise scale factor
    mAsFullDose = 60
    mAsLowDose = 30
    
    red_factor = mAsLowDose / mAsFullDose
    
    path_data = "data/"
    path_models = "final_models/rlz_{}/{}_PL4/".format(rlz,dts)
    path_logs = "final_logs/rlz_{}/{}/{}-{}mAs".format(rlz,dts,time.strftime("%Y-%m-%d-%H%M%S", time.localtime()), mAsLowDose)
    
    
    path_final_generator = path_models + "generator_DBT_VCT-{}mAs.pth".format(mAsLowDose)
    path_final_critic = path_models + "critic_DBT_VCT-{}mAs.pth".format(mAsLowDose)
    
    LR = 1e-4/10
    batch_size = 230
    n_epochs = 60
    
    dataset_path = '{}DBT_VCT_training_{}mAs_{}.h5'.format(path_data,mAsLowDose,dts)
    
    # Tensorboard writer
    summarywriter = SummaryWriter(log_dir=path_logs)
    
    makedir(path_models)
    makedir(path_logs)
    
    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    generator = Gen()
    
    # Create the optimizer and the LR scheduler
    optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.5)    
    
    # Send it to device (GPU if exist)
    generator = generator.to(device)
    # critic = critic.to(device)
    
    # Load gen pre-trained model parameters (if exist)
    start_epoch = load_model(generator, 
                             optimizer, 
                             scheduler,
                             # path_final_model=path_final_generator,
                             path_pretrained_model="final_models/rlz_{}/{}_L1/generator_DBT_VCT-{}mAs.pth".format(rlz,dts,mAsLowDose))

    # Create dataset helper
    train_set = VCTDataset(dataset_path, red_factor=red_factor, vmin=50., vmax= 5000.)
    
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size, 
                                              shuffle=True,
                                              pin_memory=True)
    
    ssim = libs.pytorch_ssim.SSIM(window_size = 11)

    vgg = Vgg16(requires_grad=False).to(device)
        
    # Loop on epochs
    for epoch in range(start_epoch, n_epochs):
        
      print("Epoch:[{}] LR:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    
      # Train the model for 1 epoch
      train(generator,
             vgg,
             optimizer, 
             epoch, 
             train_loader, 
             device, 
             summarywriter) 
    
      # Update LR
      scheduler.step()
    
      # Save the model
      torch.save({
                 'epoch': epoch,
                 'model_state_dict': generator.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 }, path_final_generator)
      
      if (epoch + 1) % 10 == 0:
          # Testing code
          os.system("python main_testing.py --rlz {} --dts {} --typ PL4".format(rlz, dts))
          os.system("python evaluation/MNSE.py")