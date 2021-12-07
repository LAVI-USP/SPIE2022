"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pydicom
import pandas as pd
import sys

sys.path.insert(1,'..')
sys.path.insert(1,'.')
import libs.pyeval

from libs.utilities import readDicom
from scipy.ndimage import binary_erosion
from skimage.morphology import disk

#%% General parameters

path2Read = 'data'

DL_types = ['ResNet']


#%% Image parameters

nRows = 1788    
nCols = 1564 

mAsFullDose = 60
reducFactors = [50]

n_all_fullDose = 20
n_rlzs_fullDose = 10
n_rlzs_GT = 10
DL_ind = 0
       
#%% Read clinical data

myInv = lambda x : np.array((1./x[0],-x[1]/x[0]))

nreducFactors = len(reducFactors)

mnse_RD = np.empty(shape=(nreducFactors,3))
resNoiseVar_RD = np.empty(shape=(nreducFactors,3))
bias2_RD = np.empty(shape=(nreducFactors,3))

mnse_MB = np.empty(shape=(nreducFactors,3))
resNoiseVar_MB = np.empty(shape=(nreducFactors,3))
bias2_MB = np.empty(shape=(nreducFactors,3))

mnse_DL = np.empty(shape=(nreducFactors,3))
resNoiseVar_DL = np.empty(shape=(nreducFactors,3))
bias2_DL = np.empty(shape=(nreducFactors,3))

# Read all full dose images
fullDose_all = np.empty(shape=(nRows,nCols,15,n_all_fullDose))

paths = sorted(list(pathlib.Path(path2Read + "31_" + str(mAsFullDose) ).glob('*/')))

if paths == []:
    raise ValueError('No FD results found.') 

print('Reading FD images...')
for idX, path in enumerate(paths):
    fullDose_all[:,:,:,idX] = readDicom(path,(nRows,nCols))
       
# Generate the GT - fisrt time
groundTruth = np.mean(fullDose_all[:,:,:,n_rlzs_GT:], axis=-1)

# Generate a boolean mask for the breast
maskBreast = groundTruth < 2500

# for s in range(maskBreast.shape[-1]):
#     maskBreast[:,:,s] = binary_erosion(maskBreast[:,:,s],disk(39,dtype=bool))

# Normalize the GT realizations
for z in range(n_rlzs_GT, n_all_fullDose):
    for p in range(15):
        unique_rlzs = fullDose_all[:,:,p,z]
        unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
        fullDose_all[:,:,p,z] = np.reshape(unique_rlzs, (nRows,nCols))
    
# Generate again the GT after normalization
groundTruth = np.mean(fullDose_all[:,:,:,n_rlzs_GT:], axis=-1)

# Normalize the full dose realizations
for z in range(n_rlzs_GT):
    for p in range(15):
        unique_rlzs = fullDose_all[:,:,p,z]
        unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
        fullDose_all[:,:,p,z] = np.reshape(unique_rlzs, (nRows,nCols))
              

# Apply breask mask on both full dose anf GT realizations
groundTruth_rlzs = np.expand_dims(fullDose_all[:,:,:,n_rlzs_GT:][maskBreast], axis=0)
fullDose_rlzs = np.expand_dims(fullDose_all[:,:,:,:n_rlzs_GT][maskBreast], axis=0)

# mnse_FD, resNoiseVar_FD, bias2_FD, _ = pyeval.MNSE(groundTruth_rlzs, fullDose_rlzs)

# df = pd.DataFrame(np.array(((["{:.2f} [{:.2f}, {:.2f}]".format(100*mnse_FD[0], 100*mnse_FD[1], 100*mnse_FD[2]),"{:.2f} [{:.2f}, {:.2f}]".format(100*resNoiseVar_FD[0], 100*resNoiseVar_FD[1], 100*resNoiseVar_FD[2]),"{:.2f} [{:.2f}, {:.2f}]".format(100*bias2_FD[0], 100*bias2_FD[1], 100*bias2_FD[2])]),
#                             ),ndmin=2),
#                               columns=['Total MNSE', 'Residual-Noise', 'Bias-Squared'],
#                               index=["Full Dose"])
        
# df.to_csv(r'outputs.txt', sep=' ', header=None, mode='a')

del fullDose_all, fullDose_rlzs

# Read MB, restored results and reuced doses
# for reduc in reducFactors:
    
#     mAsReducFactors = int((reduc / 100) * mAsFullDose)
    
#     # Reduced doses
#     paths = list(pathlib.Path(path2Read + "31_" + str(mAsReducFactors) ).glob('*')) 
#     if paths == []:
#         raise ValueError('No RD results found.')
#     paths = [path for path in paths if not 'MB' in str(path) and not 'DL' in str(path)]
#     reduDose_rlzs = np.empty(shape=(nRows,nCols,15,n_rlzs_fullDose))
#     for idX, path in enumerate(paths):
#         all_rlzs =  readDicom(path,(nRows,nCols))
#         for p in range(15):
#             # Reduced doses
#             unique_rlzs = all_rlzs[:,:,p]
#             unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
#             reduDose_rlzs[:,:,p,idX] = np.reshape(unique_rlzs, (nRows,nCols))
            
#     # reduDose[reduc] = reduDose_rlzs


#     # MB restored doses
#     paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors) ).glob('MB*')) 
#     if paths == []:
#         raise ValueError('No MB results found.')
#     restDose_MB_rlzs = np.empty(shape=(nRows,nCols,15,n_rlzs_fullDose))
#     for idX, path in enumerate(paths):
#         all_rlzs =  readDicom(path,(nRows,nCols))
#         for p in range(15):
#             # MB restored doses
#             unique_rlzs = all_rlzs[:,:,p]
#             unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
#             restDose_MB_rlzs[:,:,p,idX] = np.reshape(unique_rlzs, (nRows,nCols))
        
#     # restDose_MB[reduc] = restDose_MB_rlzs 
    
#     # Calculations for FD, MB and RD
#     for idX, reduc in enumerate(reducFactors): 
#         mnse_RD[idX,:], resNoiseVar_RD[idX,:], bias2_RD[idX,:], _= pyeval.MNSE(groundTruth_rlzs, np.expand_dims(reduDose_rlzs[maskBreast], axis=0))
#         mnse_MB[idX,:], resNoiseVar_MB[idX,:], bias2_MB[idX,:], _= pyeval.MNSE(groundTruth_rlzs, np.expand_dims(restDose_MB_rlzs[maskBreast], axis=0))
       
#         df = pd.DataFrame(np.array(((["{:.2f} [{:.2f}, {:.2f}]".format(100*mnse_MB[idX,0], 100*mnse_MB[idX,1], 100*mnse_MB[idX,2]),"{:.2f} [{:.2f}, {:.2f}]".format(100*resNoiseVar_MB[idX,0], 100*resNoiseVar_MB[idX,1], 100*resNoiseVar_MB[idX,2]),"{:.2f} [{:.2f}, {:.2f}]".format(100*bias2_MB[idX,0], 100*bias2_MB[idX,1], 100*bias2_MB[idX,2])]),
#                                     (["{:.2f} [{:.2f}, {:.2f}]".format(100*mnse_RD[idX,0], 100*mnse_RD[idX,1], 100*mnse_RD[idX,2]),"{:.2f} [{:.2f}, {:.2f}]".format(100*resNoiseVar_RD[idX,0], 100*resNoiseVar_RD[idX,1], 100*resNoiseVar_RD[idX,2]),"{:.2f} [{:.2f}, {:.2f}]".format(100*bias2_RD[idX,0], 100*bias2_RD[idX,1], 100*bias2_RD[idX,2])]),
#                                     ),ndmin=2),
#                           columns=['Total MNSE', 'Residual-Noise', 'Bias-Squared'],
#                           index=["Model-Based","Noisy-{}mAs".format(mAsReducFactors)])
        
#     df.to_csv(r'outputs.txt', sep=' ', header=None, mode='a')
    
# del reduDose_rlzs, restDose_MB_rlzs, unique_rlzs, all_rlzs


# %% MNSE calculation

# Loop through DL methods
for indDL, DL_type in enumerate(DL_types):
    
    # Read DL 
    for idX, reduc in enumerate(reducFactors): 
        
        mAsReducFactors = int((reduc / 100) * mAsFullDose)
        
        print('Reading and calculating {}({}mAs) images...'.format(DL_type,mAsReducFactors))
        
        # DL restored doses
        paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors) ).glob('DBT_DL_' + DL_type + '*')) 
        if paths == []:
            raise ValueError('No DL results found.')
        restDose_DL_rlzs = np.empty(shape=(nRows,nCols,15,n_rlzs_fullDose))
        for idZ, path in enumerate(paths):
            all_rlzs =  readDicom(path,(nRows,nCols))
            for p in range(15):            
                # DL restored doses
                unique_rlzs = all_rlzs[:,:,p]
                unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
                restDose_DL_rlzs[:,:,p,idZ]  = np.reshape(unique_rlzs, (nRows,nCols))
                              

        mnse_DL[idX,:], resNoiseVar_DL[idX,:], bias2_DL[idX,:], _= pyeval.MNSE(groundTruth_rlzs, np.expand_dims(restDose_DL_rlzs[maskBreast], axis=0))
        
        
        df = pd.DataFrame(np.array(["{:.2f} [{:.2f}, {:.2f}]".format(100*mnse_DL[idX,0], 100*mnse_DL[idX,1], 100*mnse_DL[idX,2]),
                            "{:.2f} [{:.2f}, {:.2f}]".format(100*resNoiseVar_DL[idX,0], 100*resNoiseVar_DL[idX,1], 100*resNoiseVar_DL[idX,2]),
                            "{:.2f} [{:.2f}, {:.2f}]".format(100*bias2_DL[idX,0], 100*bias2_DL[idX,1], 100*bias2_DL[idX,2])], ndmin=2),
                          columns=['Total MNSE', 'Residual-Noise', 'Bias-Squared'],
                          index=["DL-{}-{}mAs".format(DL_type,mAsReducFactors)])
        
        df.to_csv(r'outputs.txt', sep=' ', header=None, mode='a')
    