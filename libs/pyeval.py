"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""
    
import numpy as np
import scipy.stats as st

def MNSE(rlzs_GT, rlzs):
    '''
    
    Description: Mean normalized squared error (MNSE), which we compute as 
    follows: first, for every pixel, we compute the normalized quadratic error
    as the squared difference from the ground-truth divided by the ground-
    truth; second, the MNSE is obtained as the average of the pixelwise 
    normalized quadratic errors on the breast over the entire image. 
    
    Input:
        - rlzs_GT = Realizations of the ground-truth
        - rlzs = Realizations of non-ground-truth images
    
    Output:
        - mnse = (Mean normalized squared error (MNSE), CI)
        - resNoiseVar = (Normalized residual noise variance, CI)
        - bias2 = (Normalized bias squared, CI)
        - proof = Proof of the decomposition
            
    
    Source: https://doi.org/10.1088/1361-6501/aab2f6

    Please, if you use this metric, we ask you to cite the above paper.
    
    Code by Lucas Borges. Python by Rodrigo
    
    '''
    
    n_rlzs_GT = rlzs_GT.shape[-1]   # Number of ground-truth realizations
    n_rlzs = rlzs.shape[-1]         # Number of non-ground-truth realizations
    
    # Generate the ground-truth from the realizations
    groundTruth = np.mean(rlzs_GT, axis=-1)
    
    # There are a limited number of images to estimate the GT, so there is an 
    # error associated with the measurements. This Factor will cancel out this
    # error (The factor is basically (ResidualNoiseSTD*1/sqrt(N))^2
    resNoiseVar_GT = np.mean(np.var(rlzs_GT, ddof=1, axis=-1) / groundTruth)
    factor1 = (resNoiseVar_GT / n_rlzs_GT)
    
    #  Estimate the Mean Normalized Squared Error (MNSE) from each realization
    # the 'normalization' term here is the signal expectation
    mnse = np.empty(shape=n_rlzs)
    for r in range(n_rlzs):
        nqe = ((rlzs[:,:,r] - groundTruth) ** 2) / groundTruth
        mnse[r] = nqe.mean() - factor1
    
    # Estimate the normalized residual noise variance
    resNoiseVar = np.var(rlzs, ddof=1, axis=-1) / groundTruth
    
    # Calculate the confidence interval
    resNoiseVar_CI = st.t.interval(0.95, resNoiseVar.shape[1]-1, 
                                   loc=np.mean(resNoiseVar), 
                                   scale=st.sem(resNoiseVar,axis=-1)[0])
    
    resNoiseVar = resNoiseVar.mean()


    # Estimate the normalized bias squared
    bias2 = ((np.mean(rlzs, axis=-1) - groundTruth) ** 2) / groundTruth
    
    # Again, there is an error associated with the limited number of realiza-
    # tions that we used to estimate the bias. This second factor is related 
    # to the number of realizations used for the bias estimation (n_rlzs), 
    # while Factor 1 is related to the number of realizations used for the 
    # GT (n_rlzs_GT). 
    factor2 = (resNoiseVar / n_rlzs)
    
    # The bias must now be adjusted by two factors: one of them due to the 
    # 'imperfect' GT (Factor1) and the second one due to the limited number of 
    # realizations used to estimate the bias itself (Factor2)
    bias2 = bias2 - factor1 - factor2 
    
    # Calculate the confidence interval
    bias2_CI = st.t.interval(0.95, bias2.shape[1]-1, 
                                   loc=np.mean(bias2), 
                                   scale=st.sem(bias2,axis=-1)[0])
    
    bias2 = bias2.mean()  
    
    # Since the bias squared and the residual noise variance are the decompo-
    # sitions of the MNSE, the sum of bias^2 + Residual Variance must be equal 
    # to the MNSE
    mnse_CI = st.t.interval(0.95, mnse.shape[0]-1, 
                                   loc=np.mean(mnse), 
                                   scale=st.sem(mnse))
    mnse = mnse.mean() 
    proof = mnse - resNoiseVar - bias2
    
    # print('==================================')
    # print('Total MNSE: {:.2f}%'.format(100*mnse))
    # print('Residual Noise: {:.2f}%'.format(100*resNoiseVar))
    # print('Bias Squared: {:.2f}%'.format(100*bias2))
    # print('Proof (must be ~0%): {:.2e}%'.format(100*proof))
    # print('==================================')
    
    return np.hstack([mnse,mnse_CI]), np.hstack([resNoiseVar,resNoiseVar_CI]), np.hstack([bias2,bias2_CI]), proof

