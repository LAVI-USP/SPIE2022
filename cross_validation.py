"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import os 

datasetsizes = [256000, 192000, 128000, 64000]

for rlz in range(1,4):
    
    for datasetsize in datasetsizes:
        
        print("\nL1 por 60 epochs - {}-{}\n".format(rlz,datasetsize))
        file1 = open("outputs.txt","a")  
        file1.write("\nL1 por 60 epochs - {}-{}\n".format(rlz,datasetsize))
        file1.close() 
        
        os.system("python main_training_L1.py --rlz {} --dts {}".format(rlz, datasetsize))
                
        print("\nPL4 apos L1 - {}-{}\n".format(rlz,datasetsize))
        file1 = open("outputs.txt","a")  
        file1.write("\nPL4 apos L1 - {}-{}\n".format(rlz,datasetsize))
        file1.close() 
        
        os.system("python main_training_PL.py --rlz {} --dts {}".format(rlz, datasetsize))
       