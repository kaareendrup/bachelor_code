# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:31:57 2020

@author: Kaare
"""

#%% Imports

import numpy as np
from tqdm import trange

#%% Define variables

# Define target molecule
molecule = 'C60-1812'

# Load file

pfile = molecule + '_sample_pentagons.npy'
hfile = molecule + '_sample_hexagons.npy'

pdata = np.load(pfile)
hdata = np.load(hfile)

z = pdata[0,0,:,3]
zsteps = int((len(z)-1) /2)

#%% Define function

def shapefit(shapedata):
    
    #Create an empty array to hold the fitted b-values
    bfit = []
    
    # Shapedata holds the samples densities for each polygon in d x M x N array
    # d is the number of polygons, M is the sample points on each polygon and N is the samle points in the z-direction
    for d in trange(len(shapedata)):
        
        # Select the densities for the shape in question
        densities = shapedata[d]
        
        # The a-values aren't fitted, they are the densities where z=0
        a = densities[:,zsteps]
                
        # Create the d (density) vector
        M = len(z)
        N = len(densities)    
        d = np.zeros(M*N)
        
        # Apply the correct values to each entry in the d vector
        for i in range(N):
            for j in range(M):
                d[M*i+j] = np.log(densities[i,j]) - np.log(a[i])
        
        # Create A (coefficient) matrix
        A = np.zeros([M*N,N])
        
        # Apply the correct values to each entry in the coefficient matrix
        for i in range(N):
            for j in range(M):
                for k in range(N):
                    if i == k:
                        A[M*i+j,k] = - abs(z[j])
    
        # Calculate a & b values using numpy linear least squares fit
        bfit.append([a, np.linalg.lstsq(A,d, rcond=None)[0]])
            
    bfit = np.array(bfit)
            
    return bfit
    

#%% Run the function on the pentagons and hexagons and save the output

pentagonsab = shapefit(pdata[:,:,:,0])
hexagonsab = shapefit(hdata[:,:,:,0])

np.save(molecule + '_pentaab', pentagonsab)
np.save(molecule + '_hexaab', hexagonsab)
