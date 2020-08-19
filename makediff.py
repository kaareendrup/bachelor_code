# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:56:16 2020

@author: Kaare
"""

import cube
import numpy as np
import matplotlib.pyplot as plt

#%% Define variables

# Define target molecule
molecule = 'C60-1'

# Load file
#folder = r'C:\Users\Kaare\Documents\Nanoscience\6. semester\Bachelor\Indledende beregninger\Data/' + molecule + '/'

# Read cube file
C = cube.read_cube(molecule + '.cube')
# Define voxel dimensions and volume
dX = np.diagonal(C['dX'])
voxel_volume = dX[0]*dX[1]*dX[2]

# Import the density data from the cube
I = np.swapaxes(C['data'],0,2)#[:120,:,:118]

densities = I#(I[0::2,0::2,0::2] + I[0::2,0::2,1::2] + I[0::2,1::2,0::2] + I[0::2,1::2,1::2]+I[1::2,0::2,0::2] + I[1::2,0::2,1::2] + I[1::2,1::2,0::2] + I[1::2,1::2,1::2])/8

# Import model densities
densitiesnew = np.load(molecule + '_mockden.npy')#[:,:,:59]

#%%

# Calculate the difference between the cube and the model
diff = densities - densitiesnew

# Find different parameters of the error
maxerr = np.max(abs(diff))
meandiff = np.mean(abs(diff))
explorediff = np.sqrt(np.squeeze(diff)**2)

# Sum the density data, the new data and the error data
densum = np.sum(densities)*voxel_volume
newsum = np.sum(densitiesnew)*voxel_volume

integral = np.sum(explorediff)*voxel_volume

# Flatten the error array to plot a histogram
diffflat = explorediff.flatten()

hist, bin_edges = np.histogram(diffflat, bins=200)
bin_midpoints = (bin_edges[:-1] + bin_edges[1:])/2
plt.plot(bin_midpoints,hist)
plt.show()

plt.hist(diffflat, 200)
plt.xlabel('Density difference (electrons / voxel volume')
plt.ylabel('Frequency')
plt.show()


#Save differences for furter use

np.save(molecule + '_den2', densitiesnew)
np.save(molecule + '_dendiff', explorediff)
