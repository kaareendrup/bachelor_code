# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:13:57 2020

@author: Kaare
"""


#%% Imports

import cube
import numpy as np
from tqdm import trange
# Dette import må gerne laves om til noget der ikke er afhængigt at molekylet
import C60ih as c60
import matplotlib.pyplot as plt

#%% Define variables

# Define target molecule
molecule = 'C60-1812'
gridscale = 1/4

# Load file

#folder = r'C:\Users\Kaare\Documents\Nanoscience\6. semester\Bachelor\Indledende beregninger\Data/' + molecule + '/'

pfile = molecule + '_pentaab.npy'
hfile = molecule + '_hexaab.npy'

pentaab = np.load(pfile)
hexaab = np.load(hfile)

# Read cube file
C = cube.read_cube(molecule + '.cube')

atoms = C['X']
origin = C['X0']

# Overwriting cube atom coordinates
#atoms = np.load('60-1_coorfix.npy')

gridx = int(np.round(C['Ngrid'][0] * gridscale))
gridy = int(np.round(C['Ngrid'][1] * gridscale))
gridz = int(np.round(C['Ngrid'][2] * gridscale))

# Define unit vectors
xvec = C['dX'][()][0][0] / gridscale
yvec = C['dX'][()][1][1] / gridscale
zvec = C['dX'][()][2][2] / gridscale
vec = np.array([xvec,yvec,zvec])

# Import shape index arrays
pentagons = np.squeeze(c60.pentagons)
hexagons = np.squeeze(c60.hexagons)

pmids = np.mean(atoms[pentagons], axis=1)
hmids = np.mean(atoms[hexagons], axis=1)

# Import samples
pentagondata = np.load(molecule + '_sample_pentagons.npy')
hexagondata = np.load(molecule + '_sample_hexagons.npy')
zsteps = int((len(hexagondata[0,0,:,0]) - 1)/2)

#%% Normalvector function 

def nvec(stype, index):
    
    # Find the coordinates of 3 atoms in the chosen shape
    atomnums = stype[index][np.array([0,2,4])]
    atomcoor = atoms[atomnums]
    
    # Find the cross product of vectors between the atoms
    vector1 = atomcoor[0]-atomcoor[1]
    vector2 = atomcoor[2]-atomcoor[1]
    
    crossp = np.cross(vector2,vector1)
    nvector = crossp / np.sqrt(crossp.dot(crossp))
    
    return nvector

#%% Define coordinate to density
# This function takes a voxel coordinate, its intersection with the a shape and the shapes index 
# It finds the density value associated with the coordinate and outputs it.

def abdensity(shapetag, shape, index, voxcoor, intcoor, satoms, mid):

    # Define the vecor from the first of the atoms of the polygon and the midpoint.
    vec2 = satoms[0] - mid
    
    # Find the normal vector
    nvector = nvec(shape, index)

    # Find the spatial positions of the sample points by rotating around the normal vector    
    if shapetag == 'p':
        vecrot = np.array([vec2 * np.cos(v) + np.cross(nvector, vec2) * np.sin(v) + nvector * np.dot(nvector,vec2) * (1 - np.cos(v)) for v in pentagondata[index,:,zsteps,1]])
        lengths = np.repeat(pentagondata[index,:,zsteps,2,np.newaxis], 3, axis=1)
    else:
        vecrot = np.array([vec2 * np.cos(v) + np.cross(nvector, vec2) * np.sin(v) + nvector * np.dot(nvector,vec2) * (1 - np.cos(v)) for v in hexagondata[index,:,zsteps,1]])
        lengths = np.repeat(hexagondata[index,:,zsteps,2,np.newaxis], 3, axis=1)
    
    # And finding multiplying the normalized rotated vector with the distance from sample to midpoint
    positions = np.multiply(vecrot/np.linalg.norm(vec2),lengths) + mid
    
    # Find the difference in position between the sample points and the intersection
    diffs = np.linalg.norm(positions - np.repeat(intcoor[:,np.newaxis], len(positions), axis=1).T, axis=1)
    
    # Find the minimum value, and select the sample index to calculate the density
    samplecoor = np.where(diffs == np.min(diffs))[0][0]
    
    # Calculate the density using the correct a, the correct b and the distance
    if shapetag == 'p':
        rho = pentaab[index,0,samplecoor] * np.exp(0-pentaab[index,1,samplecoor] * np.linalg.norm(voxcoor - intcoor))
    else:
        rho = hexaab[index,0,samplecoor] * np.exp(0-hexaab[index,1,samplecoor] * np.linalg.norm(voxcoor - intcoor))
        
    return rho

#%% Define barycentric checker
# This function accepts a shape and a voxel coordinate
# It checks if the voxel intersects the shape orthogonally and inside one of the triangles
# It outputs true or false based on this, as well as the intersection coordinates. 

def barychecker(shapetag, shape,index, coor, atomcoors, mid, l):
    
    coordiff = np.linalg.norm(atomcoors - coor, axis=1)
    minindex = np.where(coordiff == np.min(coordiff))[0].item()
    
    # Check of the previous or the next is the closest
    if coordiff[minindex-1] < coordiff[minindex - l + 1]:
        tricoor = [atomcoors[minindex-1],atomcoors[minindex],mid]
    else:
        tricoor = [atomcoors[minindex],atomcoors[minindex - l + 1],mid]
    
    # Find the intersection of the plane of the triangle
    
    # Find the dot product of vectors between the atoms
    vec1 = tricoor[1]-tricoor[2]
    vec2 = tricoor[0]-tricoor[2]
    
    crossp = np.cross(vec2,vec1)    
    nvector = crossp / np.sqrt(crossp.dot(crossp))
        
    # Define vector between mid and the point
    v = coor - mid
    
    # Dot v with normal vector to find the scalar distance from point to plane
    dist = np.dot(v,nvector)
    
    # Multiply normal with distance and subtract from coordinates
    intersection = coor - dist*nvector
    
    va = tricoor[0]-intersection
    vb = tricoor[1]-intersection
    vc = tricoor[2]-intersection
    
    areafull = np.linalg.norm(np.cross(tricoor[1]-tricoor[0],tricoor[2]-tricoor[0]))
    
    # Find the barycentric coordinates of the intersection
    wa = np.linalg.norm(np.cross(vb,vc))/areafull
    wb = np.linalg.norm(np.cross(va,vc))/areafull
    wc = np.linalg.norm(np.cross(va,vb))/areafull
    
    if 0 <= wa <= 1 and 0 <= wb <= 1 and 0 <= wc <= 1 and 0.9 < wa + wb + wc < 1.1:
        onsurface = True
    else:
        onsurface = False
    
    return onsurface, intersection

#%% Run sampling

# Make voxel grid
dgrid = np.zeros([gridx,gridy,gridz])

# Loop over all voxels
for x in trange(int(gridx)):
    for y in range(int(gridy)):
        for z in range(int(gridz)):
            
            # Find voxel world coordinates
            coordinatesw = np.array([x,y,z]) * vec + origin + 1/2 * vec
            
            # Loop over each shape
            for p in range(len(pentagons)):
                
                # Define shape parameters
                atomcoors = atoms[pentagons[p]]
                midpoint = pmids[p]
                
                onsurftest, planecoor = barychecker('p', pentagons, p, coordinatesw, atomcoors, midpoint, 5)
                if onsurftest == True:
                    dgrid[x,y,z] = abdensity('p', pentagons, p, coordinatesw, planecoor, atomcoors, midpoint)
                
            for h in range(len(hexagons)):
                
                # Define shape parameters
                atomcoors = atoms[hexagons[h]]
                midpoint = hmids[h]
                
                onsurftest, planecoor = barychecker('h', hexagons, h, coordinatesw, atomcoors, midpoint, 6)
                if onsurftest == True:
                    dgrid[x,y,z] = abdensity('h', hexagons, h, coordinatesw, planecoor, atomcoors, midpoint)

#%% Single shape loop

# Make voxel grid
dgrid = np.zeros([gridx,gridy,gridz])

# Loop over all voxels
for x in trange(int(gridx)):
    for y in range(int(gridy)):
        for z in range(int(gridz)):
            
            # Find voxel world coordinates
            coordinatesw = np.array([x,y,z]) * vec + origin + 1/2 * vec
            
            # Loop over each shape
            for p in range(len(pentagons[:1])):
                
                # Define shape parameters
                atomcoors = atoms[pentagons[p]]
                midpoint = pmids[p]
                
                onsurftest, planecoor = barychecker('p', pentagons, p, coordinatesw, atomcoors, midpoint, 5)
                if onsurftest == True:
                    dgrid[x,y,z] = abdensity('p', pentagons, p, coordinatesw, planecoor, atomcoors, midpoint)
                    
            for h in range(len(hexagons[:2])):
                
                # Define shape parameters
                atomcoors = atoms[hexagons[h]]
                midpoint = hmids[h]
                
                onsurftest, planecoor = barychecker('h', hexagons, h, coordinatesw, atomcoors, midpoint, 6)
                if onsurftest == True:
                    dgrid[x,y,z] = abdensity('h', hexagons, h, coordinatesw, planecoor, atomcoors, midpoint)
                                        
#%%

np.save(molecule + '_mockden', dgrid)