# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:16:18 2020

@author: Endrup
"""

#%% Imports

import cube
# Dette import må gerne laves om til noget der ikke er afhængigt at molekylet
import C60ih as c60 
import numpy as np
import tqdm

#%% Define variables

# Define target molecule
molecule = 'C60-1812'

# Define resolution parameters
segments = 12
zsteps = 8
zdist = 3


#%% Read files and initialize

# Read cube file
C = cube.read_cube(molecule + '.cube')

atoms = C['X']
# Overwriting cube atom coordinates
#atoms = np.load('60-1_coorfix.npy')
# Reversing z,y,x coordinates and adding new dimension because the script was originally built to read data of that shape
densities = np.swapaxes(C['data'],0,2)[:,:,:,np.newaxis]
origin = C['X0']

# Define unit vectors
xvec = C['dX'][()][0][0]
yvec = C['dX'][()][1][1]
zvec = C['dX'][()][2][2]
vec = np.array([xvec,yvec,zvec])

# Import shape index arrays
pentagons = np.squeeze(c60.pentagons)
hexagons = np.squeeze(c60.hexagons)

#%% Define value from coordinate function

def value(coordinates):
    
    # Define the coordinates in terms of voxels
    point = ((coordinates - origin) / vec)
    
    # Define the nearest voxel and the remainder of the distance to it
    vox = np.round(point).astype(np.int64)
    rem = point - vox
    
    # Make a combinatory list of ones and zeros to add and subtract the remainder
    signs = [1, -1]
    signm = []
    for i in signs:
        for j in signs:
            for k in signs:
                signm.append([i,j,k])
    
    # Make a list of the amount volume of the 8 voxels the coordinate samples from  
    voxelw = np.prod(0.5 - np.multiply(rem,signm), axis=1)
    
    # Make a corresponding list of the density values in each voxel
    binar = [1,0]
    val = []
    for i in binar:
        for j in binar:
            for k in binar:
                val.append(np.asscalar(densities[vox[0]-i][vox[1]-j][vox[2]-k] ))
    denvalues = np.array(val)
    
    # Dot product to find the weighted density value
    return np.dot(voxelw,denvalues)

#%% Define segments function that takes two coordinates and calculates values at segments between them

def segment(coor1, coor2, midp, segments):
    
    # Find coordinates of points between atom 1 and 2 and calculate the values between them
    segcoordinates = np.linspace(coor1,coor2,segments)
    
    # Initialize lists
    segvalues = [value(segcoordinates[0])]
    segangles = [0]
    segdist = [np.linalg.norm(segcoordinates[0] - midp)]
    
    for i in segcoordinates[1:]:
        
        # Append density value
        segvalues.append(value(i))

        # Append angle 
        mc1 = segcoordinates[0] - midp
        mc2 = i - midp
        segangles.append(np.arccos( np.dot(mc1,mc2) / (np.linalg.norm(mc1)*np.linalg.norm(mc2)) ))
        
        # Append distance
        segdist.append(np.linalg.norm(mc2))
    
    return np.array(segvalues), np.array(segangles), np.array(segdist)

#%% Make function that samlpes a shape via triangles accepts coordinates
 
def trival(coordinates, segments):
    
    #Find the midpoint of the coodinates
    midp = np.mean(coordinates, axis=0)
    
    shapeval = []
    shapeangles = []
    shapedists = []
    
    # Iterate over atoms to sample each triangle in the shape
    for a in range(len(coordinates)):
        
        # Make two list of coordinates between two atoms in the shape and the midpoint
        sides = [np.linspace(coordinates[a], midp, segments), np.linspace(coordinates[a-(len(coordinates)-1)], midp, segments)]
        
        trival = []
        tri_angles = []
        tridists = []
        
        # For each pair of points in the lists, sample points between them. Sample one point less per segment.
        for i in range(segments):
            segval, segangles, segdists = segment(sides[0][i], sides[1][i], midp, segments-i)
            trival.append(segval)
            tri_angles.append(segangles)
            tridists.append(segdists)
        
        shapeval.append(np.concatenate(trival))
        
        # Add the maximum angle of the previous array (if there is a previous array) so get the total angle
        if a == 0:
            shapeangles.append(np.concatenate(tri_angles))
        else:
            shapeangles.append(np.concatenate(tri_angles) + shapeangles[-1][-2])
        
        shapedists.append(np.concatenate(tridists))

    return np.concatenate(shapeval), np.concatenate(shapeangles), np.concatenate(shapedists)

#%% Define normalvector function from shape
    
def nvec(stype, index):
    
    # Find the coordinates of 3 atoms in the chosen shape
    atomnums = stype[index][np.array([0,2,4])]
    atomcoor = atoms[atomnums]
    
    # Find the cross product of vectors between the atoms
    vec1 = atomcoor[0]-atomcoor[1]
    vec2 = atomcoor[2]-atomcoor[1]
    
    crossp = np.cross(vec2,vec1)
    nvector = crossp / np.sqrt(crossp.dot(crossp))
    
    return nvector

#%% Make z-dimension function

def shapesample(stype, index, segments, zsteps, zdist):
    
    # Find the noraml vector for the shape and divide by zdist
    vector = nvec(stype, index)
    
    # Find coordinates of the atom
    atomcoordinates = atoms[stype[index]]
    
    # Make an array of offset coordinates 
    coordinates = []
    zdistances = []
    for i in range(-zsteps,zsteps+1):
        coordinates.append(np.add(atomcoordinates, vector*zdist*i/zsteps))
        # linalg.norm(vector) is 1, since the vector is normalized. This is only left in for readability
        zdistances.append(np.linalg.norm(vector)*zdist*i/zsteps) 
        
    # Run the triangle sample function on each shape of offset coordinates
    value_all = []
    angles_all = []
    distances_all = []
    
    for shape in coordinates:
        shapeval, shapeangles, shapedists = trival(shape, segments)
        value_all.append(shapeval)
        angles_all.append(shapeangles)
        distances_all.append(shapedists)
    
    return np.array([np.array(value_all), np.array(angles_all), np.array(distances_all), np.tile(zdistances,(len(value_all[0]),1)).T  ])

#%% Run functions on entire molecule and save

# Loop over all pentagons
parray = np.array([shapesample(pentagons,i, segments, zsteps, zdist) for i in tqdm.trange(len(pentagons))]).swapaxes(1, 3)
# Loop over all hexagons
harray = np.array([shapesample(hexagons,i, segments, zsteps, zdist) for i in tqdm.trange(len(hexagons))]).swapaxes(1, 3)

# Save to file
np.save(molecule + '_sample_pentagons', parray)
np.save(molecule + '_sample_hexagons', harray)