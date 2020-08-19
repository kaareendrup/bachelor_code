# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:16:18 2020
@author: Endrup
"""

#%% Imports

import cube
# Dette import må gerne laves om til noget der ikke er afhængigt at molekylet
import C60_1 as c60 
import c60nt as c60nt
import numpy as np
import matplotlib.pyplot as plt 
import tqdm

#%% Define variables

# Define target molecule
molecule = 'C60-1'

# Define grid size
gridx = 150

# Define the maximum offset distance and the number of slices in each direction
dist = 2
slices = 0

#%% Read files and initialize

# Read cube file
#folder = r'C:\Users\Kaare\Documents\Nanoscience\6. semester\Bachelor\Indledende beregninger\Data/' + molecule + '/'

C = cube.read_cube(molecule + '.cube')

atoms = C['X']
# Overwriting cube atom coordinates # SKAL IKKE MED I OPGAVEN
#atoms = np.load('60-1_coorfix.npy')
# Reversing z,y,x coordinates and adding new dimension because the script was originally built to read data of that shape
densities = np.swapaxes(C['data'],0,2)[:,:,:,np.newaxis]

# Overwriting densities with difference or mock densities
densities = np.load(molecule + '_dendiff.npy')

# Define the origin of the grid
origin = C['X0']

# Define unit vectors
dX = np.diagonal(C['dX'])
xvec,yvec,zvec = dX #* 2
vec = np.array([xvec,yvec,zvec])

# Import shape index arrays
pentagons = np.squeeze(c60.pentagons)
hexagons = np.squeeze(c60.hexagons)

# Overwrite shape index for c60-1
pentagons = np.squeeze(c60nt.pentagons)
hexagons = np.squeeze(c60nt.hexagons)

# Import 2D layout data
faces_index = np.load(molecule + '_indexes.npy')
positions = np.load(molecule + '_positions.npy')

# Multiply the Eisenstein coordinates by the conversion matrix
faces_coor = []
m = np.array([[1,np.cos(np.pi/3)],[0,np.sin(np.pi/3)]])

for c in positions:
    faces_coor.append(np.dot(c,m.T) - [-5,-6])    #2,0 for C60-1812 
faces_coor = np.array(faces_coor)

for t in faces_coor:
    a = np.array([t[0],t[1],t[2],t[0]]).T
    plt.plot(a[0],a[1])

plt.gca().invert_yaxis()
plt.xlabel('Converted x-coordinate')
plt.ylabel('Converted y-coordinate')
plt.show()

#%% Define value from coordinate function
# This function performs the weighted sample of the 8 voxels around a point

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
    
    # Make a list of the amount of volume of the 8 voxels the coordinate samples from  
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

#%% Find pentagon og hexagon index from two atoms
# This functions searches of the shape that has two atoms in its chain. 
# All pairs of neighbors are in two shapes, but since the order is maintained, only one shape is returned
    
def shapeindexer(atom1,atom2):
    # Make a list of the indexes of all pentagons containing atom 1
    pindexes = np.array(np.where(pentagons == atom1)).transpose()  
    

    # Check if the atom pair is in one of the pentagons, output the index and the pentagons if true
    for i in pindexes:
        if pentagons[i[0],i[1]-4] == atom2:
            index = i[0]
            shape = pentagons
        
    # If the pair is not in one of the pentagons, repeat the process for the hexagons
    # Make a list of the indexes of all hexagons containing atom 1
    hindexes = np.array(np.where(hexagons == atom1)).transpose()

    # Check if the atom pair is in the hexagons, output the index and the hexagons if true
    for j in hindexes:
        if hexagons[j[0],j[1]-5] == atom2:
            index = j[0]
            shape = hexagons

    return shape, index

#%% Define barycentric sampler
# This functions samples a point in 3D from barycentric coordinates provided by the 2D structure
# It uses the barycentric weights from 2D to calculate the 3D point between 3 cornes in a triangle 
# Then it samples that point with the value function
# The input is the the indexes of two atoms that make up two cornes of the triangles and the barycentric coordinates
# offset is the offset in the z-direction if necessary 
    
def barysampler(aindexes, coors, offset):
    
    # Find the pentagon containing the atoms in the correct order, find the coordinates of all atoms and their center
    shape, shapeindex = shapeindexer(aindexes[0],aindexes[1])
    
    atomcoors = np.array([atoms[a] for a in shape[shapeindex]])    
    midpoint = np.mean(atomcoors, axis=0)
    
    # Find the normalvector of the shape and normalize
    nvec = np.cross(atoms[aindexes[0]]-midpoint,atoms[aindexes[1]]-midpoint)
    nvec = nvec / np.linalg.norm(nvec)
    
    # Multiply 3D coordinates of atoms by their barycentric weight and sum to get sample point
    coordinates = np.array([atoms[aindexes[0]],atoms[aindexes[1]],midpoint]) + (nvec * offset)  
    weighted = coordinates.transpose()*np.array(coors)    
    sample = np.sum(weighted, axis=1)
    
    return value(sample)

#%% Make grid looper
# This function makes a grid based on the length of the x dimensions and samples each pixel
# The functions loops over each triangle in the unfolding
# For each triangle, it makes a bounding box, so the entire grid is not checked for the barycentric condition

def mapper(gridx, offset):

    # Create the y-dimenstion relative to the length of the x-dimension
    gridy = np.ceil(np.max(faces_coor[:,:,1])/np.max(faces_coor[:,:,0])*gridx).astype('int')
    
    # The length of a pixel in coordinate units 
    gridvec = np.max(faces_coor[:,:,0])/gridx
    
    # Array of -100 entries
    grid = (np.zeros((gridy, gridx+1)) - 100)
    
    # Loop over each face
    for t in tqdm.trange(len(faces_index)):
        
        # Make bounding box in face coordinates
        xmin = np.min(faces_coor[t][:,0])
        ymin = np.min(faces_coor[t][:,1])
        
        # Find the boundaries of the bounding box in pixel coordinates
        pixxmin = int(np.floor(xmin/gridvec))
        pixxmax = int(pixxmin + np.ceil(1/gridvec)) + 1
        pixymin = int(np.floor(ymin/gridvec))
        pixymax = int(pixymin + np.ceil(np.sqrt(3)/2/gridvec)) + 1
        
        # Define all the cornes of the face in question
        tricoor = faces_coor[t]
        
        # Loop over the pixels of the bounding box and sample 
        for y in range(pixymin,pixymax):
            for x in range(pixxmin,pixxmax):
                pixcoor = np.array([x,y])*gridvec
                
                # Find the barycentric coordinates of the pixel
                wa = np.linalg.norm(np.cross(tricoor[1]-pixcoor,tricoor[2]-pixcoor))/np.linalg.norm(np.cross(tricoor[1]-tricoor[0],tricoor[2]-tricoor[0]))
                wb = np.linalg.norm(np.cross(tricoor[0]-pixcoor,tricoor[2]-pixcoor))/np.linalg.norm(np.cross(tricoor[1]-tricoor[0],tricoor[2]-tricoor[0]))
                wc = np.linalg.norm(np.cross(tricoor[0]-pixcoor,tricoor[1]-pixcoor))/np.linalg.norm(np.cross(tricoor[1]-tricoor[0],tricoor[2]-tricoor[0]))
                
                # Sample if the pixel fulfills the barycentric inside criteria
                if 0 <= wa <= 1 and 0 <= wb <= 1 and 0 <= wc <= 1 and 0.9999 < wa + wb + wc < 1.0001:
                    grid[y,x] = barysampler(faces_index[t,:2],[wa,wb,wc], offset)                    
    return grid

#%% Perform mapping on different offsets in the z-dimension
# Run function on each slice, or just once if slices is zero

gridstack = []

if slices == 0:
    gr = mapper(gridx, 0)
    gridstack.append(gr)
else:    
    # Loop over each slice, the number of slices is "slices" in each direction, therefore the slices*2+1
    for i in range(slices*2 + 1):
        gr = mapper(gridx, dist*(i/slices -1))
        gridstack.append(gr)

gridstack = np.array(gridstack)
            
# Show the data made from the grid
for i in range(len(gridstack)):
    
    g = gridstack[i]
    gridplot = np.ma.masked_where(g < -50, g)
    
    im = plt.imshow(gridplot, vmin=0, vmax=1, cmap='viridis')
    plt.xlabel('Pixel x-coordinate')
    plt.ylabel('Pixel y-coordinate')
    cbar = plt.colorbar(im)
    cbar.set_label('Electron density (electrons / voxel volume)')
    plt.savefig('map_' +str(i) + '.jpg', dpi=300)
    plt.show()

# Save single image of the middle slice
cmap = plt.cm.viridis
image = cmap(np.ma.masked_where(gridstack[slices] < -50, gridstack[slices]))
plt.imsave('objmap.jpg', image)

#%% Flatten and save

gridselect = np.array([i[i > -50] for i in gridstack])
np.save(folder + 'output/' + molecule + '_barysample', gridselect)

#%% Make a list of coordinates of midpoints for vedo export

pmids = np.load(molecule + '_pmids.npy')
hmids = np.load(molecule + '_hmids.npy')


mids = []

for p in pentagons:
    pcoors = atoms[p]
    mids.append(np.mean(pcoors,axis=0))
    
for h in hexagons:
    hcoors = atoms[h]
    mids.append(np.mean(hcoors,axis=0)) 
    
# Append to atom coordinates to make a list of all vertex coordinates, after that, convert to voxel coordinates
vertexes = (np.concatenate((atoms,np.array(mids))) - origin) / vec

# replace with proper coordinates
for p in range(len(pmids)):
    vertexes[int(pmids[p])] = (np.mean(atoms[pentagons[p]],axis=0) - origin) / vec
for h in range(len(hmids)):
    vertexes[int(hmids[h])] = (np.mean(atoms[hexagons[h]],axis=0) - origin) / vec

# Go through each face and output:
# - list of texture vertices u/v coordinates [x,y]
# - list of faces: vertex/texture index [a,b,c]
# - list of non-unique atom coordinates [x,y,z]

vtexture = []
faces = []
atomcoors = []

for f in range(len(faces_coor)):
    n = f*3
    verindexes = faces_index[f]
    vercoor = faces_coor[f]
    for a in range(3):
        vtexture.append(vercoor[a])
        atomcoors.append(vertexes[verindexes[a]])
    faces.append([n,n+1,n+2])
vtexture = np.array(vtexture)

# Turn the atom eisenstein coordinates into grid coordinates, and round to make sure to get the right pixel index
vtexture = vtexture / (np.max(faces_coor[:,:,0])/gridx)
vtexture = np.round(vtexture)

# Normalize
vtexture[:,0] = vtexture[:,0] /(np.max(vtexture[:,0]))
vtexture[:,1] = 1 - vtexture[:,1] /(np.max(vtexture[:,1]))

faces = np.array(faces)
atomcoors = np.array(atomcoors)

# Output
np.save('atom_coordinates.npy',atomcoors)
np.save('face_indexes.npy',faces)
np.save('texture_coordinates.npy',vtexture)
