import cube
import c60nt as c60
import numpy as np
import matplotlib.pyplot as plt
from vedo import *

# Import the cube data
C = cube.read_cube("C60-1.cube")
# Define the voxel grid size
Nx, Ny, Nz = C['Ngrid'];

# Import the voxel data
I = np.minimum(C['data'],1)
scalar_field = I #((I[0::2,0::2,0::2] + I[0::2,0::2,1::2] + I[0::2,1::2,0::2] + I[0::2,1::2,1::2]+I[1::2,0::2,0::2] + I[1::2,0::2,1::2] + I[1::2,1::2,0::2] + I[1::2,1::2,1::2])/8)

# Only show half of the data
scalar_field[Nx//2:,:,:] = 0;

# Import the model data to plot alongside the cube data
scalar_field_new = np.load('C60-1812_mockden.npy')
scalar_field_new = np.swapaxes(scalar_field_new,0,2)

# Overwrite voxel grid dimensions
Nx, Ny, Nz = scalar_field.shape

# Import the difference voxel data
difffield = np.load(r'C:\Users\Kaare\Documents\Nanoscience\6. semester\Bachelor\Indledende beregninger\Data\C60-1812\Output\C60-1812_dendiff.npy')
difffield = np.swapaxes(difffield,0,2)

# Define different parameters for visualizing the voxel data
vol = Volume(scalar_field)
vol.color(["green", "pink", "blue"])
vol.alpha([0, 0.1, 0.1, 0.1, 0.15])

vol1 = Volume(scalar_field)
vol1.color(["green", "pink", "blue"])
vol1.alpha([0, 0.1, 0.5, 0.7, 0.7, 1]) # This one looks pretty good

vol2 = Volume(scalar_field_new)
vol2.color(["green", "pink", "blue"])
vol2.alpha([0, 0.1, 0.5, 0.7, 0.7, 1])

vol3 = Volume(difffield)
vol3.color(["green", "pink", "blue"])
vol3.alpha([0, 0.1, 0.5, 0.7, 0.7, 1])

#vol.scale(1/3)
#vol.pos(C['X0'])

# Import difference data to be plotted showing only half of the volume
halfdiff = np.load(r'C:\Users\Kaare\Documents\Nanoscience\6. semester\Bachelor\Indledende beregninger\Data\C60-1812\Output\C60-1812_dendiff.npy')
halfdiff = np.swapaxes(halfdiff ,0,2)
halfdiff[Nx//2:,:,:] = 0;

# Define visualization parameters
vol4 = Volume(halfdiff)
vol4.color(["green", "pink", "blue"])
vol4.alpha([0, 0.05, 0.05, 0.05, 0.1, 0.1])

# Create lego visualization 
lego = vol4.legosurface(vmin=0.01)
lego.addScalarBar3D()

print('numpy array from Volume:', 
       vol.getPointArray().shape, 
       vol.getDataArray().shape)

# Find atom coordinates and attach spheres to mark their location
Xi = cube.world_to_grid_coords(C,C['X']);
spheres = [Sphere(pos=x,r=1.5, c='lg') for x in Xi]

# Load vertex coordinates, face indexes and texture coordinates for texture mapping
folder = r'C:\Users\Kaare\Documents\Nanoscience\6. semester\Bachelor\Troubleshoot folder/'
Ai = np.load(folder + 'atom_coordinates.npy')
faces = np.load(folder + 'face_indexes.npy')
texcoords = np.load(folder + 'texture_coordinates.npy')

# Import polygon data to plot alongside densities
#pentagons = Mesh([Xi,c60.pentagons],c='b',alpha=1)
#hexagons  = Mesh([Xi,c60.hexagons],c='o',alpha=1)

# Import face data to plot using texture coordinates
triangles = Mesh([Ai,faces],c='b',alpha=1).texture(folder + 'c60-1_sample_objmap.jpg', tcoords=texcoords, repeat=True)
triangles2 = Mesh([Ai,faces],c='b',alpha=1).texture(folder + 'c60-1_ab_objmap.jpg', tcoords=texcoords, repeat=True)

# Variations of plotting options
#show([vol,pentagons,hexagons])
#show([[vol,pentagons,hexagons]+spheres, [lego,pentagons,hexagons]+spheres], N=2, azimuth=10)
#show([[vol1]+spheres, [vol1,triangles]+spheres, [lego,triangles]], N=3, azimuth=10)


show([[triangles], [triangles2]], N=2, azimuth=10)
#show([[vol2], [vol1], [vol3], [lego]+spheres], N=4, azimuth=10)

#show([[vol2]], N=1, azimuth=10)
