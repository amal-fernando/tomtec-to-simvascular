import tkinter as tk
from tkinter import filedialog
import numpy as np
from funzioni import riordina, timeplot, resample_u, pv_to_np, np_to_pv
from funzioni import volume, normalplot, mmg_remesh, mesh_quality, get_bounds
from funzioni import fourier
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import pyvista as pv
import tetgen
import os


'''
This script is used to load a set of .ucd files, extract the coordinates of the nodes and the connectivity of the 
triangles, and then perform interpolation on the data. The script also includes functions for visualizing the data and 
saving it in different formats.
'''

#================================================
# 1. Parse the header file to extract information
#================================================

print("Script started")
# Initialize tkinter but hide main window
root = tk.Tk()
root.withdraw()

# Open file dialog to select header file
file_path = filedialog.askopenfilename(
    title="Select header file",
    filetypes=[("Text files", "*.txt")]
)

# Read and parse the header file
with open(file_path, 'r') as fid:
    # Get base filename without extension and remove last 10 chars
    base_name = os.path.basename(file_path)
    radice_dataset = base_name[:-10]
    path= os.path.dirname(file_path)

    # Skip first line
    next(fid)
    
    # Read second line for number of frames
    line = fid.readline().strip().split()
    frames_0 = int(line[-1])
    
    # Read third line for number of nodes 
    line = fid.readline().strip().split()
    N_vertices = int(line[-1])
    
    # Read fourth line for number of triangles
    line = fid.readline().strip().split()
    N_faces = int(line[-1])

    # Retrieve information on RR interval, end-diastolic and end-systolic times
    # Skip to 9th line (skip 4 more lines)
    for _ in range(3):
        next(fid)
    
    # Read 8th line and get RR interval
    line = fid.readline().strip().split()
    rr = float(line[-2])  # Assuming float is last number in line

    # Read 9th line and get end-diastolic time
    line = fid.readline().strip().split()
    ed = float(line[-2])  # Assuming float is last number in line

    # Read 10th line and get end-systolic time
    line = fid.readline().strip().split()
    es = float(line[-2])  # Assuming float is last number in line
    
    print(f"Time step value (rr, ed, es): {rr, ed, es}")

    # Retrieve time values
    # Skip to 14th line (skip 4 more lines after line 9)
    for _ in range(3):
        next(fid)

    t0 = np.zeros((frames_0, 1))  # Initialize array to store time values

    # Loop through frames to find matching time
    for i in range(frames_0):
        line = fid.readline().strip()
        t0[i] = float(line.split()[0])  # Get first number from line

    #print(f"Time vector: {t0}")

#==============================================================================
# 2. Load the .ucd files and extract node coordinates and triangle connectivity
#==============================================================================

# Initialize arrays with zeros
v0 = np.zeros((N_vertices, 3, frames_0)) #3D matrix to store coordinates of each node at each time point
f0 = np.zeros((N_faces, 3)) #2D matrix to store connectivity of each triangle (not time dependent)

print("Start loading files")

# First loop: from start_frame to fasi
for i in range(frames_0):
    # Generate filename
    data = f"{radice_dataset}{i:02d}.ucd"
    full_path = os.path.join(path, data)
    # print(f"{full_path=}")
    
    # Load node coordinates
    temp_data = np.loadtxt(full_path, skiprows=1, delimiter=' ', usecols=(1,2,3), max_rows=N_vertices)
    v0[:,:,i] = temp_data[:N_vertices]
    
    # Load face connectivity only for first iteration
    if i == 0:
        temp_faces = np.loadtxt(full_path, skiprows=N_vertices+1, usecols=(3,4,5), max_rows=N_faces)
        f0[:,:] = temp_faces[:N_faces]  # Convert to 1-based indexing
        f0 = f0.astype(np.int64)

print('v0.shape =', v0.shape)
# print(v0[:,:,0])
print('f0.shape =', f0.shape)
# print(f0)

print("Loading Done")

# u_orig = v0 - v0[:, :, 0][:, :, np.newaxis]  # Original displacements
# np.save('u_orig.npy', u_orig) # Save the original displacements
# np.save('v0_orig.npy', v0[:,:,0]) # Save the original coordinates of the initial frame
# np.save('f_orig.npy', f0) # Save the original connectivity
# np.save('time.npy', t0) # Save the time vector

# timeplot(v0,f0) # plot della mesh rada nel tempo \decommentare
# normalplot(v0, f0, 1) # plot delle normali alla mesh rada \decommentare

#============================================================
# 3. Reorder and extend the time vector and the displacements
#============================================================

# rr_true = t0[-1]
[t1,v1] = riordina(t0,v0,ed,rr)

# Extend the displacements to increase the number of cycles
num_sequences = 3 # Number of sequences wanted
nt = len(t1) - 1  # Elimino l'ultimo dato, che deve essere inserito solo in coda alla sequenza

# Duplica la matrice iniziale
v2 = np.tile(v1[:, :, :nt], (1, 1, num_sequences + 1))  # Crea un ciclo di troppo inizialmente
v2 = v2[:, :, :-nt+1]  # Elimina i dati dopo il duplicato del primo in coda alla sequenza

#print('v2.shape =', v2.shape)
# print(v2[:,:,75])

frames_2 = v2.shape[2]

# Costruzione del vettore tempo t2
t2 = np.zeros(num_sequences * nt + 1)

for i in range(1, num_sequences + 1):
    t2[(i - 1) * nt:i * nt] = t1[:nt] + (i - 1) * rr
t2[-1] = rr * num_sequences

#===========================================================
# 4. Interpolation of the data to create intermediate frames
#===========================================================

num_intermedie = 4  # Number of intermediate frames to insert between each original frame
frames_3  = frames_2 + (frames_2- 1) * num_intermedie

# Initialize v3 with the appropriate dimensions
v3 = np.zeros((v2.shape[0], v2.shape[1], frames_3))
v_cubic = np.zeros(v3.shape)  # For cubic interpolation
v_linear = np.zeros(v3.shape)  # For linear interpolation
v_pchip = np.zeros(v3.shape)  # For PCHIP interpolation

# Create a time vector for the new frames
t3 = np.linspace(t2[0], t2[-1], frames_3)

# Cubic interpolation
for i in range(v2.shape[0]):  # For each node
    for dim in range(3):  # For each spatial dimension (x, y, z)
        coordinates = v2[i, dim, :] # Extract the coordinates in the current dimension
        spline = CubicSpline(t2, coordinates, bc_type='periodic') # Create an interpolator for the current node and dimension
        v_cubic[i, dim, :] = spline(t3) # Interpolate to the new time points

# Linear interpolation
for i in range(v2.shape[0]):  # For each node
    for dim in range(3):  # For each spatial dimension (x, y, z)
        coordinates = v2[i, dim, :]
        linear_interp = interp1d(t2, coordinates, kind='linear')
        v_linear[i, dim, :] = linear_interp(t3)

# PCHIP interpolation
for i in range(v2.shape[0]):
    for dim in range(3):
        coordinates = v2[i, dim, :]
        pchip_interp = PchipInterpolator(t2, coordinates)
        v_pchip[i, dim, :] = pchip_interp(t3)

# Fourier interpolation
v_fourier = fourier(v_pchip, t2, 0)

# Create the plot for the original and interpolated volumes
plt.figure()
plt.plot(t2,volume(v2,f0), '*', label='volOriginal')
plt.plot(t3,volume(v_cubic, f0), label='volCubic')
plt.plot(t3,volume(v_linear, f0), label='volLinear')
plt.plot(t3,volume(v_pchip, f0), label='volPchip')
plt.plot(t3,volume(v_fourier, f0), label='volFourier')
plt.legend()
plt.grid(True)
plt.xlabel('Time Frame')
plt.ylabel('Volume')
plt.title('Volume Over Time')
plt.show()

v3 = v_fourier  # Use the Fourier interpolated data for further processing
normalplot(v3, f0, 1)  # Plot the normals of the interpolated mesh
#=================================
# 5. Mesh evaluation and remeshing
#=================================

mesh = np_to_pv(v3[:,:,0],f0) # Load the mesh from the numpy arrays
# mesh.save("coarse_mesh.vtp")  # Save the coarse mesh for reference
avg,_,_ = mesh_quality(mesh)

if avg['aspect_ratio'] > 1.5 or avg['radius_ratio'] > 1.5:
    print("The mesh quality is poor.")
    print("Remeshing should be done.")

else:
    print("The mesh quality is good.")
    print("No need to remesh.")

edges = mesh.extract_all_edges().compute_cell_sizes()
avg_edge = edges['Length'].mean()
hausd = avg_edge  # Use this instead of a hardcoded 0.3

# gmsh_mesh = gmsh_remesh(mesh)
remesh = mmg_remesh(mesh, hausd=hausd,verbose=True  ) # Remesh the mesh using MMG

avg,_,_ = mesh_quality(remesh)

if avg['aspect_ratio'] > 1.5 or avg['radius_ratio'] > 1.5:
    print("The mesh quality is poor.")
    print("Remeshing should be done.")

else:
    print("The mesh quality is good.")
    print("No need to remesh.")

remesh = remesh.triangulate() # Ensure the remeshed mesh is triangulated
vert, fac = pv_to_np(remesh)  # Convert remeshed mesh to numpy arrays
vert = vert[:, :, np.newaxis]
normalplot(vert, fac, 0)  # Plot the normals of the remeshed mesh
#======================================================
# 6. Generate the volume mesh from the remeshed surface
#======================================================
# TetGen options (preserve surface, good quality)
tetgen_options = "pq1.2a0.333"

tet = tetgen.TetGen(remesh)
tet.tetrahedralize(order=1, switches=tetgen_options) # "pq1.2a0.333" or tet.tetrahedralize(order=1, switches=tetgen_options)

# Convert to PyVista mesh
grid = tet.grid

# grid.plot(show_edges=True)

# get cell centroids
cells = grid.cells.reshape(-1, 5)[:, 1:]
cell_center = grid.points[cells].mean(1)

# extract cells below the 0 xy plane
mask = cell_center[:, 2] < 0
cell_ind = mask.nonzero()[0]
subgrid = grid.extract_cells(cell_ind)

# advanced plotting
plotter = pv.Plotter()
plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
plotter.add_mesh(remesh, 'r', 'wireframe')
plotter.add_legend([[' Input Mesh ', 'r'],
                    [' Tessellated Mesh ', 'black']])
plotter.show()

# Save as VTU
grid.point_data["GlobalNodeID"] = np.arange(grid.points.shape[0]) + 1
grid.cell_data["GlobalElementID"] = np.arange(grid.n_cells) + 1
grid.cell_data["ModelRegionID"] = np.ones(grid.n_cells, dtype=int)  # Set to one for all cells
grid.save("mesh-complete.mesh.vtu")

surf_mesh = grid.extract_geometry()

surf_mesh.save("mesh-complete.exterior.vtp")

_,_,_ = get_bounds(surf_mesh) # ("cyl.stl")






#==================================================================================
# 7. Remapping the coarse meshes of the time interpolated data on the remeshed mesh
#==================================================================================

v4_, f4 = pv_to_np(remesh)  # Convert remeshed mesh to numpy arrays

# Initialize v3 with the appropriate dimensions
v4 = np.zeros((v4_0.shape[0], v4_0.shape[1], v3.shape[2]))

tr = f0
p = v3[:, :, 0]

# Loop through each time step
for i in range(v3.shape[2] - 1):
    vr = v3[:, :, i]
    ur = v3[:, :, i + 1] - v3[:, :, i]

    # Assuming resample_u is already defined, and returns pp and upp
    pp, upp = resample_u(vr, tr, ur, p)

    # Store the results in v3
    v4[:, :, i] = pp
    p = upp + pp
    v4[:, :, i + 1] = p

# timeplot(v3,f2)