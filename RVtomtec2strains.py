import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
from geometry_functions import find_sides
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial

# Define global variables
global v, f, vicini, valenza, lati, F, C, B, E, e, A, Adef, Ecp, Edp

# Define max_vicini
max_vicini = 10

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
    fasi = int(line[-1])
    
    # Read third line for number of nodes 
    line = fid.readline().strip().split()
    N_nodi = int(line[-1])
    
    # Read fourth line for number of triangles
    line = fid.readline().strip().split()
    N_elementi = int(line[-1])

    # Skip to 9th line (skip 4 more lines)
    for _ in range(4):
        next(fid)
    
    # Read 9th line and get float number
    line = fid.readline().strip().split()
    time_step = float(line[-2])  # Assuming float is last number in line
    
    print(f"Time step value: {time_step}")

    # Skip to 14th line (skip 4 more lines after line 9)
    for _ in range(4):
        next(fid)

    # Initialize variables
    Enddiastole_frame = 1
    Enddiastole_time = time_step  # Using time_step we read earlier

    # Loop through frames to find matching time
    for i in range(fasi):
        line = fid.readline().strip()
        current_time = float(line.split()[0])  # Get first number from line
        
        if Enddiastole_time == current_time:
            Enddiastole_frame = i  # Adjust for 0-based indexing
            break
        else:
            Enddiastole_frame += 1
        
        # Skip rest of line (equivalent to MATLAB's fgetl)
        
    print(f"Enddiastole frame: {Enddiastole_frame}")


# Initialize arrays with zeros
v = np.zeros((N_nodi, 3, fasi)) #3D matrix to store coordinates of each node at each time point
f = np.zeros((N_elementi, 3)) #2D matrix to store connectivity of each triangle (not time dependent)
vicini = np.zeros((max_vicini, N_nodi)) #2D matrix to store neighbors ids of each node
valenza = np.zeros((N_nodi, 1)) #1D matrix to store number of neighbors of each node
lati = np.zeros((max_vicini, 3, N_nodi, fasi)) #3D matrix to store vectors from each node to its neighbors
F = np.zeros((3, 3, N_nodi, fasi)) #4D matrix to store deformation gradient tensor of each node at each time point
C = np.zeros((3, 3, N_nodi, fasi)) #4D matrix to store right Cauchy-Green tensor of each node at each time point
B = np.zeros((3, 3, N_nodi, fasi)) #4D matrix to store left Cauchy-Green tensor of each node at each time point
E = np.zeros((3, 3, N_nodi, fasi)) #4D matrix to store Green-Lagrange strain tensor of each node at each time point
e = np.zeros((3, 3, N_nodi, fasi)) #4D matrix to store Almansi strain tensor of each node at each time point
A = np.zeros((N_nodi, fasi)) #2D matrix to store area of each node at each time point
Adef = np.zeros((N_nodi, fasi)) #2D matrix to store areal deformation of each node at each time point
Ecp = np.zeros((N_nodi, 3, fasi)) #3D matrix to store principal strains of E at each node at each time point
Edp = np.zeros((3, 3, N_nodi, fasi)) #4D matrix to store principal directions of E at each node at each time point

# print('Start loading files')

# Initialize new index for reordering
start_frame = Enddiastole_frame
new_index = 0  # Python uses 0-based indexing

print('start_frame:', start_frame)
print('fasi:', fasi)

print('Start loading files')

# First loop: from start_frame to fasi
for i in range(start_frame+1, fasi):
    # Generate filename
    data = f"{radice_dataset}{i:02d}.ucd"
    full_path = os.path.join(path, data)
    print(f"{full_path=}")
    
    # Load node coordinates
    temp_data = np.loadtxt(full_path, skiprows=1, delimiter=' ', usecols=(1,2,3), max_rows=N_nodi)
    v[:,:,new_index] = temp_data[:N_nodi]
    
    # Load face connectivity only for first iteration
    if i == start_frame+1:
        temp_faces = np.loadtxt(full_path, skiprows=N_nodi+1, usecols=(3,4,5), max_rows=N_elementi)
        f[:,:] = temp_faces[:N_elementi]  # Convert to 1-based indexing
        f = f.astype(np.int64)
    
    new_index += 1

# Second loop: from 0 to Enddiastole_frame
for i in range(Enddiastole_frame+1):
    # Generate filename
    data = f"{radice_dataset}{i:02d}.ucd"
    full_path = os.path.join(path, data)
    print(f"{full_path=}")
    
    # Load node coordinates
    temp_data = np.loadtxt(full_path, skiprows=1, delimiter=' ', usecols=(1,2,3), max_rows=N_nodi)
    v[:,:,new_index] = temp_data[:N_nodi]
    new_index += 1

print('Done')

# print('v.shape =', v.shape)
# print(v[:,:,0])
# print('f.shape =', f.shape)
# print(f)

# for i in range(fasi):
#     # Generate filename (equivalent to MATLAB's strcat + sprintf)
#     data = f"{radice_dataset}{i:02d}.ucd"
#     full_path = os.path.join(path, data)
#     #print(f"{full_path=}")

#     # Load node coordinates (equivalent to dlmread with specific range)
#     # Skip first line, read columns 1-3
#     temp_data = np.loadtxt(full_path, skiprows=1, delimiter=' ', usecols=(1,2,3), max_rows=N_nodi)
#     v[:,:,i] = temp_data[:N_nodi]
#     #print(f"v[:,:,{i}].shape = {v[:,:,i].shape}")
    
#     # Load face connectivity only for first iteration
#     if i == 0:  # Note: Python uses 0-based indexing
#         # Skip N_nodi+1 rows, read columns 3-5
#         temp_faces = np.loadtxt(full_path, skiprows=N_nodi+1, usecols=(3,4,5), max_rows=N_elementi)
#         f[:,:] = temp_faces[:N_elementi] # Do not add 1 to convert 0-based to 1-based indexing
#         # Convert f to integers
#         # print("Before conversion:", f.dtype)
#         f = f.astype(np.int64)
#         # print("After conversion:", f.dtype)

# print('Done')


#print('v.shape =', v.shape)
#print(v[:,:,0])
#print('f.shape =', f.shape)
#print(f)
#print(type(f[1,1]))


print('Initializing variables at first time point')

for j in range(N_nodi):  # Note: Python uses 0-based indexing
    # Call find_sides function
    next_points, sides, n = find_sides(j, v[:,:,0], f, None)
    
    # Store results in global arrays
    vicini[:n,j] = next_points
    valenza[j] = n
    lati[:n,:,j,0] = sides
    
    # Initialize identity matrices
    F[:,:,j,0] = np.eye(3)
    C[:,:,j,0] = F[:,:,j,0]
    B[:,:,j,0] = F[:,:,j,0]
    
    # Calculate area using cross product
    # Roll sides array to get pairs for cross product
    sides_rolled = np.roll(sides, -1, axis=0)
    cross_products = np.cross(sides, sides_rolled)
    A[j,0] = 0.5 * np.sum(np.sqrt(np.sum(cross_products**2, axis=1)))

print('Initialization complete')

vicini = vicini.astype(np.int64)
valenza = valenza.astype(np.int64)

# print(vicini)
# print(valenza)
# print(lati)

# Set frame range (adjust for 0-based indexing)
f1 = 1  # starts from second frame
f2 = fasi  # last frame

# Remove integer conversion of lati (keep as float)
vicini = vicini.astype(np.int64)
valenza = valenza.astype(np.int64)
# lati = lati.astype(np.int64)  # Remove this line - keep lati as float

print('Perfroming strain analysis')

for i in range(f1, f2):
    for j in range(N_nodi):
        n = valenza[j].item()
        # Get ordered neighbors from vicini
        _, sides, _ = find_sides(j, v[:,:,i], f, vicini[:n,j])  # Pass existing vicini order
        lati[:n,:,j,i] = sides
        
        # Use pseudo-inverse for better numerical stability
        initial_vectors = lati[:n,:,j,0].T
        defgrad = sides.T @ np.linalg.pinv(initial_vectors, rcond=1e-10)  # Add rcond threshold
        F[:,:,j,i] = defgrad
        C[:,:,j,i] = defgrad.T @ defgrad
        B[:,:,j,i] = defgrad @ defgrad.T
        
        # Calculate strain tensors
        E[:,:,j,i] = 0.5 * (C[:,:,j,i] - np.eye(3))
        e[:,:,j,i] = 0.5 * (np.eye(3) - np.linalg.inv(B[:,:,j,i]))
        
        # Calculate areas and deformation
        sides_rolled = np.roll(sides, -1, axis=0)
        cross_products = np.cross(sides, sides_rolled)
        A[j,i] = 0.5 * np.sum(np.sqrt(np.sum(cross_products**2, axis=1)))
        Adef[j,i] = 100 * (A[j,i] - A[j,0]) / A[j,0]
        
        # Compute eigenvalues and eigenvectors
        comp_princ, dir_princ = np.linalg.eig(E[:,:,j,i])
        
        # Sort eigenvalues and eigenvectors
        idx = np.argsort(comp_princ)[::-1]  # descending order
        comp_princ = comp_princ[idx]
        dir_princ = dir_princ[:, idx]
        
        # Store results
        Ecp[j,:,i] = comp_princ
        Edp[:,:,j,i] = dir_princ

print("Processing complete")

# print(A) ok
# print(Adef) ok
# print(Ecp[:,:,1]) ok
print(Edp[:,:,937,24]) #has different signs than MATLAB

# At the start of plotting code

# Calculate min/max values for Adef colormap
Adefmax = np.max(Adef)
Adefmin = np.min(Adef)

# Calculate coordinate ranges
coordmin = np.zeros((fasi, 3))
coordmax = np.zeros((fasi, 3))
for j in range(fasi):
    coordmin[j,:] = np.min(v[:,:,j], axis=0)
    coordmax[j,:] = np.max(v[:,:,j], axis=0)
minimi = np.min(coordmin, axis=0)
massimi = np.max(coordmax, axis=0)


# Create figure
fig = plt.figure(figsize=(20, 8))
fig.suptitle('Areal strains')

# Create subplots for each frame
for j in range(f2):
    ax = fig.add_subplot(3, 9, j+1, projection='3d')
    
    # Plot triangular surface
    surf = ax.plot_trisurf(
        v[:,0,j], v[:,1,j], v[:,2,j],
        triangles=f,
        cmap='viridis',
        edgecolor='none',
        alpha=1,
        shade=True,
        array=Adef[:,j]
    )
    
    # Set axis limits and properties
    ax.set_xlim(minimi[0], massimi[0])
    ax.set_ylim(minimi[1], massimi[1])
    ax.set_zlim(minimi[2], massimi[2])
    ax.set_box_aspect([1,1,1])
    
    # Set color scale
    surf.set_clim(Adefmin, Adefmax)
    
    # Add lighting effect
    ax.view_init(elev=30, azim=45)
    
    # Add title
    ax.set_title(f'Frame-{j:02d}')

# Add colorbar to last subplot
cax = fig.add_subplot(3, 9, f2+1)
plt.colorbar(surf, ax=cax)

# Link 3D views
def on_move(event, axes):
    if event.inaxes in axes:
        for ax in axes:
            if ax != event.inaxes:
                ax.view_init(elev=event.inaxes.elev, azim=event.inaxes.azim)
        fig.canvas.draw_idle()

axes = [ax for ax in fig.axes if isinstance(ax, Axes3D)]
fig.canvas.mpl_connect('motion_notify_event', partial(on_move, axes=axes))


# Define the on_move function
def on_move(event, axes):
    if event.inaxes in axes:
        for ax in axes:
            if ax != event.inaxes:
                ax.view_init(elev=event.inaxes.elev, azim=event.inaxes.azim)
        fig2.canvas.draw_idle()


# Create figure for max principal strains
fig2 = plt.figure(figsize=(20, 8))
fig2.suptitle('Max Principal strains')

# Calculate min/max values
massimo = np.max(Ecp[:,0,:])  # equivalent to max(max(squeeze(Ecp(:,1,:))))
minimo = np.min(Ecp[:,0,:])

# Create subplots for each frame
for j in range(f2):
    if j == 0:
        ax = fig2.add_subplot(3, 9, j+1)
    else:
        ax = fig2.add_subplot(3, 9, j+1, projection='3d')
        
        # Plot triangular surface
        surf = ax.plot_trisurf(
            v[:,0,j], v[:,1,j], v[:,2,j],
            triangles=f,
            cmap='viridis',
            edgecolor='none',
            alpha=1,
            shade=True,
            array=Ecp[:,0,j]
        )
        
        # Correct indexing for Edp - match MATLAB's squeeze operation
        dptoplot = Edp[:,0,:,j].T  # Get first principal direction vectors

        ax.quiver(
            v[:,0,j], v[:,1,j], v[:,2,j],  # positions
            dptoplot[:,0], dptoplot[:,1], dptoplot[:,2],  # vector components
            length=0.1,
            normalize=True
        )
        
        # Set axis limits and properties
        ax.set_xlim(minimi[0], massimi[0])
        ax.set_ylim(minimi[1], massimi[1])
        ax.set_zlim(minimi[2], massimi[2])
        ax.set_box_aspect([1,1,1])
        
        # Set color scale
        surf.set_clim(minimo, massimo)
        
        # Add lighting effect
        ax.view_init(elev=30, azim=45)
        
        # Add title
        ax.set_title(f'Frame-{j:02d}')

# Add colorbar to last subplot
cax = fig2.add_subplot(3, 9, f2+1)
plt.colorbar(surf, ax=cax)

# Link 3D views
axes = [ax for ax in fig2.axes if isinstance(ax, Axes3D)]
fig2.canvas.mpl_connect('motion_notify_event', partial(on_move, axes=axes))


# Define the on_move function
def on_move(event, axes):
    if event.inaxes in axes:
        for ax in axes:
            if ax != event.inaxes:
                ax.view_init(elev=event.inaxes.elev, azim=event.inaxes.azim)
        fig3.canvas.draw_idle()

# Create figure for min principal strains
fig3 = plt.figure(figsize=(20, 8))
fig3.suptitle('Min Principal strains')

# Calculate min/max values
massimo = np.max(Ecp[:,2,:])  # Using index 2 for 3rd component
minimo = np.min(Ecp[:,2,:])

# Get max areal strain 
Adefmax = np.max(Adef)

# Create subplots for each frame
for j in range(f2):
    if j == 0:
        ax = fig3.add_subplot(3, 9, j+1)
    else:
        ax = fig3.add_subplot(3, 9, j+1, projection='3d')
        
        # Plot triangular surface
        surf = ax.plot_trisurf(
            v[:,0,j], v[:,1,j], v[:,2,j],
            triangles=f,
            cmap='viridis',
            edgecolor='none',
            alpha=1,
            shade=True,
            array=Ecp[:,2,j]
        )
        
        # Correct indexing for Edp - match MATLAB's squeeze operation
        dptoplot = Edp[:,2,:,j].T  # Get first principal direction vectors

        ax.quiver(
            v[:,0,j], v[:,1,j], v[:,2,j], # positions
            dptoplot[:,2], dptoplot[:,2], dptoplot[:,2], # vector components
            length=0.1,
            normalize=True
        )
        
        # Set axis limits and properties
        ax.set_xlim(minimi[0], massimi[0])
        ax.set_ylim(minimi[1], massimi[1])
        ax.set_zlim(minimi[2], massimi[2])
        ax.set_box_aspect([1,1,1])
        
        # Set color scale
        surf.set_clim(minimo, massimo)
        
        # Add lighting effect
        ax.view_init(elev=30, azim=45)
        
        # Add title
        ax.set_title(f'Frame-{j:02d}')

# Add colorbar to last subplot
cax = fig3.add_subplot(3, 9, f2+1)
plt.colorbar(surf, ax=cax)

# Link 3D views
axes = [ax for ax in fig3.axes if isinstance(ax, Axes3D)]
fig3.canvas.mpl_connect('motion_notify_event', partial(on_move, axes=axes))

plt.show(block=True)  # Block until all figures are closed
