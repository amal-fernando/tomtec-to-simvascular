import tkinter as tk
from tkinter import filedialog
import numpy as np
from funzioni import riordina, timeplot, resample_u, pv_to_np, np_to_pv, write_motion
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

#=========================================
# 2. Create folder structure to save files
#=========================================

# Create the base directory for the dataset
subfolders = [
    os.path.join(radice_dataset[:-1], "mesh"),
    os.path.join(radice_dataset[:-1], "mesh", "mesh-surfaces"),
    os.path.join(radice_dataset[:-1], "plots")
]

# Create the subfolders if they do not exist
for folder in subfolders:
    os.makedirs(folder, exist_ok=True)

#==============================================================================
# 3. Load the .ucd files and extract node coordinates and triangle connectivity
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
normalplot(v0, f0, 1, os.path.join(radice_dataset[:-1], "plots", "normalplot_coarse.obj"), show=False) # plot delle normali alla mesh rada \decommentare


#============================================================
# 4. Reorder and extend the time vector and the displacements
#============================================================

# rr_true = t0[-1]
[t1,v1] = riordina(t0,v0,ed,rr) # t1 is the reordered time vector, v1 is the reordered displacements matrix

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

#=================================
# 5. Mesh evaluation and remeshing
#=================================

mesh = np_to_pv(v2[:,:,0],f0) # Load the mesh from the numpy arrays
# mesh.save("coarse_mesh.vtp")  # Save the coarse mesh for reference
avg,_,_ = mesh_quality(mesh)

if avg['aspect_ratio'] > 1.5 or avg['radius_ratio'] > 1.5:
    print("The mesh quality is poor.")
    print("Remeshing should be done.")
    flag = True
else:
    print("The mesh quality is good.")
    print("No need to remesh.")


if flag:
    # edges = mesh.extract_all_edges().compute_cell_sizes()
    # avg_edge = edges['Length'].mean()
    # hausd = avg_edge  # Use this instead of a hardcoded 0.3
    remesh = mmg_remesh(mesh, hausd=0.1,verbose=True  ) # Remesh the mesh using MMG

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
    normalplot(vert, fac, 0, os.path.join(radice_dataset[:-1], "plots", "normalplot_fine.obj"), show=False)  # Plot the normals of the remeshed mesh
else:
    remesh = mesh.triangulate() # Ensure the mesh is triangulated

# Compute the average edge length of the remeshed surface
edges = remesh.extract_all_edges()
length = edges.compute_cell_sizes(length=True).cell_data['Length']
l = np.mean(length)

#======================================================
# 6. Generate the volume mesh from the remeshed surface
#======================================================

# TetGen options (preserve surface, good quality)
a = l**3 / (6 * np.sqrt(2))
tetgen_options = f"pq1.2a{a}Y"

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
# plotter.export_gltf(os.path.join(radice_dataset[:-1], "plots", "volume.glb"))
# plotter.save_graphic(os.path.join(radice_dataset[:-1], "plots", "volume.pdf"))
plotter.close() # plotter.show() # Show the plotter window

# Save as VTU
grid.point_data["GlobalNodeID"] = np.arange(grid.points.shape[0]) + 1
grid.cell_data["GlobalElementID"] = np.arange(grid.n_cells) + 1
grid.cell_data["ModelRegionID"] = np.ones(grid.n_cells, dtype=int)  # Set to one for all cells
grid.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-complete.mesh.vtu"))

surf_mesh = grid.extract_geometry()

# Verify if the surface mesh has been modified
print(f"Surface mesh has been modified: {not np.array_equal(surf_mesh.points, remesh.points)}")

inlet, outlet, wall, surface = get_bounds(surf_mesh) # Extract inlet, outlet, and wall surfaces from the remeshed surface mesh

# Save the inlet, outlet, and wall surfaces
inlet.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-surfaces", "inlet.vtp"))
outlet.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-surfaces", "outlet.vtp"))
wall.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-surfaces", "wall.vtp"))
surface.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-complete.exterior.vtp"))
#==================================================================================
# 7. Remapping the coarse meshes of the time interpolated data on the remeshed mesh
#==================================================================================

v3_0, f1 = pv_to_np(surf_mesh)  # Convert remeshed mesh to numpy arrays

# Initialize v4 with the appropriate dimensions
v3 = np.zeros((v3_0.shape[0], v3_0.shape[1], v2.shape[2]))

tr = f0
p = v3_0

# Loop through each time step
for i in range(v2.shape[2] - 1):
    vr = v2[:, :, i]
    ur = v2[:, :, i + 1] - v2[:, :, i]

    # Assuming resample_u is already defined, and returns pp and upp
    pp, upp = resample_u(vr, tr, ur, p)

    # Store the results in v4
    v3[:, :, i] = pp
    p = upp + pp
    v3[:, :, i + 1] = p

# timeplot(v3,f1)
#===========================================================
# 8. Interpolation of the data to create intermediate frames
#===========================================================

num_intermedie = 4  # Number of intermediate frames to insert between each original frame
frames_3  = frames_2 + (frames_2- 1) * num_intermedie

# Initialize v4 with the appropriate dimensions
v4 = np.zeros((v3.shape[0], v3.shape[1], frames_3))
v_cubic = np.zeros(v4.shape)  # For cubic interpolation
v_linear = np.zeros(v4.shape)  # For linear interpolation
v_pchip = np.zeros(v4.shape)  # For PCHIP interpolation

# Create a time vector for the new frames
t3 = np.linspace(t2[0], t2[-1], frames_3)

# Cubic interpolation
for i in range(v3.shape[0]):  # For each node
    for dim in range(3):  # For each spatial dimension (x, y, z)
        coordinates = v3[i, dim, :] # Extract the coordinates in the current dimension
        spline = CubicSpline(t2, coordinates) # Create an interpolator for the current node and dimension
        v_cubic[i, dim, :] = spline(t3) # Interpolate to the new time points

# Linear interpolation
for i in range(v3.shape[0]):  # For each node
    for dim in range(3):  # For each spatial dimension (x, y, z)
        coordinates = v3[i, dim, :]
        linear_interp = interp1d(t2, coordinates, kind='linear')
        v_linear[i, dim, :] = linear_interp(t3)

# PCHIP interpolation
for i in range(v3.shape[0]):
    for dim in range(3):
        coordinates = v3[i, dim, :]
        pchip_interp = PchipInterpolator(t2, coordinates)
        v_pchip[i, dim, :] = pchip_interp(t3)

# Fourier interpolation
v_fourier = fourier(v_pchip, t2, 0)

# Create the plot for the original and interpolated volumes
# plt.figure()
# plt.plot(t2,volume(v3,f1), '*', label='volOriginal')
# plt.plot(t3,volume(v_cubic, f1), label='volCubic')
# plt.plot(t3,volume(v_linear, f1), label='volLinear')
# plt.plot(t3,volume(v_pchip, f1), label='volPchip')
# plt.plot(t3,volume(v_fourier, f1), label='volFourier')
# plt.legend()
# plt.grid(True)
# plt.xlabel('Time Frame')
# plt.ylabel('Volume')
# plt.title('Volume Over Time')
# plt.savefig(os.path.join(radice_dataset[:-1], "plots", "volume_interpolation.png"))
# plt.show()

v4 = v_fourier  # Use the Fourier interpolated data for further processing
# normalplot(v4, f1, 1)  # Plot the normals of the interpolated mesh

#==========================================================
# 9. Generate the input files for the simulation
#==========================================================
V = volume(v4, f1)  # Calculate the volume for the Fourier interpolated data
# Create a time vector for the new frames
time = t3.flatten()/1000
flow_rate = np.gradient(V, time)  # Automatically handles variable time steps N.B. 1 mm3/ms = 1 cm3/s

# Save to .flow file
with open(os.path.join(radice_dataset[:-1], "inlet.flow"), "w") as f:
    f.write(f"{len(t3.flatten())} {inlet.n_points}\n")  # Write number of time steps and nodes
    for t, q in zip(t3.flatten()/1000, flow_rate):
        # Convert q to a scalar and write to file
        f.write(f"{t:.7f} {q:.7f}\n")


# Calculate the displacement
displacement = v4 - v4[:,:,0][:, :, np.newaxis]

write_motion(displacement, t3, wall, surface, os.path.join(radice_dataset[:-1], "wall" ))
write_motion(displacement, t3, inlet, surface, os.path.join(radice_dataset[:-1], "inlet"))
write_motion(displacement, t3, outlet, surface, os.path.join(radice_dataset[:-1], "outlet"))


''' Genero file bct.vtp per 'inlet' '''

# === Caricamento dati ===
inlet = pv.read(os.path.join(radice_dataset[:-1], "mesh", "mesh-surfaces", "inlet.vtp"))
points = inlet.points
node_ids = inlet.point_data["GlobalNodeID"]
normals = np.array(inlet.point_normals)


time = t3.flatten()/1000  # Convert to seconds
nl = len(time)

# === Calcolo area ===
mesh = pv.PolyData(points)
tri = mesh.delaunay_2d()
area = tri.area

# === Centroide dell'inlet ===
centroid = points.mean(axis=0)

# === Raggio massimo (per profilo parabolico) ===
radii = np.linalg.norm(points[:, :2] - centroid[:2], axis=1)
r_max = radii.max()


# === Lista di tutte le matrici di velocità ===
velocità = []  # contiene velocity_0.0000, velocity_0.0001, ...

'''
# === Preparazione file ===
lines = [f"{len(points)} {nl}"]

for i, pt in enumerate(points):
    x, y, z = pt
    nn = int(node_ids[i])
    normal = normals[i]

    # Distanza radiale dal centroide nel piano (x, y)
    r = np.linalg.norm(pt[:2] - centroid[:2])
    shape_factor = 1 - (r / r_max) ** 2  # parabola

    lines.append(f"{x:.6f} {y:.6f} {z:.6f} {nl} {nn}")
    for j in range(nl):
        Q = flow_rate[j]
        v_mean = Q / area
        v_mag = 2 * v_mean * shape_factor  # parabola: vmax = 2 * v_mean
        vx, vy, vz = normal * v_mag
        t = time[j]
        lines.append(f"{vx:.6f} {vy:.6f} {vz:.6f} {t:.6f}")
'''

# === Ciclo sui tempi per costruire le matrici ===
for j in range(nl):
    Q = flow_rate[j]
    v_mean = Q / area
    t = time[j]

    # Matrice velocità per questo istante: N righe, 3 colonne (vx, vy, vz)
    velocity_matrix = []

    for i, pt in enumerate(points):
        r = np.linalg.norm(pt[:2] - centroid[:2])
        shape_factor = 1 - (r / r_max) ** 2  # profilo parabolico
        v_mag = 2 * v_mean * shape_factor
        vx, vy, vz = normals[i] * v_mag
        velocity_matrix.append([vx, vy, vz])

    velocity_matrix = np.array(velocity_matrix)

    # Aggiunge il campo al file mesh
    field_name = f"velocity_{t/1000:.4f}"
    inlet.point_data[field_name] = velocity_matrix * -1

    # # Crea una variabile dinamica: es. velocity_0.0002
    # var_name = f"velocity_{t:.4f}"
    # globals()[var_name] = velocity_matrix
    #
    # # Aggiungi alla lista globale
    # velocità.append(velocity_matrix)

# === Salvataggio in file VTP ===
inlet.save(os.path.join(radice_dataset[:-1], "bct.vtp"))


print("Done!")

# Generate the .inp file for the simulation

config_text = f"""\
#----------------------------------------------------------------
# General simulation parameters

Continue previous simulation: 0
Number of spatial dimensions: 3
Number of time steps: 3076
Time step size: 0.001
Spectral radius of infinite time step: 0.50
Searched file name to trigger stop: STOP_SIM

#Save results in folder: {radice_dataset[:-1]} 
Save results to VTK format: 1
Name prefix of saved VTK files: result
Increment in saving VTK files: 40
Start saving after time step: 1
Increment in saving restart files: 50
Convert BIN to VTK format: 0

Simulation requires remeshing: F

Verbose: 1
Warning: 0
Debug: 0

#----------------------------------------------------------------
# Mesh & Domains

Add mesh: lumen {{
   Mesh file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/mesh/mesh-complete.mesh.vtu
   Add face: lumen_wall {{
      Face file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/mesh/mesh-surfaces/wall.vtp
   }}
   Add face: lumen_inlet {{
      Face file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/mesh/mesh-surfaces/inlet.vtp
   }}
   Add face: lumen_outlet {{
      Face file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/mesh/mesh-surfaces/outlet.vtp
   }}
   Domain: 1
}}

#----------------------------------------------------------------
# Equations

Add equation: FSI {{
   Coupled: 1
   Min iterations: 3
   Max iterations: 20
   Tolerance: 1e-3
   
   #Remesher: Tetgen {{
   #   Max edge size: lumen {{ val: 2. }}
   #   Min dihedral angle: 10
   #   Max radius ratio: 1.3
   #   Remesh frequency: 50
   #   Frequency for copying data: 1
   #}}


   Domain: 1 {{
      Equation: fluid
      Density: 0.00106
      Viscosity: Constant {{Value: 0.004}}
      Backflow stabilization coefficient: 0.2
   }}

   Domain: 2 {{
      Equation: struct
      Constitutive model: stVK
      Density: 1
      Elasticity modulus: 100000
      Poisson ratio: 0.4
   }}

   Output: Spatial {{
      Displacement: f
      Velocity: t
      Pressure: t
      WSS: t
      Vorticity: t
      Absolute_velocity: t
   }}

   Output: Alias {{
      Displacement: FS_Displacement
   }}

   LS type: GMRES {{
      Preconditioner: FSILS
      Max iterations: 100
      Tolerance: 1e-4
      Absolute tolerance: 1e-4
      Krylov space dimension: 100
   }}

   #Initialize RCR from flow: t
   #Add BC: lumen_inlet {{
      #Type: Dir
      #Time dependence: Unsteady
      #Temporal values file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/inlet.flow
      #Profile: Parabolic
      #Impose flux: 1
      #Time dependence: General
      ##BCT file path: /global-scratch/bulk_pool/afernando/Docker/Tubo/bct.vtp
   #}}
   
   Add BC: lumen_inlet {{
      Type: Dirichlet 
      Time dependence: General
      Temporal and spatial values file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/inlet_displacement.dat
      Profile: Flat
      Zero out perimeter: 0
      Impose flux: 0
      Impose on state variable integral: 1
   }}
   
   Add BC: lumen_outlet {{
      Type: Neu
      Time dependence: RCR
      RCR values: "0.010299941677760955, 1.5940597148021995, 1.182852092486154"
      Distal pressure: 0.0
      Zero out perimeter: 0
      Impose flux: 0
   }}

   Add BC: lumen_wall {{
      Type: Dirichlet
      Time dependence: General
      Temporal and spatial values file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/wall_displacement.dat
      Profile: Flat
      Zero out perimeter: 0
      Impose flux: 0
      Impose on state variable integral: 1
   }}
}}

Add equation: mesh {{
   Coupled: 1
   Min iterations: 2
   Max iterations: 5
   Tolerance: 1e-4
   Poisson ratio: 0.3

   LS type: CG {{
      Preconditioner: FSILS
      Tolerance: 1e-4
   }}

   Output: Spatial {{
      Displacement: t
   }}

   Add BC: lumen_inlet {{
      Type: Dirichlet 
      Time dependence: General
      Temporal and spatial values file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/inlet_displacement.dat
      Profile: Flat
      Zero out perimeter: 0
      Impose flux: 0
      Impose on state variable integral: 1
   }}
   
   Add BC: lumen_outlet {{
      Type: Dirichlet 
      Time dependence: General
      Temporal and spatial values file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/outlet_displacement.dat
      Profile: Flat
      Zero out perimeter: 0
      Impose flux: 0
      Impose on state variable integral: 1
   }}
   
   Add BC: lumen_wall {{
      Type: Dirichlet 
      Time dependence: General
      Temporal and spatial values file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/wall_displacement.dat
      Profile: Flat
      Zero out perimeter: 0
      Impose flux: 0
      Impose on state variable integral: 1
   }}
}}
"""

# Scrive il testo in un file
with open(os.path.join(radice_dataset[:-1], f"{radice_dataset[:-1]}_fsi.inp" ), "w") as file:
    file.write(config_text)

print("Script completed successfully.")