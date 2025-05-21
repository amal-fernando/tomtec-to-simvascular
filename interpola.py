import tkinter as tk
from tkinter import filedialog
import numpy as np
from funzioni import riordina, timeplot, resample_u
from funzioni import volume, normalplot
from funzioni import fourier
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import pyvista as pv
import xml.etree.ElementTree as ET
import os
import tetgen
from pyvista import CellType

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
    fasi = int(line[-1])
    
    # Read third line for number of nodes 
    line = fid.readline().strip().split()
    N_nodi = int(line[-1])
    
    # Read fourth line for number of triangles
    line = fid.readline().strip().split()
    N_elementi = int(line[-1])

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

    t = np.zeros((fasi, 1))  # Initialize array to store time values

    # Loop through frames to find matching time
    for i in range(fasi):
        line = fid.readline().strip()
        t[i] = float(line.split()[0])  # Get first number from line

    print(f"Time vector: {t}")

# Retrive the coordinates of the nodes and the connectivity of the triangles
# Initialize arrays with zeros
v = np.zeros((N_nodi, 3, fasi)) #3D matrix to store coordinates of each node at each time point
f = np.zeros((N_elementi, 3)) #2D matrix to store connectivity of each triangle (not time dependent)

print("Start loading files")

# First loop: from start_frame to fasi
for i in range(fasi):
    # Generate filename
    data = f"{radice_dataset}{i:02d}.ucd"
    full_path = os.path.join(path, data)
    # print(f"{full_path=}")
    
    # Load node coordinates
    temp_data = np.loadtxt(full_path, skiprows=1, delimiter=' ', usecols=(1,2,3), max_rows=N_nodi)
    v[:,:,i] = temp_data[:N_nodi]
    
    # Load face connectivity only for first iteration
    if i == 0:
        temp_faces = np.loadtxt(full_path, skiprows=N_nodi+1, usecols=(3,4,5), max_rows=N_elementi)
        f[:,:] = temp_faces[:N_elementi]  # Convert to 1-based indexing
        f = f.astype(np.int64)

print('v.shape =', v.shape)
# print(v[:,:,0])
print('f.shape =', f.shape)
# print(f)

print("Loading Done")

u_orig = v - v[:, :, 0][:, :, np.newaxis]  # Original displacements
np.save('u_orig.npy', u_orig) # Save the original displacements
np.save('v0_orig.npy', v[:,:,0]) # Save the original coordinates of the initial frame
np.save('f_orig.npy', f) # Save the original connectivity
np.save('time.npy', t) # Save the time vector

# timeplot(v,f) # plot della mesh rada nel tempo \decommentare
# normalplot(v, f, 1) # plot delle normali alla mesh rada \decommentare

# Estensione del segnale (spostamenti)

# # Load the .mat file of dense matrix
# data = scipy.io.loadmat('v2.mat')
# v2 = data['vertices']
# data = scipy.io.loadmat('f2.mat')
# f2 = data['faces']-1

# Load the remeshed mesh
mesh = pv.read("remeshed_mesh.stl")

v2 = np.array(mesh.points)
f2 = mesh.faces.reshape(-1, 4)[:,1:]  # Remove first column (number of vertices)
v2 = v2[:, :, np.newaxis]

# timeplot(v2,f2)
# normalplot(v2, f2,0)

# Initialize matrix3 with the appropriate dimensions
v3 = np.zeros((v2.shape[0], v2.shape[1], v.shape[2]))

tr = f
p = v2[:, :, 0]

# Loop through each time step
for i in range(v.shape[2] - 1):
    vr = v[:, :, i]
    ur = v[:, :, i + 1] - v[:, :, i]

    # Assuming resample_u is already defined, and returns pp and upp
    pp, upp = resample_u(vr, tr, ur, p)

    # Store the results in v3
    v3[:, :, i] = pp
    p = upp + pp
    v3[:, :, i + 1] = p

# timeplot(v3,f2)

rr_true = t[-1]
[time,matrix1] = riordina(t,v3,ed,rr)


# Estensione del segnale
# Numero totale di sequenze (inclusa quella originale)
num_sequences = 3
nt = len(time) - 1  # Elimino l'ultimo dato, che deve essere inserito solo in coda alla sequenza

# Duplica la matrice iniziale
matrix2 = np.tile(matrix1[:, :, :nt], (1, 1, num_sequences + 1))  # Crea un ciclo di troppo inizialmente
matrix2 = matrix2[:, :, :-nt+1]  # Elimina i dati dopo il duplicato del primo in coda alla sequenza

# print('matrix2.shape =', matrix2.shape)
# print(matrix2[:,:,75])

fasi = matrix2.shape[2]

# Costruzione del vettore time2
time2 = np.zeros(num_sequences * nt + 1)

for i in range(1, num_sequences + 1):
    time2[(i - 1) * nt:i * nt] = time[:nt] + (i - 1) * rr
time2[-1] = rr * num_sequences


# print('time2.shape =', time2.shape)
# print(time2)



volumeOri = np.zeros(matrix2.shape[2])
# Calcolo del volume
for i in range(matrix2.shape[2]):
    mesh = pv.PolyData(matrix2[:, :, i], np.hstack([np.full((f2.shape[0], 1), 3), f2]))
    volumeOri[i] = mesh.volume


volOriginal = volume(matrix2, f2)
print('volOriginale.shape =', volOriginal.shape)
print(volOriginal)

# Create the plot
plt.figure()
plt.plot(time2,volOriginal, label='volOriginal')
plt.legend()
plt.grid(True)
plt.xlabel('Time [ms]')
plt.ylabel('Volume [mm^3]')
plt.title('Volume Over Time')
plt.show()

# Parametri di input
num_intermedie = 4

# Inizializzazione del vettore vol
vol = np.zeros(len(volOriginal) + (len(volOriginal) - 1) * num_intermedie)
tempo = np.zeros(len(vol))
index = 0  # Indice per riempire vol

for i in range(len(volOriginal)):
    vol[index] = volOriginal[i]  # Copia il valore originale
    tempo[index] = time2[i]  # Copia il valore originale
    index += 1  # Avanza l'indice
    
    if i < len(volOriginal) - 1:  # Evita l'ultimo caso
        vol[index:index+num_intermedie] = np.nan  # Riempie con NaN
        tempo[index:index+num_intermedie] = np.linspace(time2[i], time2[i+1], num_intermedie + 2)[1:-1]
        index += num_intermedie  # Avanza l'indice dopo gli NaN

volOriginal = vol
print('volOriginale.shape =', volOriginal.shape)
print(volOriginal)

timeOriginal = tempo
print('timeOriginale.shape =', timeOriginal.shape)
print(timeOriginal)

new_time_frames = len(timeOriginal)  # Number of time frames for interpolation

# Prepare an array to store the interpolated coordinates
matrixCubic = np.zeros((matrix2.shape[0], 3, new_time_frames))

# Interpolate each point (node) separately
for i in range(matrix2.shape[0]):  # For each node
    for dim in range(3):  # For each spatial dimension (x, y, z)
        # Extract the coordinates in the current dimension
        coordinates = matrix2[i, dim, :]
        
        # Create a cubic spline interpolator for the current node and dimension
        spline = CubicSpline(time2, coordinates)
        
        # Interpolate to the new time points
        timeCubic = np.linspace(time2[0], time2[-1], new_time_frames)
        matrixCubic[i, dim, :] = spline(timeOriginal)

# interpolated_coordinates now contains the interpolated cloud of points

print('matrixCubic.shape =', matrixCubic.shape)
print(matrixCubic[:,:,75])

# print('timeCubic.shape =', timeCubic.shape)
# print(timeCubic)

matrixLinear = np.zeros((matrix2.shape[0], 3, new_time_frames))

# Linear interpolation for each point in the 3D space
for i in range(matrix2.shape[0]):  # For each node
    for dim in range(3):  # For each spatial dimension (x, y, z)
        coordinates = matrix2[i, dim, :]
        linear_interp = interp1d(time2, coordinates, kind='linear')
        matrixLinear[i, dim, :] = linear_interp(timeOriginal)


matrixPchip = np.zeros((matrix2.shape[0], 3, new_time_frames))

# Use PCHIP interpolation for each point
for i in range(matrix2.shape[0]):
    for dim in range(3):
        coordinates = matrix2[i, dim, :]
        pchip_interp = PchipInterpolator(time2, coordinates)
        matrixPchip[i, dim, :] = pchip_interp(timeOriginal)


# Fourier interpolation
#matrixFourier = np.zeros((matrix2.shape[0], 3, new_time_frames))
[matrixFourier,timeFourier,fas] = fourier(matrixPchip, time2, 0)

# timeplot(matrixFourier,f2)

# Create the plot
plt.figure()
plt.plot(timeOriginal,volOriginal, '*', label='volOriginal')
plt.plot(timeOriginal,volume(matrixCubic, f2), label='volCubic')
plt.plot(timeOriginal,volume(matrixLinear, f2), label='volLinear')
plt.plot(timeOriginal,volume(matrixPchip, f2), label='volPchip')
plt.plot(timeOriginal,volume(matrixFourier, f2), label='volFourier')
plt.legend()
plt.grid(True)
plt.xlabel('Frame')
plt.ylabel('Volume')
plt.title('Volume Over Time')
plt.show()

# Save the interpolated coordinates to a .vtu file
# Define input and output folders
output_dir = "output_volume_meshes/"
os.makedirs(output_dir, exist_ok=True)

# Load your data
vertices = matrixFourier # np.load("vertices.npy")  # Shape: Nx3xT
connectivity = f2 # np.load("connectivity.npy")  # Shape: Mx3
inlet_nodes = np.load("inlet_nodes.npy")  # List of node indices
outlet_nodes = np.load("outlet_nodes.npy")  # List of node indices

# Convert inlet_nodes and outlet_nodes to integer type
inlet_nodes = inlet_nodes.astype(np.int32)
outlet_nodes = outlet_nodes.astype(np.int32)

num_time_steps = vertices.shape[2]  # T time frames

# TetGen options (preserve surface, good quality)
tetgen_options = "pq1.2a0.005"

# Process each time step
for t in range(num_time_steps):
    print(f"Processing time step {t}/{num_time_steps - 1}")

    # Extract current frame vertices
    vtx = vertices[:, :, t]

    # Create a PyVista PolyData surface mesh
    surface_mesh = pv.PolyData(vtx, np.hstack((np.full((connectivity.shape[0], 1), 3), connectivity)).astype(np.int64))

    # Clean and repair the surface mesh if needed
    if not surface_mesh.is_manifold:
        print("Warning: The surface mesh is not manifold!")
        surface_mesh = surface_mesh.clean()
    surface_mesh = surface_mesh.clean()

    # Convert to TetGen format
    tet = tetgen.TetGen(surface_mesh)

    # Perform tetrahedralization
    try:
        tet.tetrahedralize(tetgen_options)
    except RuntimeError as e:
        print(f"Failed to tetrahedralize: {e}")
        continue

    # Get volume mesh data
    tetra_nodes = tet.node
    tetra_elements = tet.grid.cells.reshape(-1, 5)[:, 1:]  # Fix element extraction

    # Convert to PyVista UnstructuredGrid for saving
    # Create a PyVista UnstructuredGrid
    vol_mesh = pv.UnstructuredGrid((np.hstack((np.full((tetra_elements.shape[0], 1), 4), tetra_elements))), np.full(tetra_elements.shape[0], CellType.TETRA, dtype=np.uint8), tetra_nodes)

    # Label inlets and outlets
    vol_mesh.point_data["boundary_conditions"] = np.zeros(tetra_nodes.shape[0])
    vol_mesh.point_data["boundary_conditions"][inlet_nodes] = 1  # Label inlet nodes
    vol_mesh.point_data["boundary_conditions"][outlet_nodes] = 2  # Label outlet nodes

    # Save volume mesh
    output_file = os.path.join(output_dir, f"mesh_{t:04d}.vtk")
    vol_mesh.save(output_file)

    print(f"Saved: {output_file}")

print("✅ Volume meshing completed for all time steps!")

# Salviamo ogni frame come file .vtu
pvd_filename = "mesh_sequence.pvd"
vtu_files = []

for t in range(matrixFourier.shape[2]):
    points = matrixFourier[:, :, t]  # Estrai nodi al frame t
    mesh = pv.PolyData(matrixFourier[:, :, t], np.hstack([np.full((f2.shape[0], 1), 3), f2]))
    grid = pv.UnstructuredGrid(mesh)

    vtu_filename = f"mesh_{t}.vtu"
    grid.save(vtu_filename)
    vtu_files.append((t, vtu_filename))

# Creiamo il file .pvd
pvd_root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
collection = ET.SubElement(pvd_root, "Collection")

for time, filename in vtu_files:
    ET.SubElement(collection, "DataSet", timestep=str(time), group="", part="0", file=filename)

# Salva il file .pvd
tree = ET.ElementTree(pvd_root)
tree.write(pvd_filename)

print(f"Sequenza salvata in {pvd_filename}")

print('Done')