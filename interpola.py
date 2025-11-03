import tkinter as tk
from tkinter import filedialog, simpledialog
import numpy as np
from funzioni import riordina, timeplot, resample_u, pv_to_np, np_to_pv, write_motion
from funzioni import volume, normalplot, mmg_remesh, mesh_quality, nearest_multiple
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
import time
from tqdm.auto import tqdm
from funzioni import extract_surface_region_from_old

start_time = time.time()  # ⏱️
'''
This script is used to load a set of .ucd files, extract the coordinates of the nodes and the connectivity of the 
triangles, and then perform interpolation on the data. The script also includes functions for visualizing the data and 
saving it in different formats.
'''

#==================================================
# 0. Find all header files in the current directory
#==================================================
print("Script started")
# Directory principale
root = tk.Tk()
root.withdraw()

# Open file dialog to select header file
root_dir = filedialog.askdirectory(
    title="Select header file directory"
)
# root_dir = r'C:/Users/amal1/Desktop/Tesi/Dati RV TomTec'

# Dizionario per salvare i path
paths = {}

# Trova tutte le cartelle Pxxxx
patient_dirs = sorted([
    d for d in os.listdir(root_dir)
    if d.startswith('P') and os.path.isdir(os.path.join(root_dir, d))
])

for patient in patient_dirs:
    paths[patient] = {}

    for phase in ['pre', 'post']:
        subfolder = f'{patient}_{phase}'
        header_path = os.path.join(root_dir, patient, subfolder, 'Tomtec', f'{subfolder}_header.txt')

        if os.path.exists(header_path):
            paths[patient][phase] = header_path
        else:
            print(f'[ATTENZIONE] File mancante: {header_path}')
            paths[patient][phase] = None  # oppure salta, se preferisci

header_files_path = [
    path for patient in paths.values()
    for path in patient.values()
    if path is not None
]

# ============================================
# Permetti all’utente di scegliere da dove iniziare
# ============================================
if not header_files_path:
    print("Nessun file header trovato.")
    exit()

# Mostra l’elenco numerato
print("\nFile trovati:")
for i, path in enumerate(header_files_path):
    print(f"{i}: {path}")

# Chiedi l'indice di partenza
start_index = simpledialog.askinteger(
    "Indice di partenza",
    f"Inserisci l'indice da cui iniziare (0 - {len(header_files_path)-1}):",
    minvalue=0,
    maxvalue=len(header_files_path)-1
)

if start_index is None:
    print("Nessun indice selezionato. Uscita.")
    exit()

#================================================
# 1. Parse the header file to extract information
#================================================

# Initialize tkinter but hide main window
# root = tk.Tk()
# root.withdraw()
#
# # Open file dialog to select header file
# file_path = filedialog.askopenfilename(
#     title="Select header file",
#     filetypes=[("Text files", "*.txt")]
# )

# pv.global_theme.allow_empty_mesh = True

for file_path in header_files_path[start_index:]:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        continue  # Skip to the next file if the current one does not exist

    print(f"Processing file: {file_path}")
    # Read and parse the header file
    with open(file_path, 'r') as fid:
        # Get base filename without extension and remove last 10 chars
        base_name = os.path.basename(file_path)
        radice_dataset = base_name[:-10]
        path = os.path.dirname(file_path)

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

        # print(f"Time vector: {t0}")

    rr, increment, _ = nearest_multiple(rr)

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

    # Create a save path for the mesh
    plot_save_path = subfolders[2]
    show = True  # Flag to control whether to show the plot or not
    test = True  # Flag to control whether to perform cubic and linear interpolation
    #==============================================================================
    # 3. Load the .ucd files and extract node coordinates and triangle connectivity
    #==============================================================================

    # Initialize arrays with zeros
    v0 = np.zeros((N_vertices, 3, frames_0)) # 3D matrix to store coordinates of each node at each time point
    f0 = np.zeros((N_faces, 3)) # 2D matrix to store connectivity of each triangle (not time dependent)

    print("Start loading files")

    # First loop: from start_frame to fasi
    for i in range(frames_0):
        # Generate filename
        data = f"{radice_dataset}{i:02d}.ucd"
        full_path = os.path.join(path, data)
        # print(f"{full_path=}")

        # Load node coordinates
        temp_data = np.loadtxt(full_path, skiprows=1, delimiter=' ', usecols=(1,2,3), max_rows=N_vertices)
        v0[:, :, i] = temp_data[:N_vertices]

        # Load face connectivity only for first iteration
        if i == 0:
            temp_faces = np.loadtxt(full_path, skiprows=N_vertices + 1, usecols=(3,4,5), max_rows=N_faces)
            f0[:, :] = temp_faces[:N_faces]  # Convert to 1-based indexing
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

    timeplot(v0,f0) # plot della mesh rada nel tempo \decommentare
    normalplot(v0, f0, 1, os.path.join(radice_dataset[:-1], "plots", "normalplot_coarse.obj"), show) # plot delle normali alla mesh rada \decommentare

    vol = volume(v0, f0)/1000  # Calculate the volume of the coarse mesh in mL
    if test:
        plt.figure()
        plt.plot(t0, vol, 'o-', label='volOriginal')
        plt.xticks(t0.squeeze())
        plt.yticks(vol)
        plt.xlabel('Time [ms]')
        plt.ylabel('Volume [mL]')
        plt.title('Volume Over Time')
        plt.grid(True)
        plt.legend()
        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path, "volume_coarse.png"))
            print(f'Mesh saved to {plot_save_path}')
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()  # Close the plotter if not showing

    print(f"End Systole and End Diastole times: {es}, {ed}")

    # Finds the indices of end-systolic and end-diastolic frames
    idx_es = np.argmin(vol)
    idx_ed = np.argmax(vol)

    # Check if the end-systolic and end-diastolic times are aligned with the time vector
    is_aligned = (t0[idx_es] == es) and (t0[idx_ed] == ed)

    print(f"These values are aligned with the time vector: {is_aligned}")
    print(f"ESV, EDV: {vol[idx_es]:.2f} ml, {vol[idx_ed]:.2f} mL")
    print(f"SV, EF: {(vol[idx_ed] - vol[idx_es]):.2f} mL, {(vol[idx_ed] - vol[idx_es]) / vol[idx_ed] * 100:.2f} %")

    #============================================================
    # 4. Reorder and extend the time vector and the displacements
    #============================================================

    # rr_true = t0[-1]
    [t1, v1] = riordina(t0, v0, ed, rr) # t1 is the reordered time vector, v1 is the reordered displacements matrix

    if test:
        vol = volume(v1, f0)/1000  # Calculate the volume of the coarse mesh in mL
        plt.figure()
        plt.plot(t1, vol, 'o-', label='volOriginal')
        plt.xticks(t1.squeeze())
        plt.yticks(vol)
        plt.xlabel('Time [ms]')
        plt.ylabel('Volume [mL]')
        plt.title('Volume Over Time')
        plt.grid(True)
        plt.legend()
        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path, "volume_reordered.png"))
            print(f'Mesh saved to {plot_save_path}')
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()  # Close the plotter if not showing

    new_ed = t1[0]  # Update end-diastolic time to the last time point of the reordered vector
    new_es = new_ed - (ed - es)  # Update end-systolic time to the last time point plus RR interval

    # Extend the displacements to increase the number of cycles
    num_sequences = 3 # Number of sequences wanted
    nt = len(t1) - 1  # Elimino l'ultimo dato, che deve essere inserito solo in coda alla sequenza

    # Duplica la matrice iniziale
    v2 = np.tile(v1[:, :, :nt], (1, 1, num_sequences + 1))  # Crea un ciclo di troppo inizialmente
    v2 = v2[:, :, :-nt + 1]  # Elimina i dati dopo il duplicato del primo in coda alla sequenza

    #print('v2.shape =', v2.shape)
    # print(v2[:,:,75])

    frames_2 = v2.shape[2]

    # Costruzione del vettore tempo t2
    t2 = np.zeros(num_sequences * nt + 1)

    for i in range(1, num_sequences + 1):
        t2[(i - 1) * nt:i * nt] = t1[:nt] + (i - 1) * rr
    t2[-1] = rr * num_sequences

    if test:
        vol = volume(v2, f0)/1000  # Calculate the volume of the coarse mesh in mL
        plt.figure()
        plt.plot(t2, vol, 'o-', label='volOriginal')
        # plt.xticks(t2.squeeze())
        # plt.yticks(vol)
        plt.xlabel('Time [ms]')
        plt.ylabel('Volume [mL]')
        plt.title('Volume Over Time')
        plt.grid(True)
        plt.legend()
        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path, "volume_extended.png"))
            print(f'Mesh saved to {plot_save_path}')
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()  # Close the plotter if not showing

    #=================================
    # 5. Mesh evaluation and remeshing
    #=================================

    mesh = np_to_pv(v2[:,:,0],f0) # Load the mesh from the numpy arrays
    # mesh.save("coarse_mesh.vtp")  # Save the coarse mesh for reference
    avg,_,_ = mesh_quality(mesh)

    inflow = np.load("inflow.npy", allow_pickle=True)
    outflow = np.load("outflow.npy", allow_pickle=True)

    inlet_mesh = mesh.extract_cells(inflow).extract_surface()
    outlet_mesh = mesh.extract_cells(outflow).extract_surface()

    boundary_cells = np.concatenate([inflow, outflow])

    all_cells = np.arange(mesh.n_cells)
    wall_cells = np.setdiff1d(all_cells, boundary_cells)

    wall_mesh = mesh.extract_cells(wall_cells)

    p = pv.Plotter()
    p.add_mesh(wall_mesh, 'lightgrey', lighting=True, show_edges=True)
    p.show()


    if avg['aspect_ratio'] > 1.5 or avg['radius_ratio'] > 1.5:
        print("The mesh quality is poor.")
        print("Remeshing should be done.")
        flag = True
    else:
        print("The mesh quality is good.")
        print("No need to remesh.")
        flag = False


    if flag:
        # edges = mesh.extract_all_edges().compute_cell_sizes()
        # avg_edge = edges['Length'].mean()
        # hausd = avg_edge  # Use this instead of a hardcoded 0.3
        remesh = mmg_remesh(mesh, hausd=0.1, hmax=2.5, hmin=2 ,verbose=True) # Remesh the mesh using MMG

        ### TEST ###
        # inlet_remesh = mmg_remesh(inlet_mesh, hausd=0.1, verbose=True) # Remesh the inlet mesh using MMG
        # outlet_remesh = mmg_remesh(outlet_mesh, hausd=0.1, verbose=True)
        # wall_remesh = mmg_remesh(wall_mesh, hausd=0.1, ar=10, max_aspect_ratio=3, verbose=True) # Remesh the outlet mesh using MMG
        # remesh = inlet_remesh.merge([outlet_remesh, wall_remesh], merge_points=True, tolerance=1e-6)
        #
        avg, _, _ = mesh_quality(remesh)

        if avg['aspect_ratio'] > 1.5 or avg['radius_ratio'] > 1.5:
            print("The mesh quality is poor.")
            print("Remeshing should be done.")

        else:
            print("The mesh quality is good.")
            print("No need to remesh.")

        remesh = remesh.triangulate() # Ensure the remeshed mesh is triangulated
        vert, fac = pv_to_np(remesh)  # Convert remeshed mesh to numpy arrays
        vert = vert[:, :, np.newaxis]
        normalplot(vert, fac, 0, os.path.join(radice_dataset[:-1], "plots", "normalplot_fine.obj"), show)  # Plot the normals of the remeshed mesh
    else:
        remesh = mesh.triangulate() # Ensure the mesh is triangulated


    ### TEST ###
    # p = pv.Plotter()
    # # p.add_mesh(remesh, 'lightgrey', lighting=True, show_edges=True)
    # p.add_mesh(inlet_remesh, 'blue', lighting=True, show_edges=True, label='Inlet', opacity=0.5)
    # p.add_mesh(outlet_remesh, 'red', lighting=True, show_edges=True, label='Outlet', opacity=0.5)
    # p.add_mesh(wall_remesh, 'green', lighting=True, show_edges=True, label='Wall', opacity=0.5)
    # p.show()

    # Compute the average edge length of the remeshed surface
    edges = remesh.extract_all_edges()
    length = edges.compute_cell_sizes(length=True).cell_data['Length']
    l = np.mean(length)

    remesh = remesh.clean()
    if not remesh.is_manifold:
        print("Warning: remesh is non-manifold")

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

    grid.plot(show_edges=True)

    # get cell centroids
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < (np.mean(cell_center[:,2]))
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
    # plotter.show()
    plotter.close() # plotter.show() # Show the plotter window

    # Save as VTU
    grid.point_data["GlobalNodeID"] = np.arange(grid.points.shape[0]) + 1
    grid.cell_data["GlobalElementID"] = np.arange(grid.n_cells) + 1
    grid.cell_data["ModelRegionID"] = np.ones(grid.n_cells, dtype=int)  # Set to one for all cells
    grid.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-complete.mesh.vtu"))

    surface = grid.extract_geometry()

    # Verify if the surface mesh has been modified
    print(f"Surface mesh has been modified: {not np.array_equal(surface.points, remesh.points)}")

    inflow = np.load("inflow.npy", allow_pickle=True)
    outflow = np.load("outflow.npy", allow_pickle=True)

    inlet_mesh = mesh.extract_cells(inflow).extract_surface()
    outlet_mesh = mesh.extract_cells(outflow).extract_surface()

    inlet = extract_surface_region_from_old(surface, inlet_mesh, distance_threshold=0.2)
    outlet = extract_surface_region_from_old(surface, outlet_mesh, distance_threshold=0.2)
    io = inlet + outlet
    surface_with_dist = surface.compute_implicit_distance(io)
    dist = np.abs(surface_with_dist.point_data["implicit_distance"])
    wall_pts = dist > 0.2
    wall = surface.extract_points(wall_pts, adjacent_cells=True)
    wall = wall.connectivity().threshold(0).extract_surface() # Remove inlet and outlet points from the surface mesh

    p = pv.Plotter()
    # p.add_mesh(inlet, color='blue', show_edges=True, label='Inlet', opacity=0.5)
    # p.add_mesh(outlet, color='red', show_edges=True, label='Outlet', opacity=0.5)
    p.add_mesh(wall, color='green', show_edges=True, label='Wall', opacity=0.5)
    # p.add_mesh(surface, color='lightgrey', show_edges=True, label='Surface', opacity=0.5)
    p.add_legend()
    if plot_save_path is not None:
        p.save_graphic(os.path.join(plot_save_path, "boundary.svg"))
        print(f'Mesh saved to {plot_save_path}')
    if show:
        p.show()
    else:
        p.close()

    # inlet, outlet, wall, surface = get_bounds(surface)

    # Save the inlet, outlet, and wall surfaces
    inlet.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-surfaces", "inlet.vtp"))
    outlet.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-surfaces", "outlet.vtp"))
    wall.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-surfaces", "wall.vtp"))
    surface.save(os.path.join(radice_dataset[:-1], "mesh", "mesh-complete.exterior.vtp"))
    #==================================================================================
    # 7. Remapping the coarse meshes of the time interpolated data on the remeshed mesh
    #==================================================================================

    v3_0, f1 = pv_to_np(surface)  # Convert remeshed mesh to numpy arrays

    # Initialize v4 with the appropriate dimensions
    v3 = np.zeros((v3_0.shape[0], v3_0.shape[1], v2.shape[2]))

    tr = f0
    p = v3_0

    # Loop through each time step
    for i in tqdm(range(v2.shape[2] - 1)):
        vr = v2[:, :, i]
        ur = v2[:, :, i + 1] - v2[:, :, i]

        # Assuming resample_u is already defined, and returns pp and upp
        pp, upp = resample_u(vr, tr, ur, p)

        # Store the results in v4
        v3[:, :, i] = pp
        p = upp + pp
        v3[:, :, i + 1] = p

    # timeplot(v3,f1)

    vol = volume(v3, f1)/1000  # Calculate the volume of the coarse mesh in mL
    if test:
        plt.figure()
        plt.plot(t2, vol, 'o-', label='volOriginal')
        # plt.xticks(t2.squeeze())
        # plt.yticks(vol)
        plt.xlabel('Time [ms]')
        plt.ylabel('Volume [mL]')
        plt.title('Volume Over Time')
        plt.grid(True)
        plt.legend()
        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path, "volume_interpolated.png"))
            print(f'Mesh saved to {plot_save_path}')
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()  # Close the plotter if not showing

    # Finds the indices of end-systolic and end-diastolic frames
    idx_es = np.argmin(vol[:len(vol)//3]) # Only consider the first third of the volume for end-systolic
    idx_ed = np.argmax(vol[:len(vol)//3]) # Only consider the first third of the volume for end-diastolic

    # Check if the end-systolic and end-diastolic times are aligned with the time vector
    is_aligned = np.isclose(t2[idx_es],new_es) and np.isclose(t2[idx_ed], new_ed)
    print(f"These values are aligned with the time vector: {is_aligned}")
    print(f"ESV, EDV: {vol[idx_es]:.2f} ml, {vol[idx_ed]:.2f} mL")
    print(f"SV, EF: {(vol[idx_ed] - vol[idx_es]):.2f} mL, {(vol[idx_ed] - vol[idx_es]) / vol[idx_ed] * 100:.2f} %")
    #===========================================================
    # 8. Interpolation of the data to create intermediate frames
    #===========================================================

    # num_intermedie = 4  # Number of intermediate frames to insert between each original frame
    dt = 1 # Time step size for the new frames [ms]
    t3 = np.arange(t2[0], t2[-1] + dt, dt)  # New time vector with the desired time step
    if t3[-1] > t2[-1] and not np.isclose(t3[-1], t2[-1]):
        t3 = t3[:-1]  # Remove the last time point if it exceeds the original end time
    # t3[-1] = t2[-1]  # Ensure the last time point matches the original end time
    frames_3 = len(t3)  # Total number of frames in the new time vector
    # frames_3  = frames_2 + (frames_2- 1) * num_intermedie

    # Initialize v4 with the appropriate dimensions
    v4 = np.zeros((v3.shape[0], v3.shape[1], frames_3))
    v_cubic = np.zeros(v4.shape)  # For cubic interpolation
    v_linear = np.zeros(v4.shape)  # For linear interpolation
    v_pchip = np.zeros(v4.shape)  # For PCHIP interpolation

    # Create a time vector for the new frames
    # t3 = np.linspace(t2[0], t2[-1], frames_3)


    if test:
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
    for i in tqdm(range(v3.shape[0])):
        for dim in range(3):
            coordinates = v3[i, dim, :]
            pchip_interp = PchipInterpolator(t2, coordinates)
            v_pchip[i, dim, :] = pchip_interp(t3)

    # Fourier interpolation
    v_fourier = fourier(v_pchip, t3)
    V = volume(v_fourier, f1)  # Calculate the volume for the Fourier interpolated data
    # Create the plot for the original and interpolated volumes
    plt.figure()
    plt.plot(t2,volume(v3,f1)/1000, '*', label='volOriginal')
    if test:
        plt.plot(t3,volume(v_cubic, f1)/1000, label='volCubic')
        plt.plot(t3,volume(v_linear, f1)/1000, label='volLinear')
    plt.plot(t3,volume(v_pchip, f1)/1000, label='volPchip')
    plt.plot(t3,V/1000, label='volFourier')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time Frame')
    plt.ylabel('Volume')
    plt.title('Volume Over Time')
    if plot_save_path is not None:
        plt.savefig(os.path.join(plot_save_path, "volume_interpolation.png"))
        print(f'Mesh saved to {plot_save_path}')
    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close()  # Close the plotter if not showing

    v4 = np.ndarray.view(v_fourier)  # Use the Fourier interpolated data for further processing
    # normalplot(v4, f1, 1)  # Plot the normals of the interpolated mesh

    vol = V/1000 # Convert volume to mL for the Fourier interpolated data
    if test:
        plt.figure()
        plt.plot(t3, vol, 'o-', label='volOriginal')
        # plt.xticks(t2.squeeze())
        # plt.yticks(vol)
        plt.xlabel('Time [ms]')
        plt.ylabel('Volume [mL]')
        plt.title('Volume Over Time')
        plt.grid(True)
        plt.legend()
        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path, "volume_interpolated_final.png"))
            print(f'Mesh saved to {plot_save_path}')
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()  # Close the plotter if not showing

    # Finds the indices of end-systolic and end-diastolic frames
    idx_es = np.argmin(vol[:len(vol)//3]) # Only consider the first third of the volume for end-systolic
    idx_ed = np.argmax(vol[:len(vol)//3]) # Only consider the first third of the volume for end-diastolic

    # Check if the end-systolic and end-diastolic times are aligned with the time vector
    is_aligned = np.isclose(t3[idx_es],new_es) and np.isclose(t3[idx_ed], new_ed)
    if not is_aligned:
        new_ed = t3[idx_ed]
        new_es = t3[idx_es]
    is_aligned = np.isclose(t3[idx_es],new_es) and np.isclose(t3[idx_ed], new_ed)
    print(f"These values are aligned with the time vector: {is_aligned}")
    print(f"ESV, EDV: {vol[idx_es]:.2f} ml, {vol[idx_ed]:.2f} mL")
    print(f"SV, EF: {(vol[idx_ed] - vol[idx_es]):.2f} mL, {(vol[idx_ed] - vol[idx_es]) / vol[idx_ed] * 100:.2f} %")

    #==========================================================
    # 9. Generate the input files for the simulation
    #==========================================================

    # a. Generate the .dat file for the displacements
    displacement = v4 - v4[:,:,0][:, :, np.newaxis] # Displacements from the initial frame for the whole mesh

    write_motion(displacement, t3, wall, surface, os.path.join(radice_dataset[:-1], "wall" ))
    write_motion(displacement, t3, inlet, surface, os.path.join(radice_dataset[:-1], "inlet"))
    write_motion(displacement, t3, outlet, surface, os.path.join(radice_dataset[:-1], "outlet"))

    # b. Generate the .flow file for the inlet
    from funzioni import flow
    Q_in, Q_out = flow(V, t3, plot_save_path, show)

    # Save to inlet.flow file
    with open(os.path.join(radice_dataset[:-1], "inlet.flow"), "w") as f:
        f.write(f"{len(t3.flatten())} {inlet.n_points}\n")  # Write number of time steps and nodes
        for t, q in zip(t3.flatten()/1000, -Q_in):
            # Convert q to a scalar and write to file
            f.write(f"{t:.7f} {q:.7f}\n")

    # Save to outlet.flow file
    with open(os.path.join(radice_dataset[:-1], "outlet.flow"), "w") as f:
        f.write(f"{len(t3.flatten())} {outlet.n_points}\n")  # Write number of time steps and nodes
        for t, q in zip(t3.flatten()/1000, -Q_out):
            # Convert q to a scalar and write to file
            f.write(f"{t:.7f} {q:.7f}\n")

    # c. Generate the .bct file for the inlet
    # === Load the inlet mesh ===
    # inlet = pv.read(os.path.join(radice_dataset[:-1], "mesh", "mesh-surfaces", "inlet.vtp"))
    points = inlet.points
    node_ids = inlet.point_data["GlobalNodeID"]
    normals = np.array(inlet.point_normals)

    t3_s = t3.flatten()/1000  # Convert to seconds
    nl = len(t3_s)

    # === Calculate the area of the inlet surface ===
    mesh = pv.PolyData(points)
    tri = mesh.delaunay_2d()
    area = tri.area

    # === Centroide of the inlet surface ===
    centroid = points.mean(axis=0)

    # === Max radius from the centroid ===
    radii = np.linalg.norm(points[:, :2] - centroid[:2], axis=1)
    r_max = radii.max()


    # === List to store velocity matrices ===
    # velocità = []  # contiene velocity_0.0000, velocity_0.0001, ...

    # === Cycle through each time step ===
    for j in range(nl):
        Q = Q_in[j]
        v_mean = Q / area
        t = t3_s[j]

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

    # === Saving in a .vtp file ===
    inlet.save(os.path.join(radice_dataset[:-1], "bct.vtp"))

    # d. Generate the .inp file for the simulation
    distal_pressure = 8 * 133.322  # Distal pressure in Pa
    initial_pressure = 12 * 133.322   # Initial pressure in Pa

    config_text = f"""\
    #----------------------------------------------------------------
    # General simulation parameters
    
    Continue previous simulation: 0
    Number of spatial dimensions: 3
    Number of time steps: {frames_3}
    Time step size: {dt/1000}
    Spectral radius of infinite time step: 0.50
    Searched file name to trigger stop: STOP_SIM
    
    Save results in folder: simulation
    Save results to VTK format: 1
    Name prefix of saved VTK files: result
    Increment in saving VTK files: {increment//dt}
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
       Initial pressures file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/pressure/pressure_100.vtu
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
    
       Initialize RCR from flow: t
       
       # Inlet - from the inlet.flow file
       Add BC: lumen_inlet {{
          Type: Dir
          Time dependence: Unsteady
          Temporal values file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/inlet.flow
          Profile: Parabolic
          Impose flux: 1
       }}
       
       # Inlet - from the bct.vtp file
       #Add BC: lumen_inlet {{
       #   Type: Dir
       #   Time dependence: General
       #   BCT file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/bct.vtp
       #}}
       
       Add BC: lumen_outlet {{
          Type: Neu
          Time dependence: RCR
          RCR values: "0.0044, 123.75, 0.0293"
          Distal pressure: {distal_pressure}
          Initial pressure: {initial_pressure}
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

    # e. Generete the pressure.inp file for the initial pressures
    pressure_text = f"""\
    #----------------------------------------------------------------
    # General simulation parameters
    
    Continue previous simulation: 0
    Number of spatial dimensions: 3
    Number of time steps: 100
    Time step size: 0.05
    Spectral radius of infinite time step: 0.50
    Searched file name to trigger stop: STOP_SIM
    
    Save results in folder: pressure
    Save results to VTK format: 1
    Name prefix of saved VTK files: pressure
    Increment in saving VTK files: 100
    Start saving after time step: 1
    Increment in saving restart files: 200
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
    
    Add equation: fluid {{
       Coupled: 1
       Min iterations: 3
       Max iterations: 10
       Tolerance: 1e-3
       Backflow stabilization coefficient: 0.2
    
       Density: 0.00106
    
       Viscosity: Constant {{Value: 0.004}}
       
    #   Carreau-Yasuda {{
    #      Limiting high shear-rate viscosity: 0.022
    #      Limiting low shear-rate viscosity: 0.22
    #      Shear-rate tensor multiplier (lamda): 0.11
    #      Shear-rate tensor exponent (a): 0.644
    #      Power-law index (n): 0.392
    #   }}
    
       Output: Spatial {{
          Velocity: t
          Pressure: t
          Traction: t
          WSS: t
       }}
    
       LS type: GMRES {{
          Preconditioner: FSILS
          Max iterations: 100
          Tolerance: 1e-4
          Absolute tolerance: 1e-4
          Krylov space dimension: 100
       }}   
    
       Add BC: lumen_inlet {{
          Type: Dir
          Time dependence: Unsteady
          Temporal values file path: /global-scratch/bulk_pool/afernando/Docker/{radice_dataset[:-1]}/inlet.flow
          Profile: Parabolic
          Impose flux: t
       }}
    
       Add BC: lumen_outlet {{
          Type: Neu
          Value: {15*133.322:.2f} 
       }}
    
       Add BC: lumen_wall {{
          Type: Dir
          Time dependence: Steady
          Value: 0.0
       }}
    }}
    """

    # Write the pressure configuration to a file
    with open(os.path.join(radice_dataset[:-1], "pressure.inp"), "w") as file:
        file.write(pressure_text)

    end_time = time.time()  # ⏱️

    print(f"Script completed successfully in {end_time - start_time:.2f} seconds.")

    plt.close('all')  # Close all matplotlib plots

print("Done")