import numpy as np

def riordina(t, v, t0, T):
    """
    Reorders temporal and spatial data for a cyclic phenomenon.

    Parameters:
        t (array): Time instants vector
        v (ndarray): 3D matrix of point coordinates (num_points x 3 x num_instants)
        t0 (float): New initial time instant
        T (float): Phenomenon period

    Returns:
        tnew, vnew: Reordered time vector and coordinate matrix
    """
    t = t.reshape(-1)  # Reshape to 1D at start of function
    lt = len(t)
    nv, nc, nt = v.shape

    # Validate input dimensions
    if lt != nt:
        raise ValueError("Length of t must match third dimension of v")
    
    # Initialize output arrays
    tnew = np.zeros(lt + 1)  # +1 to include T at end
    vnew = np.zeros((nv, nc, nt + 1))  # +1 for periodic data

    # Find index of t0
    a = np.where(t==t0)[0][0]  

    # print("Shapes:")
    # print(f"t shape: {t.shape}")
    # print(f"tnew shape: {tnew.shape}")
    # print(f"t[a:] shape: {t[a:].shape}")
    # print(a)

    # Reorder time vector
    tnew[:lt - a] = t[a:] - t[a]  # First part
    if a > 0:
        tnew[lt - a:lt] = t[:a] - t[a] + T  # Second part
    tnew[-1] = T  # Add period at end

    n = len(tnew)
    tnew = tnew.reshape(n)

    # Reorder spatial data
    vnew[:, :, :lt - a] = v[:, :, a:]  # First part
    if a > 0:
        vnew[:, :, lt - a:lt] = v[:, :, :a]  # Second part
    vnew[:, :, -1] = vnew[:, :, 0]  # Add periodic point

    return tnew, vnew



import trimesh

def volume(matrix1, TM1):
    """
    Calcola il volume racchiuso dalla superficie triangolare per ogni frame.

    Parameters:
        matrix1 (ndarray): Matrice 3D delle coordinate dei nodi (num_nodi x 3 x num_frame).
        TM1 (ndarray): Matrice di connettività dei triangoli (num_triangoli x 3).

    Returns:
        vol (ndarray): Vettore dei volumi calcolati per ciascun frame.
    """
    # print("Calcolando il volume per ciascun frame usando trimesh...")
    vol = np.zeros(matrix1.shape[2])  # Preallocazione del vettore vol

    for j in range(matrix1.shape[2]):
        # Estrai i vertici del frame corrente
        vertices = matrix1[:, :, j]
        
        # Crea la mesh e calcola il volume
        mesh = trimesh.Trimesh(vertices=vertices, faces=TM1)
        vol[j] = mesh.volume  # Salva il volume per il frame corrente

    # print("Fatto.")
    return vol


def fourier(matrix2, time, num_intermedie):
    # print('Increasing number of frames with Fourier interpolation.')

    # Numero totale di frame interpolati
    num_frames_original = matrix2.shape[2]
    num_frames_interpolati = num_frames_original + (num_frames_original - 1) * num_intermedie

    # Pre-allocazione per la nuova matrice
    new_matrix = np.zeros((matrix2.shape[0], matrix2.shape[1], num_frames_interpolati))
    new_time = np.linspace(time[0], time[-1], num_frames_interpolati)  # Nuova timeline uniforme
    nt = (len(new_time) - matrix2.shape[2]) // 2
    dt = new_time[1] - new_time[0]

    # Ciclo per interpolare armonicamente ogni punto della mesh
    for i in range(matrix2.shape[0]):  # Per ogni nodo
        for j in range(matrix2.shape[1]):  # Per ogni direzione (x, y, z)
            # Estrazione del segnale originale
            signal = matrix2[i, j, :]

            # Trasformata di Fourier del segnale
            F = np.fft.fft(signal)
            F = np.fft.fftshift(F)
            F = np.concatenate((np.zeros(nt), F, np.zeros(nt)))  # Aggiungi zeri per l'interpolazione
            F = np.fft.ifftshift(F)
            k = np.fft.ifft(F) * len(F) / len(signal)
            new_matrix[i, j, :] = np.real(k)

            # Frequenze corrispondenti per la derivata
            freqs = np.fft.fftfreq(len(F), d=dt)

    fasi = num_frames_interpolati
    # print('Done.')

    return new_matrix


from scipy.spatial import distance

def resample_u(vr, tr, ur, p):
    """
    Proietta i punti su una superficie triangolata e calcola lo spostamento tramite interpolazione.
    
    Parametri:
    - vr: nodi della superficie triangolata (Nx3)
    - tr: triangoli della superficie triangolata (Mx3)
    - ur: spostamenti dei nodi della superficie triangolata (Nx3)
    - p: punti da proiettare sulla superficie triangolata (Kx3)
    
    Ritorna:
    - pp: proiezioni dei punti sulla superficie triangolata (Kx3)
    - upp: spostamenti interpolati dei punti proiettati (Kx3)
    """
    
    # Triangolazione come oggetto
    T_faces = tr
    T_vertices = vr
    
    # Proiezione dei punti p sulla superficie triangolata
    pp, _ = point2trimesh(T_faces, T_vertices, p)

    # Calcolo dei baricentri
    v1 = vr[tr[:, 0], :]
    v2 = vr[tr[:, 1], :]
    v3 = vr[tr[:, 2], :]
    g = (v1 + v2 + v3) / 3  # Baricentro di ogni triangolo
    
    # Trova l'indice del triangolo "di appartenenza"
    k = distance.cdist(pp, g).argmin(axis=1)

    # Trasformazioni per ogni triangolo
    nt = tr.shape[0]
    vtrans = np.zeros((3 * nt, 3))
    R = np.zeros((3 * nt, 3))
    T = np.zeros((3 * nt, 1))
    
    for i in range(nt):
        vtrans[i*3:(i+1)*3, :], R[i*3:(i+1)*3, :], T[i*3:(i+1)*3, 0] = transform_t(v1[i, :], v2[i, :], v3[i, :])

    # Interpolazione degli spostamenti
    np_points = pp.shape[0]
    upp = np.zeros((np_points, 3))
    
    for i in range(np_points):
        blocco = ((k[i] - 1) * 3+1, k[i] * 3)
        Rp = R[blocco, :]
        Tp = T[blocco, 0]
        Xp = vtrans[blocco, :]
        ptrans = Rp @ (pp[i, :].T - Tp.flatten())
        rs = np.linalg.solve(
            np.column_stack((Xp[:2, 1] - Xp[:2, 0], Xp[:2, 2] - Xp[:2, 0])),
            ptrans[:2] - Xp[:2, 0]
        )
        u1 = ur[tr[k[i], 0], :]
        u2 = ur[tr[k[i], 1], :]
        u3 = ur[tr[k[i], 2], :]
        upp[i, :] = u1 + (u2 - u1) * rs[0] + (u3 - u1) * rs[1]
    
    return pp, upp

# NOTA: Implementa o importa `point2trimesh` e `transform_t` in Python

def point2trimesh(faces, vertices, points):
    """
    Proietta i punti su una superficie triangolata.
    
    Parametri:
    - faces: matrice di connettività dei triangoli (Mx3)
    - vertices: matrice dei nodi della superficie triangolata (Nx3)
    - points: punti da proiettare sulla superficie triangolata (Kx3)
    
    Ritorna:
    - projected_points: proiezioni dei punti sulla superficie triangolata (Kx3)
    - projected_faces: indici dei triangoli proiettati (Kx1)
    """
    
    # Costruzione della mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Proiezione dei punti
    projected_points, distances, index = trimesh.proximity.closest_point(mesh, points)
    
    return projected_points, index


def transform_t(x1, x2, x3):
    """
    Trasforma un triangolo da coordinate cartesiane a una nuova terna di coordinate.

    Parametri:
    - x1, x2, x3: vettori riga (1x3) che contengono le coordinate cartesiane dei vertici di un triangolo

    Ritorna:
    - xnew: matrice 3x3 con le coordinate trasformate dei vertici
    - R: matrice di rotazione per la trasformazione
    - T: vettore di traslazione per la trasformazione
    """
    
    # Assicurarsi che x1, x2, x3 siano colonne
    x1 = np.array(x1).flatten()  # Assicura che sia un vettore riga
    x2 = np.array(x2).flatten()
    x3 = np.array(x3).flatten()
    
    # Calcolare il vettore normale al piano del triangolo
    c = np.cross(x2 - x1, x3 - x1)
    c = c / np.linalg.norm(c)  # Normalizzazione del vettore normale

    # Primo asse (direzione da x1 a x2)
    a = x2 - x1
    a = a / np.linalg.norm(a)  # Normalizzazione del vettore

    # Secondo asse (perpendicolare a c e a)
    b = np.cross(c, a)

    # Matrice di rotazione (R)
    R = np.vstack((a, b, c)).T  # Trasposizione della matrice di rotazione

    # Traslazione
    T = x1

    # Trasformazione dei vertici
    xnew = R @ np.vstack((x1 - T, x2 - T, x3 - T)).T  # Matrice di trasformazione (R * [x1-T, x2-T, x3-T])

    return xnew, R, T

import sys
import pyvista as pv
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

# Store the text actor in a variable
frame_text = None

def timeplot(v, f):
    print('Animation of mesh')

    # Example setup (replace with actual data)
    T = v.shape[2]  # Number of time steps
    N = v.shape[0]  # Number of nodes
    M = f.shape[0]  # Number of triangles

    # Generate a dummy evolving mesh (replace with real data)
    mesh_matrix = v  # np.random.rand(N, 3, T)  # Shape (N, 3, T)
    connectivity = f  # np.random.randint(0, N, (M, 3))  # Shape (M, 3)

    # Convert connectivity to PyVista format
    faces = np.hstack((np.full((M, 1), 3), connectivity)).astype(np.int32).flatten()

    # Create initial mesh
    mesh = pv.PolyData(mesh_matrix[:, :, 0], faces)

    # Create plotter
    pl = pv.Plotter()

    # Animation parameters
    frame_counter = [0]
    is_playing = [True]
    stop_flag = [False]  # Flag to control stopping of the function

    # Get axis limits from the entire dataset (not per frame)
    x_min, x_max = v[:, 0, :].min(), v[:, 0, :].max()
    y_min, y_max = v[:, 1, :].min(), v[:, 1, :].max()
    z_min, z_max = v[:, 2, :].min(), v[:, 2, :].max()

    # Define a symmetric range for equal axis scaling
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    center = [(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2]

    # Set fixed bounds for all frames
    pl.set_focus(center)
    pl.set_position([center[0] + max_range, center[1] + max_range, center[2] + max_range])
    pl.camera.parallel_projection = True  # Maintain orthogonal projection
    pl.camera.SetParallelScale(max_range)  # Fix zoom level
    pl.view_isometric()

    # Show bounding box with labels
    pl.show_bounds(
        grid=True,
        bounds=[x_min, x_max, y_min, y_max, z_min, z_max],
        location="all",
        xtitle="X-axis",
        ytitle="Y-axis",
        ztitle="Z-axis",
        color="black"

    )

    pl.add_mesh(
        mesh,
        color="white",  # White surface (or any neutral color)
        lighting = False  # Shading effects
    )

    mesh["scalars"] = mesh.points[:, 2]  # Use Z-values for the color gradient

    pl.add_mesh(
        mesh,
        style="wireframe",  # Wireframe only
        scalars="scalars",  # Use Z-values for coloring
        cmap="rainbow",  # Rainbow gradient
        line_width=3,  # Make the wireframe more visible
        show_scalar_bar = False,  # Disable colorbar
        lighting=True,  # Shading effects
        opacity=0.9
    )

    def update_mesh():
        global frame_text
        if frame_text:
            pl.remove_actor(frame_text)
        mesh.points = mesh_matrix[:, :, frame_counter[0]]
        mesh["scalars"] = mesh.points[:, 2]  # Update wireframe color based on Z
        pl.render()

        frame_text = pl.add_text(f"Frame: {frame_counter[0] + 1}", font_size=12, position='lower_left')


    def toggle_play(e=None):
        is_playing[0] = not is_playing[0]
        print("Playing" if is_playing[0] else "Paused")

    def next_frame(e=None):
        is_playing[0] = False
        frame_counter[0] = (frame_counter[0] + 1) % T
        update_mesh()

    def prev_frame(e=None):
        is_playing[0] = False
        frame_counter[0] = (frame_counter[0] - 1) % T
        update_mesh()

    def stop_application(e=None):
        stop_flag[0] = True  # Set flag to stop the animation function
        timer.stop()  # Stop the timer to halt the animation
        #pl.close()  # Close the plotter

    # Add keyboard controls
    pl.add_key_event('space', toggle_play)
    pl.add_key_event('Right', next_frame)
    pl.add_key_event('Left', prev_frame)
    pl.add_key_event('Escape', stop_application)

    # Add instructions
    pl.add_text(
        'Controls:\nSpace: Play/Pause\nLeft/Right: Previous/Next Frame\nEsc: Stop Animation',
        font_size=12,
        position='upper_left'
    )

    pl.add_text(
        'Mesh Plot',
        font_size=16,
        position=(0.5,0.95),
        viewport = True
    )

    # Initialize the render window
    pl.show(interactive=False, auto_close=False)
    pl.iren.initialize()

    def timer_callback():
        if is_playing[0] and not stop_flag[0]:  # Only run if the stop flag is not set
            frame_counter[0] = (frame_counter[0] + 1) % T
            update_mesh()
        elif stop_flag[0]:  # Stop the timer when the flag is set
            timer.stop()

    # Set up the QTimer
    app = QApplication(sys.argv)
    timer = QTimer()
    timer.timeout.connect(timer_callback)
    timer.start(50)  # 250 ms interval

    if pl.ren_win is None:
        return  # Exit the function if the render window is closed

    # While loop to keep checking the stop_flag
    while not stop_flag[0]:
        pl.iren.start()
    print('Animation stopped')
    #app.exec_()  # Start the application event loop
    return   # Return to exit the function when the flag is set

import scipy.io
import trimesh
from scipy.spatial import cKDTree
xnew = None

def transform_t(x1, x2, x3):
    """
    Transforms the coordinates of the vertices of a triangle into a new coordinate system
    defined by the first vertex, the second vertex, and the normal to the triangle's surface.

    Parameters:
    x1, x2, x3: numpy arrays (1D) of shape (3,), representing the coordinates of the triangle vertices.

    Returns:
    xnew: A 3x3 numpy array, where each column contains the transformed coordinates of the vertices.
    R: A 3x3 numpy array, the rotation matrix for the transformation.
    T: A 3x1 numpy array, the translation vector for the transformation.
    """
    # Ensure that x1, x2, x3 are column vectors
    x1 = np.array(x1).flatten()
    x2 = np.array(x2).flatten()
    x3 = np.array(x3).flatten()

    # Compute the normal vector to the triangle's surface (cross product of two edges)
    c = np.cross(x2 - x1, x3 - x1)
    c = c / np.linalg.norm(c)  # Normalize the normal vector

    # The first axis is the vector from x1 to x2, normalized
    a = x2 - x1
    a = a / np.linalg.norm(a)

    # The second axis is the cross product of the normal vector and the first axis
    b = np.cross(c, a)

    # Rotation matrix
    R = np.column_stack([a, b, c]).T

    # Translation vector (first vertex)
    T = x1
    # Transformed coordinates for each vertex (translate to origin and then rotate)
    xnew = R @ (np.column_stack([x1 - T, x2 - T, x3 - T]))  # Transform each vertex

    return xnew, R, T

def resample_u(vr,tr,ur,p):

    # Ensure faces are in the correct format (add '3' at the beginning)
    tr_pyvista = np.hstack([np.full((tr.shape[0], 1), 3), tr])  # Add 3 to the beginning of each row

    # Assuming 'vr' (vertices) and 'tr' (faces) are already defined
    # Create a Trimesh object from the vertices and faces
    mesh = trimesh.Trimesh(vertices=vr, faces=tr)

    # Project the points onto the surface of the mesh
    projected_points = mesh.nearest.on_surface(p)
    pp = projected_points[0]

    # Print or use projected_points
    # print(pp)
    # Create the PyVista mesh
    mesh = pv.PolyData(vr, tr_pyvista)

    # # Create a plotter
    # plotter = pv.Plotter()
    #
    # # Add the mesh to the plotter
    # plotter.add_mesh(mesh, color='lightgray', show_edges=True)
    #
    # # Add the projected points to the plotter
    # plotter.add_points(pp, color='red', point_size=10)
    #
    # # Show the plot
    # plotter.show()

    # Compute the centroids of the triangles
    v1 = vr[tr[:, 0], :]  # First vertex of each triangle
    v2 = vr[tr[:, 1], :]  # Second vertex of each triangle
    v3 = vr[tr[:, 2], :]  # Third vertex of each triangle
    centroids = (v1 + v2 + v3) / 3  # Centroid of each triangle

    # Build a KD-Tree for fast nearest-neighbor search
    tree = cKDTree(centroids)

    # Find the closest centroid for each projected point
    k = tree.query(pp)[1]  # indices of the closest centroids

    npp = len(pp) # Number of vertices
    nt = len(tr) # Number of triangles

    # Initialize matrices for transformed coordinates, rotation matrix, and translation vector
    vtrans = np.zeros((3 * nt, 3))
    R = np.zeros((3 * nt, 3))
    T = np.zeros((3 * nt, ))

    for i in range(nt):

        # For each triangle, get the transformation and store them in the corresponding matrices
        xnew,r,t = transform_t(v1[i, :], v2[i, :], v3[i, :])

        vtrans[3 * i:3 * (i + 1), :] = xnew  # Now assign
        R[3 * i:3 * (i + 1), :] = r
        T[3 * i:3 * (i + 1), ] = t
    # Interpolation of displacements for the projected points
    upp = np.zeros((npp, 3))

    for i in range(npp):
        # For each projected point, calculate the displacements based on the closest triangle
        block = slice((k[i] * 3), (k[i] + 1) * 3)
        Rp = R[block, :]
        Tp = T[block, ]
        Xp = vtrans[block, :]

        # Perform transformation of the point
        ptrans = np.dot(Rp, (pp[i] - Tp).T)

        # Calculate the barycentric coordinates (rs)
        rs = np.linalg.solve(Xp[:2, 1:] - Xp[:2, 0], ptrans[:2] - Xp[:2, 0])

        # Get the nodal displacements
        u1 = ur[tr[k[i], 0], :]
        u2 = ur[tr[k[i], 1], :]
        u3 = ur[tr[k[i], 2], :]

        # Interpolate the displacements
        upp[i, :] = u1 + (u2 - u1) * rs[0] + (u3 - u1) * rs[1]
    return pp, upp


def normalplot(v,f,i, save_path=None, show=True):
    """  Plots the normal vectors to a mesh at a specific frame. """
    print('Plotting the normal vectors to the mesh')
    # Create a PyVista mesh
    mesh = pv.PolyData(v[:, :, i], np.hstack([np.full((f.shape[0], 1), 3), f]))

    # Compute the normals
    mesh.compute_normals(cell_normals=True, point_normals=True, inplace=True)
    centers = mesh.cell_centers().points

    # Calculate appropriate arrow size based on mesh dimensions
    bounds = mesh.bounds
    diagonal = np.sqrt(np.sum((np.array(bounds[1::2]) - np.array(bounds[::2])) ** 2))
    arrow_length = diagonal * 0.03  # 10% of the mesh diagonal

    # Plot the mesh with normals
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color='lightgray')
    plotter.add_arrows(centers, mesh.cell_normals, mag=arrow_length, color='red', opacity=0.3)
    plotter.camera_position = 'iso'  # Set isometric view
    plotter.add_text(f'Normal vectors of the {i}-th frame', font_size=14, shadow=True, position='upper_edge')

    # Save the 3D mesh if a save path is provided
    if save_path is not None:
        plotter.export_obj(save_path)
        print(f'Mesh saved to {save_path}')

    # Show the plot if requested
    if show:
        plotter.show()
    else:
        plotter.close()  # Close the plotter if not showing


import os
import subprocess
import meshio
import numpy as np
import pyvista as pv
import meshio

def mmg_remesh(input_mesh, hausd=0.3, hmax=2, hmin=1.5, ar=20, max_aspect_ratio=None, max_iter=3, verbose=False):
    #crea un idetificatore univoco per la mesh
    mesh_id = os.getpid()
    input_mesh.clear_data()

    #salva la mesh in formato mesh
    pv.save_meshio(f'{mesh_id}.mesh', input_mesh)

    #esegue il comando mmg con i paramteri specificati per eseguire il remeshing
    subprocess.run(['C:/Program Files/mmg/mmgs_O3.exe', #'D:/mmg/bin/mmgs_O3.exe',
                    f'{mesh_id}.mesh',
                    '-hausd', str(hausd),
                    '-hmax', str(hmax),
                    '-hmin', str(hmin),
                    '-ar', str(ar),
                    # '-nr',
                    # '-nreg',
                    # '-xreg',
                    # '-optim',
                    f'{mesh_id}_remeshed.mesh'], stdout=subprocess.DEVNULL)
    # Legge la mesh remeshata
    new_mesh = meshio.read(f'{mesh_id}_remeshed.mesh')
    pvmesh = pv.utilities.from_meshio(new_mesh)

    # Controlla qualità del remeshing
    if max_aspect_ratio is not None:
        qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')
        it = 0
        while np.max(qual['CellQuality']) > max_aspect_ratio and it < max_iter:
            it += 1
            subprocess.run(['C:/Program Files/mmg/mmgs_O3.exe',  # 'D:/mmg/bin/mmgs_O3.exe',
                            f'{mesh_id}.mesh',
                            '-hausd', str(hausd * 2),
                            '-hmax', str(hmax),
                            '-hmin', str(hmin),
                            '-ar', str(ar),
                            # '-nr',
                            # '-nreg',
                            # '-xreg',
                            # '-optim',
                            f'{mesh_id}_remeshed.mesh'], stdout=subprocess.DEVNULL)
        new_mesh = meshio.read(f'{mesh_id}_remeshed.mesh')
        pvmesh = pv.utilities.from_meshio(new_mesh)
        qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')
        if verbose:
            print('Max aspect ratio:', np.max(qual['CellQuality']))

    # Pulisce i  file temporanei
    os.remove(f'{mesh_id}.mesh')
    os.remove(f'{mesh_id}_remeshed.mesh')
    os.remove(f'{mesh_id}_remeshed.sol')
    return pvmesh.extract_surface()
'''
import os
import subprocess
import numpy as np
import pyvista as pv
import meshio


def mmg_remesh(
    input_mesh,
    hausd=None,
    hmax=2.0,
    hmin=1.5,
    max_aspect_ratio=None,
    max_iter=3,
    use_optim=False,
    preserve_features=True,
    verbose=False
):
    """
    Remeshes a surface mesh using MMG with optional geometric preservation.

    Parameters:
        input_mesh (pyvista.PolyData): The surface mesh to remesh.
        hausd (float): Hausdorff distance; smaller values increase smoothing.
                       If None, it is estimated from average edge length.
        hmax (float): Maximum element size.
        hmin (float): Minimum element size.
        max_aspect_ratio (float): Max allowed aspect ratio for remeshed mesh.
        max_iter (int): Max iterations to improve aspect ratio.
        use_optim (bool): Whether to enable MMG surface optimization.
        preserve_features (bool): Whether to preserve sharp ridges and features.
        verbose (bool): Print diagnostics if True.

    Returns:
        pyvista.PolyData: The remeshed surface mesh.
    """
    mesh_id = os.getpid()
    input_mesh.clear_data()

    # Estimate hausd from average edge length if not provided
    if hausd is None:
        edges = input_mesh.extract_all_edges().compute_cell_sizes()
        hausd = edges['Length'].mean()
        if verbose:
            print(f"Estimated hausd = {hausd:.4f}")

    # Save input mesh as .mesh format (Medit format for MMG)
    pv.save_meshio(f'{mesh_id}.mesh', input_mesh)

    # Construct MMG command
    args = [
        'C:/Program Files/mmg/mmgs_O3.exe',
        f'{mesh_id}.mesh',
        '-hausd', str(hausd),
        '-hmax', str(hmax),
        '-hmin', str(hmin)
    ]

    if preserve_features:
        args += ['-ridge', '-detect']  # enable edge/feature detection

    if use_optim:
        args.append('-optim')  # optional smoothing

    args.append(f'{mesh_id}_remeshed.mesh')

    # Run MMG
    subprocess.run(args, stdout=subprocess.DEVNULL)

    # Read remeshed mesh
    new_mesh = meshio.read(f'{mesh_id}_remeshed.mesh')
    pvmesh = pv.utilities.from_meshio(new_mesh)

    # Check and improve quality
    if max_aspect_ratio is not None:
        qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')
        it = 0
        while np.max(qual['CellQuality']) > max_aspect_ratio and it < max_iter:
            it += 1
            hausd *= 1.5  # relax constraint
            if verbose:
                print(f"Iteration {it}: Re-running MMG with hausd = {hausd:.3f}")
            args[args.index('-hausd') + 1] = str(hausd)
            subprocess.run(args, stdout=subprocess.DEVNULL)
            new_mesh = meshio.read(f'{mesh_id}_remeshed.mesh')
            pvmesh = pv.utilities.from_meshio(new_mesh)
            qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')

        if verbose:
            print('Final max aspect ratio:', np.max(qual['CellQuality']))

    # Clean up temporary files
    for ext in ['.mesh', '_remeshed.mesh', '_remeshed.sol']:
        try:
            os.remove(f'{mesh_id}{ext}')
        except FileNotFoundError:
            pass

    return pvmesh.extract_surface()
'''


import scipy.io
import pyvista as pv
import numpy as np


def get_bounds(mesh):
    # FUNZIONE CHE DATO IN INGRESSO UNA MESH DI SUPERFICIE IN FORMATO .STL RESTITUISCE LE SUPERFICI INFLOW, OUTFLOW E WALL DELLA MESH COME .VTP

    # vertices1 = np.array(mesh.points)
    # vertices1 = vertices1[:, :, np.newaxis]
    # tr = mesh.faces.reshape(-1, 4)[:,1:]
    #
    # # Ensure faces are in the correct format (add '3' at the beginning)
    # tr_pyvista = np.hstack([np.full((tr.shape[0], 1), 3), tr])  # Add 3 to the beginning of each row
    #
    # # Select the first frame (938x3) of sparse matrix
    # vr = vertices1[:, :, 0]
    #
    # # Create the PyVista mesh
    # mesh = pv.PolyData(vr, tr_pyvista)

    # Extract the points (vertices) of the mesh
    points = mesh.points

    # global_node_ids = np.arange(points.shape[0])
    # mesh.point_data["GlobalNodeID"] = global_node_ids + 1
    # global_element_ids = np.arange(mesh.n_cells) + 1
    # mesh.cell_data["GlobalElementID"] = global_element_ids
    # model_region_ids = np.ones(mesh.n_cells, dtype=int)
    # mesh.cell_data["ModelRegionID"] = model_region_ids
    mesh.cell_data["Normals"] = mesh.cell_normals

    # Ensure the points don't contain NaN or Inf
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        print("Warning: Mesh contains invalid points (NaN or Inf).")
        points = points[~np.isnan(points).any(axis=1)]  # Remove invalid points


    # Calcola le normali delle celle
    cell_normals = np.array(mesh.cell_normals)
    point_normals = np.array(mesh.point_normals)

    normals = cell_normals

    # Tolleranza per considerare due normali "uguali" (più piccolo = più severo)
    tolerance = np.sqrt(2 * (1 - np.cos(np.radians(10))))  # Tolleranza per angolo di 10 gradi (Distanza euclidea)
    threshold = np.cos(np.radians(10)) # Tolleranza per angolo di 10 gradi (Prodotto scalare)

    # Etichettatura per le regioni piatte
    flat_regions = np.full(mesh.n_cells, -1, dtype=int)  # -1 significa non assegnato
    region_id = 0

    # Funzione per trovare regioni connesse
    def flood_fill(cell_id, region_id):
        """Assegna una regione a tutti i triangoli connessi e normali simili"""
        stack = [cell_id]
        flat_regions[cell_id] = region_id

        while stack:
            current = stack.pop()
            for neighbor in mesh.cell_neighbors(current):
                if flat_regions[neighbor] == -1:  # Se non assegnato
                    if np.dot(normals[current], normals[neighbor]) > threshold and np.linalg.norm(normals[current] - normals[neighbor]) < tolerance:
                        flat_regions[neighbor] = region_id
                        stack.append(neighbor)

    # Scansione di tutti i triangoli
    for i in range(mesh.n_cells):
        if flat_regions[i] == -1:  # Se non assegnato a una regione
            flood_fill(i, region_id)
            region_id += 1

    # Aggiungi i dati della regione alla mesh
    mesh.cell_data["ModelFaceID"] = flat_regions + 1
    # mesh.plot(scalars="ModelFaceID", cmap="viridis")
    regions, counts = np.unique(flat_regions, return_counts=True)
    # print("Regions and their sizes:", list(zip(regions.tolist(), counts.tolist())))

    # Check if any regions were found
    if len(regions) == 0:
        raise ValueError("No large flat regions detected. Try adjusting min_size or verifying flat_regions.")

    # Store identified inflow and outflow regions
    inflow_region = None
    outflow_region = None


    for region_id in regions:
        # Extract cell IDs of this region
        cell_ids = np.where(flat_regions == region_id)[0]

        # Get the corresponding vertices
        region_vertices = np.array(mesh.points[np.unique(mesh.regular_faces[cell_ids].reshape(-1))])
        region_faces = np.array(mesh.regular_faces[cell_ids])
        if len(region_vertices) > 10:
            area = np.zeros(len(region_faces))

            for i in range(len(region_faces)):
                a = region_faces[i][0]
                b = region_faces[i][1]
                c = region_faces[i][2]
                x = np.array(mesh.points[region_faces[i]])
                AB = np.array(x[1] - x[0])
                AC = np.array(x[2] - x[0])
                area[i] = 0.5 * np.linalg.norm(np.cross(AB, AC))

            region_area = np.sum(area)

            perimeter = 0

            # Dictionary to count edge occurrences
            edge_count = {}

            # Iterate over all faces to count edge occurrences
            for face in region_faces:
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i + 1) % 3]]))  # Ensure a consistent order
                    if edge in edge_count:
                        edge_count[edge] += 1  # Increment count if edge already exists
                    else:
                        edge_count[edge] = 1  # Initialize count for new edge

            # Extract only the boundary edges (edges appearing exactly once)
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

            for edge in boundary_edges:
                # Compute the length of the edge
                v1 = np.array(mesh.points[edge[0]])
                v2 = np.array(mesh.points[edge[1]])
                length = np.linalg.norm(v2 - v1) # Euclidean distance
                perimeter += length

            # print(f"Region {region_id}: Area = {region_area}, Perimeter = {perimeter}")

            circularity = 4 * np.pi * region_area / (perimeter ** 2)
            # print(f"Region {region_id} circularity score: {circularity:.3f}")

            if 0.9 < circularity < 1.1:  # Circular region
                if inflow_region is None:
                    inflow_region = cell_ids
                else:
                    outflow_region = cell_ids


    if inflow_region is None or outflow_region is None:
        raise ValueError("Failed to detect inflow and/or outflow surfaces. Try adjusting the circularity threshold.")

    mesh_inflow = mesh.extract_cells(inflow_region)
    mesh_outflow = mesh.extract_cells(outflow_region)

    # Extract the points (vertices) and the connectivity from the UnstructuredGrid mesh
    vertices_in = mesh_inflow.points
    connectivity_in = mesh_inflow.cells.reshape(-1, 4)  # Remove the first column which is the number of vertices per cell
    vertices_out = mesh_outflow.points
    connectivity_out = mesh_outflow.cells.reshape(-1, 4)  # Remove the first column which is the number of vertices per cell

    # Create a PolyData object from the extracted points and connectivity
    mesh_inflow_polydata = pv.PolyData(vertices_in, connectivity_in)
    mesh_outflow_polydata = pv.PolyData(vertices_out, connectivity_out)

    # Now compute normals on the PolyData
    mesh_inflow_polydata.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    mesh_outflow_polydata.compute_normals(cell_normals=True, point_normals=False, inplace=True)

    inflow_normals = np.array(mesh_inflow_polydata.cell_normals)
    ins = np.sum(inflow_normals, axis=0)
    inflow_normal = ins / np.linalg.norm(ins)
    outflow_normals = np.array(mesh_outflow_polydata.cell_normals)
    outs = np.sum(outflow_normals, axis=0)
    outflow_normal = outs / np.linalg.norm(outs)

    if np.dot(inflow_normal, [0, 0, 1]) > np.cos(np.radians(10)) > np.dot(outflow_normal, [0, 0, 1]):
        print("Inflow normal is aligned with Z-axis and outflow normal is not aligned with Z-axis. --> OK")
    else:
        print("Inflow normal is not aligned with Z-axis or outflow normal is aligned with Z-axis. --> Swapping inflow and outflow regions.")
        i = inflow_region
        inflow_region = outflow_region
        outflow_region = i

    # Sovrascrivi i ModelFaceID con i valori desiderati
    model_face_ids = np.full(mesh.n_cells, 1, dtype=int)  # Default a wall = 1
    model_face_ids[inflow_region] = 2  # Inlet = 2
    model_face_ids[outflow_region] = 3  # Outlet = 3
    mesh.cell_data["ModelFaceID"] = model_face_ids

    # print("Inflow Region Cells:", inflow_region)
    # print("Outflow Region Cells:", outflow_region)

    mesh_inflow = mesh.extract_cells(inflow_region)
    mesh_outflow = mesh.extract_cells(outflow_region)
    mesh_walls = mesh.extract_cells(np.setdiff1d(np.arange(mesh.n_cells), np.concatenate([inflow_region, outflow_region])))

    # # Riassocia i GlobalNodeID ai punti delle sotto-mesh
    # def map_global_node_ids(submesh, fullmesh):
    #     # Trova per ogni punto della submesh il suo indice nel fullmesh
    #     full_points = fullmesh.points
    #     sub_points = submesh.points
    #     ids = []
    #
    #     for pt in sub_points:
    #         idx = np.where(np.all(np.isclose(full_points, pt, atol=1e-8), axis=1))[0]
    #         if idx.size == 0:
    #             raise ValueError("Punto non trovato nella mesh originale.")
    #         ids.append(fullmesh.point_data["GlobalNodeID"][idx[0]])
    #
    #     return np.array(ids)

    # Extract vertices matrix and connectivity matrix for each mesh region
    vertices_inflow = np.array(mesh_inflow.points)
    connectivity_inflow = mesh_inflow.cells.reshape(-1, 4)

    vertices_outflow = np.array(mesh_outflow.points)
    connectivity_outflow = mesh_outflow.cells.reshape(-1, 4)

    vertices_walls = np.array(mesh_walls.points)
    connectivity_walls = mesh_walls.cells.reshape(-1, 4)

    # Convert the meshes to PolyData before saving
    mesh_inflow_poly = pv.PolyData(mesh_inflow.points, mesh_inflow.cells)
    mesh_outflow_poly = pv.PolyData(mesh_outflow.points, mesh_outflow.cells)
    mesh_walls_poly = pv.PolyData(mesh_walls.points, mesh_walls.cells)

    # Keep GlobalNodeID for each region
    for submesh in [mesh_inflow, mesh_outflow, mesh_walls]:
        global_node_ids = mesh.point_data["GlobalNodeID"]
        submesh.point_data["GlobalNodeID"] = global_node_ids[submesh.point_data["vtkOriginalPointIds"]]
        global_element_ids = mesh.cell_data["GlobalElementID"]
        submesh.cell_data["GlobalElementID"] = global_element_ids[submesh.cell_data["vtkOriginalCellIds"]]
        model_face_ids = mesh.cell_data["ModelFaceID"]
        submesh.cell_data["ModelFaceID"] = model_face_ids[submesh.cell_data["vtkOriginalCellIds"]]
        model_region_ids = mesh.cell_data["ModelRegionID"]
        submesh.cell_data["ModelRegionID"] = model_region_ids[submesh.cell_data["vtkOriginalCellIds"]]

    mesh_inflow_poly = pv.PolyData(mesh_inflow.points, mesh_inflow.cells)
    mesh_inflow_poly.point_data["GlobalNodeID"] = mesh_inflow.point_data["GlobalNodeID"]
    mesh_inflow_poly.cell_data["GlobalElementID"] = mesh_inflow.cell_data["GlobalElementID"]
    mesh_inflow_poly.cell_data["ModelFaceID"] = mesh_inflow.cell_data["ModelFaceID"]
    mesh_inflow_poly.cell_data["ModelRegionID"] = mesh_inflow.cell_data["ModelRegionID"]
    mesh_inflow_poly.cell_data["Normals"] = mesh_inflow_poly.cell_normals

    mesh_outflow_poly = pv.PolyData(mesh_outflow.points, mesh_outflow.cells)
    mesh_outflow_poly.point_data["GlobalNodeID"] = mesh_outflow.point_data["GlobalNodeID"]
    mesh_outflow_poly.cell_data["GlobalElementID"] = mesh_outflow.cell_data["GlobalElementID"]
    mesh_outflow_poly.cell_data["ModelFaceID"] = mesh_outflow.cell_data["ModelFaceID"]
    mesh_outflow_poly.cell_data["ModelRegionID"] = mesh_outflow.cell_data["ModelRegionID"]
    mesh_outflow_poly.cell_data["Normals"] = mesh_outflow_poly.cell_normals

    mesh_walls_poly = pv.PolyData(mesh_walls.points, mesh_walls.cells)
    mesh_walls_poly.point_data["GlobalNodeID"] = mesh_walls.point_data["GlobalNodeID"]
    mesh_walls_poly.cell_data["GlobalElementID"] = mesh_walls.cell_data["GlobalElementID"]
    mesh_walls_poly.cell_data["ModelFaceID"] = mesh_walls.cell_data["ModelFaceID"]
    mesh_walls_poly.cell_data["ModelRegionID"] = mesh_walls.cell_data["ModelRegionID"]
    mesh_walls_poly.cell_data["Normals"] = mesh_walls_poly.cell_normals

    # # Save the regions as .vtp files
    # mesh.save("surf_mesh.vtp")
    # mesh_inflow_poly.save("inlet.vtp")
    # mesh_outflow_poly.save("outlet.vtp")
    # mesh_walls_poly.save("wall.vtp")

    print("VTP files saved successfully.")

    # Save the vertices of inflow and outflow regions
    np.save("inlet_nodes.npy", vertices_inflow)
    np.save("outlet_nodes.npy", vertices_outflow)

    # Plot both circular regions
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_inflow, color='blue', show_edges=True, opacity=0.7)
    plotter.add_mesh(mesh_outflow, color='red', show_edges=True, opacity=0.7)
    plotter.add_mesh(mesh_walls, color='green', show_edges=True, opacity=0.7)
    # plotter.add_mesh(mesh,color='green', show_edges=True, opacity=0.7)
    plotter.add_axes()
    plotter.close()

    print('Done')
    return mesh_inflow_poly, mesh_outflow_poly, mesh_walls_poly, mesh

def write_motion(displacement, time, face, surf, name):
    N_nodes = face.n_points
    N_frames = time.shape[0]
    N_vectors = displacement.shape[1]

    # Crea un dizionario che mappa ID -> indice
    id_array = surf.point_data["GlobalNodeID"]
    id_to_index = {id_: i for i, id_ in enumerate(id_array)}

    # Funzione per ottenere i dati per un dato ID
    def get_vertex_data_by_id(vertex_id):
        index = id_to_index.get(vertex_id)
        if index is None:
            raise ValueError(f"L'ID {vertex_id} non è presente.")
        return displacement[index, :, :]  # restituisce una matrice 3xT (x,y,z nel tempo)

    # face_id = face.cell_data["ModelFaceID"][0]
    with open(f"{name}_displacement.dat", "w") as f:
        # Riga 1: numero nodi e numero frame
        f.write(f"{N_vectors} {N_frames} {N_nodes}\n")

        for t in range(N_frames):
            time_val = time[t] / 1000
            f.write(f"{time_val:.6f}\n")

        # Per ogni frame temporale
        for i in range(N_nodes):
            # Recupero coordinate iniziali
            vertex_id = face.point_data["GlobalNodeID"][i]
            f.write(f"{vertex_id}\n")

            # Scrittura delle coordinate aggiornate
            for t in range(N_frames):
                dx, dy, dz = get_vertex_data_by_id(vertex_id)[:, t]
                # x_disp = x + dx
                # y_disp = y + dy
                # z_disp = z + dz
                f.write(f"{dx:.6e} {dy:.6e} {dz:.6e}\n")

    print(f"Displacement saved to {name}_displacement.dat")
    return
import gmsh

def cylinder (R_cyl=5, H_cyl=60, lc=1.0):
    """
    Create a cylinder surface mesh using GMSH and save it as an STL file.

    Parameters:
    R_cyl (float): Cylinder radius.
    H_cyl (float): Cylinder height.
    lc (float): Mesh element size.
    """

    # Initialize GMSH
    gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("cylinder_surface")

    # Create a cylinder surface (no thickness)
    gmsh.model.occ.addCylinder(0, 0, -H_cyl/2, 0, 0, H_cyl, R_cyl, tag=1) # Cylinder with center at (0,0,-H/2) and radius R_cyl
    gmsh.model.occ.remove([(3,1)])  # Remove volume to keep only the surface
    gmsh.model.occ.synchronize() # Synchronize the model to ensure all operations are applied

    # Mesh it as a surface (2D elements)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc*0.1)  # Minimum element size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)  # Maximum element size
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay-based remeshing

    gmsh.model.mesh.generate(2) # Generate the mesh for 2D elements
    # gmsh.fltk.run() # Decomment to open the GMSH GUI to visualize the mesh
    gmsh.option.setNumber("Mesh.Binary", 1)  # Save mesh in binary format
    gmsh.write("cylinder_surface.stl")  # Save the mesh as STL file

    gmsh.finalize()  # Finalize GMSH

    return "cylinder_surface.stl"

def pv_to_np (mesh):
    points = np.array(mesh.points)
    faces = np.array(mesh.faces.reshape((-1, 4))[:, 1:])
    return points,  faces

def np_to_pv (points, faces):
    mesh = pv.PolyData(points,np.hstack([np.full((faces.shape[0],1),3),faces]))
    return mesh


import numpy as np


def mesh_quality(mesh):
    """
    Compute various quality metrics for a given mesh.

    Parameters:
    mesh (pyvista.PolyData): The input mesh to analyze.

    Returns:
    dict: A dictionary containing average, min, and max values for each metric.
    And prints the results.
    """
    # Define the quality metrics to compute
    metrics = ['aspect_ratio', 'radius_ratio', 'shape', 'min_angle', 'max_angle', 'area']

    # Initialize dictionary to hold average, min and max values
    average_metrics = {}
    min_metrics = {}
    max_metrics = {}

    # Loop through metrics and compute each one
    for metric in metrics:
        result = mesh.compute_cell_quality(quality_measure=metric)
        quality_values = result.cell_data["CellQuality"]

        # Save the array to the mesh
        mesh.cell_data[f'{metric}'] = quality_values

        # Compute and store the average, min and max values
        average_value = np.mean(quality_values)
        average_metrics[metric] = average_value
        min_value = np.min(quality_values)
        min_metrics[metric] = min_value
        max_value = np.max(quality_values)
        max_metrics[metric] = max_value

    # Print the average values
    print("Average Mesh Quality Metrics:")
    for metric, value in average_metrics.items():
        print(f"  {metric:>12}: {value:.4f}")

    # # Print the min values
    # print("\nMinimum Mesh Quality Metrics:")
    # for metric, value in min_metrics.items():
    #     print(f"  {metric:>12}: {value:.4f}")
    #
    # # Print the max values
    # print("\nMaximum Mesh Quality Metrics:")
    # for metric, value in max_metrics.items():
    #     print(f"  {metric:>12}: {value:.4f}")

    return average_metrics, min_metrics, max_metrics


def gmsh_remesh(mesh, target_size=0.5):
    import meshio
    import gmsh
    import numpy as np
    import pyvista as pv
    import math
    import os

    # Convert pyvista mesh to numpy
    vertices = np.array(mesh.points, dtype=np.float64)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # solo triangoli

    # Save temporary STL file
    meshio.write("mesh.stl", meshio.Mesh(points=vertices, cells=[("triangle", faces)]))

    gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.merge("mesh.stl")

    angle = 40  # min: 15, max: 120
    forceParametrizablePatches = 0  # 0: False, 1: True
    includeBoundary = True
    curveAngle = 180

    gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary,
                                     forceParametrizablePatches,
                                     curveAngle * math.pi / 180.)
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size)
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay-based remeshing
    gmsh.option.setNumber("Mesh.Smoothing", 0)  # Apply 2 smoothing iterations
    gmsh.option.setNumber("Mesh.Optimize", 1)  # Enable optimization
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)  # Optimization threshold (0-1)

    gmsh.model.mesh.generate(2)
    gmsh.write("remesh.stl")
    gmsh.finalize()

    return pv.read("remesh.stl")


'''
def gmsh_remesh(mesh):
    # Remeshing the mesh
    vertices, faces = pv_to_np(mesh)
    # Ensure vertices are float64
    vertices = np.array(vertices, dtype=np.float64)

    # Ensure triangles are int32 (not uint16 or uint8)
    faces = np.array(faces, dtype=np.int32)  # Explicit conversion

    # Create and save the mesh
    mesh = meshio.Mesh(
        points=vertices,
        cells=[("triangle", faces)]
    )

    meshio.write("mesh.stl", mesh) #, mesh, file_format="gmsh22")  # Save in Gmsh format


    gmsh.initialize()

    def createGeometryAndMesh():
        # Clear all models and merge an STL mesh that we would like to remesh (from
        # the parent directory):
        gmsh.clear()
        path = os.path.dirname(os.path.abspath(__file__))
        gmsh.merge("mesh.stl") #(os.path.join(path, os.pardir, 't13_data.stl'))

        # We first classify ("color") the surfaces by splitting the original surface
        # along sharp geometrical features. This will create new discrete surfaces,
        # curves and points.

        # Angle between two triangles above which an edge is considered as sharp,
        # retrieved from the ONELAB database (see below):
        angle = gmsh.onelab.getNumber('Parameters/Angle for surface detection')[0]

        # For complex geometries, patches can be too complex, too elongated or too
        # large to be parametrized; setting the following option will force the
        # creation of patches that are amenable to reparametrization:
        forceParametrizablePatches = gmsh.onelab.getNumber(
            'Parameters/Create surfaces guaranteed to be parametrizable')[0]

        # For open surfaces include the boundary edges in the classification
        # process:
        includeBoundary = True

        # Force curves to be split on given angle:
        curveAngle = 180

        gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary,
                                         forceParametrizablePatches,
                                         curveAngle * math.pi / 180.)

        # Create a geometry for all the discrete curves and surfaces in the mesh, by
        # computing a parametrization for each one
        gmsh.model.mesh.createGeometry()

        # Set mesh options (adjust values based on your needs)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)  # Minimum element size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)  # Maximum element size
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay-based remeshing
        gmsh.option.setNumber("Mesh.Smoothing", 0)  # Apply 2 smoothing iterations
        gmsh.option.setNumber("Mesh.Optimize", 1)  # Enable optimization
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)  # Optimization threshold (0-1)

        # Generate the new mesh
        gmsh.model.mesh.generate(2)

        # Note that if a CAD model (e.g. as a STEP file, see `t20.py') is available
        # instead of an STL mesh, it is usually better to use that CAD model instead
        # of the geometry created by reparametrizing the mesh. Indeed, CAD
        # geometries will in general be more accurate, with smoother
        # parametrizations, and will lead to more efficient and higher quality
        # meshing. Discrete surface remeshing in Gmsh is optimized to handle dense
        # STL meshes coming from e.g. imaging systems, where no CAD is available; it
        # is less well suited for the poor quality STL triangulations (optimized for
        # size, with e.g. very elongated triangles) that are usually generated by
        # CAD tools for e.g. 3D printing.
        #
        # # Create a volume from all the surfaces
        # s = gmsh.model.getEntities(2)
        # l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
        # gmsh.model.geo.addVolume([l])
        #
        # gmsh.model.geo.synchronize()
        #
        # # We specify element sizes imposed by a size field, just because we can :-)
        # f = gmsh.model.mesh.field.add("MathEval")
        # if gmsh.onelab.getNumber('Parameters/Apply funny mesh size field?')[0]:
        #     gmsh.model.mesh.field.setString(f, "F", "2*Sin((x+y)/5) + 3")
        # else:
        #     gmsh.model.mesh.field.setString(f, "F", "1")
        # gmsh.model.mesh.field.setAsBackgroundMesh(f)
        #
        # gmsh.model.mesh.generate(3)
        # gmsh.model.mesh.optimize()
        #
        gmsh.model.geo.synchronize()
        gmsh.write('remesh.stl')


        # Create ONELAB parameters with remeshing options:
        gmsh.onelab.set("""[
          {
            "type":"number",
            "name":"Parameters/Angle for surface detection",
            "values":[40],
            "min":15,
            "max":120,
            "step":1
          },
          {
            "type":"number",
            "name":"Parameters/Create surfaces guaranteed to be parametrizable",
            "values":[0],
            "choices":[0, 1]
          },
          {
            "type":"number",
            "name":"Parameters/Apply funny mesh size field?",
            "values":[0],
            "choices":[0, 1]
          }
        ]""")

    # Create the geometry and mesh it:
    createGeometryAndMesh()

    # Launch the GUI and handle the "check" event to recreate the geometry and mesh
    # with new parameters if necessary:
    def checkForEvent():
        action = gmsh.onelab.getString("ONELAB/Action")
        if len(action) and action[0] == "check":
            gmsh.onelab.setString("ONELAB/Action", [""])
            createGeometryAndMesh()
            gmsh.graphics.draw()
        return True

    if "-nopopup" not in sys.argv:
        gmsh.fltk.initialize()
        while gmsh.fltk.isAvailable() and checkForEvent():
            gmsh.fltk.wait()

    gmsh.finalize()

    grid = pv.read('remesh.stl')

    # Ensure it's an unstructured grid
    if not isinstance(grid, pv.UnstructuredGrid):
        grid = grid.cast_to_unstructured_grid()

    # Extract cell connectivity
    cells = []
    offset = 0
    cell_array = grid.cells
    num_cells = grid.n_cells

    # List to store centroids
    centroids = []

    for _ in range(num_cells):
        n_points = cell_array[offset]  # Number of points in the cell
        cell = cell_array[offset + 1: offset + 1 + n_points]  # Extract point indices
        offset += n_points + 1  # Move to the next cell
        cells.append(cell)

        # Compute centroid for this cell
        centroid = grid.points[cell].mean(axis=0)
        centroids.append(centroid)

    # Convert to NumPy array
    cell_centroids = np.array(centroids)

    # Extract cells below the xy-plane (z < 0)
    mask = cell_centroids[:, 2] < 0
    cell_indices = np.where(mask)[0]
    subgrid = grid.extract_cells(cell_indices)

    # Visualization
    plotter = pv.Plotter()
    plotter.add_mesh(subgrid, color='lightgrey', lighting=True, show_edges=True)
    plotter.add_mesh(grid.extract_surface(), color='r', style='wireframe')  # Display wireframe of original mesh
    plotter.add_legend([['Input Mesh', 'r'], ['Extracted Mesh', 'black']])
    plotter.show()

    return grid
'''