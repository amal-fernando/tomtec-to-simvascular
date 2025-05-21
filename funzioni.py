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
    print("Calcolando il volume per ciascun frame usando trimesh...")
    vol = np.zeros(matrix1.shape[2])  # Preallocazione del vettore vol

    for j in range(matrix1.shape[2]):
        # Estrai i vertici del frame corrente
        vertices = matrix1[:, :, j]
        
        # Crea la mesh e calcola il volume
        mesh = trimesh.Trimesh(vertices=vertices, faces=TM1)
        vol[j] = mesh.volume  # Salva il volume per il frame corrente

    print("Fatto.")
    return vol


def fourier(matrix2, time, num_intermedie):
    print('Increasing number of frames with Fourier interpolation.')

    # Numero totale di frame interpolati
    num_frames_original = matrix2.shape[2]
    num_frames_interpolati = num_frames_original + (num_frames_original - 1) * num_intermedie

    # Pre-allocazione per la nuova matrice
    new_matrix = np.zeros((matrix2.shape[0], matrix2.shape[1], num_frames_interpolati))
    new_matrix_deriv = np.zeros_like(new_matrix)
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

            # Calcolo derivata nel dominio di Fourier
            F_deriv = 1j * 2 * np.pi * freqs * F
            deriv = np.fft.ifft(F_deriv) * len(F) / len(signal)
            new_matrix_deriv[i, j, :] = np.real(deriv)

    fasi = num_frames_interpolati
    print('Done.')

    return new_matrix, new_matrix_deriv, new_time, fasi


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


def normalplot(v,f,i):
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
    plotter.show()

import os
import subprocess
import meshio

def mmg_remesh(input_mesh, hausd=0.3, hmax=2, hmin=1.5, max_aspect_ratio=None, max_iter=3, verbose=False):
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
                    '-nr',
                    '-nreg',
                    '-xreg',
                    '-optim',
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
                            '-nr',
                            '-nreg',
                            '-xreg',
                            '-optim',
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