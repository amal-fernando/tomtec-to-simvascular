import numpy as np
import scipy.spatial as spatial
import scipy.interpolate as interp
import meshio  # per leggere/scrivere mesh
from funzioni import timeplot, volume, resample_u, riordina, fourier
from test import get_bounds, write_motion
import os
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
import gmsh
from scipy.spatial import cKDTree
import tetgen
from funzioni import mmg_remesh


# === 1. CARICAMENTO DELLA MESH ===

time = np.load('time.npy')
# mesh_orig = meshio.read("original_mesh.vtk")
rr = 1018.67
ed = 942.905
mesh_cyl = meshio.read("cylinder_surface.stl")
T= len(time)  # Numero di frame temporali

# P_orig = mesh_orig.points  # Coordinate dei nodi originali (N1 x 3)
P_cyl = mesh_cyl.points  # Coordinate dei nodi del cilindro (N2 x 3)
T_cyl = mesh_cyl.cells_dict["triangle"]  # Triangoli del cilindro (M x 3)

# Mesh del ventricolo
P_orig = np.load('v0_orig.npy')
U_orig = np.load('u_orig.npy')
T_orig = np.load('f_orig.npy')

P_orig_t = P_orig[:, :, np.newaxis] + U_orig  # (N1, 3, T)


# === 4. CALCOLO DEL VOLUME ORIGINALE ===

# Calcola i volumi nel tempo
V_orig = volume(P_orig_t, T_orig)  # Calcola il volume originale

# _,_,cyl = get_bounds("mesh.stl")

# === 5. CALCOLO FATTORI DI SCALA VOLUMETRICA ===
V0 = V_orig[0]
scaling_factors = (V_orig / V0) ** (1/3)  # Radice cubica per il fattore di scala volumetrico

# === 6. APPLICAZIONE SCALATURA AL CILINDRO ===
# Centra il cilindro nel baricentro
P_center = np.mean(P_cyl, axis=0)
P_cyl_centered = P_cyl - P_center

P_cyl_deformed = np.zeros((P_cyl.shape[0], 3, T))

for t in range(T):
    s = scaling_factors[t]
    P_scaled = P_cyl_centered.copy()
    P_scaled[:, 0:2] *= s  # Deformazione solo radiale (x, y)
    P_cyl_deformed[:, :, t] = P_scaled + P_center

# Riordina segnale
[time_cyl,P_cyl_deformed] = riordina(time,P_cyl_deformed,ed,rr)

# timeplot(P_cyl_deformed,T_cyl)

V_cyl = volume(P_cyl_deformed, T_cyl) # Calcola il volume del cilindro deformato


# get_bounds('cylinder_surface.stl')

# plt.figure()
# plt.plot(time,v_orig, '*', label='volOriginal')
# plt.plot(time_cyl, v_cyl, label='volCyl')
# plt.legend()
# plt.grid(True)
# plt.xlabel('Frame')
# plt.ylabel('Volume')
# plt.title('Volume Over Time')
# plt.show()

# Estensione del segnale
# Numero totale di sequenze (inclusa quella originale)
num_sequences = 3
nt = len(time_cyl) - 1  # Elimino l'ultimo dato, che deve essere inserito solo in coda alla sequenza

# Duplica la matrice iniziale
matrix2 = np.tile(P_cyl_deformed[:, :, :nt], (1, 1, num_sequences + 1))  # Crea un ciclo di troppo inizialmente
matrix2 = matrix2[:, :, :-nt+1]  # Elimina i dati dopo il duplicato del primo in coda alla sequenza

# print('matrix2.shape =', matrix2.shape)
# print(matrix2[:,:,75])

fasi = matrix2.shape[2]

# Costruzione del vettore time2
time2 = np.zeros(num_sequences * nt + 1)

for i in range(1, num_sequences + 1):
    time2[(i - 1) * nt:i * nt] = time_cyl[:nt] + (i - 1) * rr
time2[-1] = rr * num_sequences

volOriginal = volume(matrix2, T_cyl)

# Parametri di input
num_intermedie = 40

# Inizializzazione del vettore vol
vol = np.zeros(len(volOriginal) + (len(volOriginal) - 1) * num_intermedie)
tempo = np.zeros(len(vol))
index = 0  # Indice per riempire vol

for i in range(len(volOriginal)):
    vol[index] = volOriginal[i]  # Copia il valore originale
    tempo[index] = time2[i]  # Copia il valore originale
    index += 1  # Avanza l'indice

    if i < len(volOriginal) - 1:  # Evita l'ultimo caso
        vol[index:index + num_intermedie] = np.nan  # Riempie con NaN
        tempo[index:index + num_intermedie] = np.linspace(time2[i], time2[i + 1], num_intermedie + 2)[1:-1]
        index += num_intermedie  # Avanza l'indice dopo gli NaN

volOriginal = vol
print('volOriginale.shape =', volOriginal.shape)
print(volOriginal)

timeOriginal = tempo
print('timeOriginale.shape =', timeOriginal.shape)
print(timeOriginal)

new_time_frames = len(timeOriginal)  # Number of time frames for interpolation


matrixPchip = np.zeros((matrix2.shape[0], 3, new_time_frames))

# Use PCHIP interpolation for each point
for i in range(matrix2.shape[0]):
    for dim in range(3):
        coordinates = matrix2[i, dim, :]
        pchip_interp = PchipInterpolator(time2, coordinates)
        matrixPchip[i, dim, :] = pchip_interp(timeOriginal)


# Fourier interpolation
#matrixFourier = np.zeros((matrix2.shape[0], 3, new_time_frames))
[matrixFourier, derivFourier,timeFourier,fas] = fourier(matrixPchip, time2, 0)
v_cyl = volume(matrixFourier, T_cyl)
# timeplot(matrixFourier,T_cyl)


mesh = pv.PolyData(matrixFourier[:,:,0],np.hstack([np.full((T_cyl.shape[0],1),3),T_cyl]))
mesh.save("cyl.stl")

# Generate the volume mesh
points = mesh.points
faces = mesh.faces.reshape((-1, 4))[:, 1:]  # Drop the leading 3s

# TetGen options (preserve surface, good quality)
tetgen_options = "pq1.2a0.005"

tet = tetgen.TetGen(points, faces)
tet.tetrahedralize(order=1, switches="pq1.2")

# Convert to PyVista mesh
tetra_mesh = tet.grid

# Save as VTU
tetra_mesh.save("mesh-complete.mesh.vtu")

mesh_volume = pv.read("mesh-complete.mesh.vtu")
mesh_volume.point_data["GlobalNodeID"] = np.arange(mesh_volume.points.shape[0]) + 1
mesh_volume.cell_data["GlobalElementID"] = np.arange(mesh_volume.n_cells) + 1
mesh_volume.cell_data["ModelRegionID"] = np.ones(mesh_volume.n_cells, dtype=int)  # Set to one for all cells
mesh_volume.save("mesh-complete.mesh.vtu")

surf_mesh = mesh_volume.extract_geometry()

surf_mesh.save("surf_mesh.vtp")

_,_,_ = get_bounds("surf_mesh.vtp") # ("cyl.stl")
# mesh["Displacement"] = displacement
# mesh.save("disp_mesh.vtp")


flow_rate = np.gradient(v_cyl, timeFourier.flatten()/1000)  # Automatically handles variable time steps N.B. 1 mm3/ms = 1 cm3/s

# Save to .flow file
with open("cyl.flow", "w") as f:
    for t, q in zip(timeFourier.flatten()/1000, flow_rate):
        # Convert q to a scalar and write to file
        f.write(f"{t:.7f} {q:.7f}\n")

wall = pv.read("wall.vtp")
inlet = pv.read("inlet.vtp")
outlet = pv.read("outlet.vtp")
surf = pv.read("surf_mesh.vtp")

# extracted_surface = mmg_remesh(surf, 0.1, 0.8, 0.4 )

''' Genero file bct.vtp per 'inlet' '''

# === Caricamento dati ===
inlet = pv.read("inlet.vtp")
points = inlet.points
node_ids = inlet.point_data["GlobalNodeID"]
normals = np.array(inlet.point_normals)

v3 = np.load("v3.npy")

# === Tempo e portata ===
v_cyl = volume(v3, T_cyl)

# # Create the plot
# plt.figure()
# plt.plot(timeFourier, v_cyl, label='volume')
# plt.legend()
# plt.grid(True)
# plt.xlabel('Time [ms]')
# plt.ylabel('Volume [mm^3]')
# plt.title('Volume Over Time')
# plt.show()

def fourier_derivative(signal, time):
    dt = time[1] - time[0]
    N = len(signal)
    freqs = np.fft.fftfreq(N, d=dt)

    F = np.fft.fft(signal)
    F_deriv = 1j * 2 * np.pi * freqs * F
    deriv = np.fft.ifft(F_deriv)

    return np.real(deriv)

def write_flow_file(filename, time, flow):
    with open(filename, 'w') as f:
        for t, q in zip(time, flow):
            f.write(f"{t:.6f} {q:.6f}\n")

# 2. Derivata via Fourier
flow_rate_f = fourier_derivative(v_cyl, timeFourier.flatten()/1000)

# 3. Salva in file .flow
write_flow_file("inlet.flow", timeFourier.flatten()/1000, flow_rate)



flow_rate = np.gradient(v_cyl, timeFourier.flatten()/1000)

# Create the plot
plt.figure()
plt.plot(timeFourier, flow_rate, label='flow_rate')
plt.plot(timeFourier, v_cyl, label='volume')
plt.plot(timeFourier, flow_rate_f, label='flow_rate_f')
plt.legend()
plt.grid(True)
plt.xlabel('Time [ms]')
plt.ylabel('Volume [mm^3]')
plt.title('Volume Over Time')
plt.show()


time = timeFourier.flatten()/1000  # Convert to seconds
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
inlet.save("bct.vtp")

'''
# === Salvataggio ===
with open("bct.dat", "w") as f:
    f.write("\n".join(lines))
'''

v2 = np.array(surf.points)
f2 = surf.faces.reshape(-1, 4)[:,1:]  # Remove first column (number of vertices)
v2 = v2[:, :, np.newaxis]

# Initialize matrix3 with the appropriate dimensions
v3 = np.zeros((v2.shape[0], v2.shape[1], matrixFourier.shape[2]))

tr = T_cyl
p = v2[:, :, 0]

# Loop through each time step
# for i in range(matrixFourier.shape[2] - 1):
#     vr = matrixFourier[:, :, i]
#     ur = matrixFourier[:, :, i + 1] - matrixFourier[:, :, i]
#
#     # Assuming resample_u is already defined, and returns pp and upp
#     pp, upp = resample_u(vr, tr, ur, p)
#
#     # Store the results in v3
#     v3[:, :, i] = pp
#     p = upp + pp
#     v3[:, :, i + 1] = p

# np.save("v3.npy", v3)

# Load the saved v3 array
v3 = np.load("v3.npy")

# Calculate the displacement
displacement = v3 - v3[:,:,0][:, :, np.newaxis]

write_motion(displacement, timeFourier, wall, surf, "wall")
write_motion(displacement, timeFourier, inlet, surf, "inlet")
write_motion(displacement, timeFourier, outlet, surf, "outlet")

'''
N_nodes = wall.n_points
N_frames = timeFourier.shape[0]
N_vectors = displacement.shape[1]

print(surf.point_data.keys())  # Controlla se 'GlobalNodeID' è tra le chiavi

# Crea un dizionario che mappa ID -> indice
id_array = surf.point_data["GlobalNodeID"]
id_to_index = {id_: i for i, id_ in enumerate(id_array)}

# Funzione per ottenere i dati per un dato ID
def get_vertex_data_by_id(vertex_id):
    index = id_to_index.get(vertex_id)
    if index is None:
        raise ValueError(f"L'ID {vertex_id} non è presente.")
    return displacement[index, :, :]  # restituisce una matrice 3xT (x,y,z nel tempo)

with open("motion.dat", "w") as f:
    # Riga 1: numero nodi e numero frame
    f.write(f"{N_vectors} {N_frames} {N_nodes}\n")

    for t in range(N_frames):
        time_val = timeFourier[t] / 1000
        f.write(f"{time_val:.6e}\n")

    # Per ogni frame temporale
    for i in range(N_nodes):
        # Recupero coordinate iniziali
        vertex_id = wall.point_data["GlobalNodeID"][i]
        f.write(f"{vertex_id}\n")

        # Scrittura delle coordinate aggiornate
        for t in range(N_frames):
            dx, dy, dz = get_vertex_data_by_id(vertex_id)[:, t]
            # x_disp = x + dx
            # y_disp = y + dy
            # z_disp = z + dz
            f.write(f"{dx:.6e} {dy:.6e} {dz:.6e}\n")



print("Displacement saved to motion.dat")
'''
print("Done!")
# === 7. SALVATAGGIO .VTK DEI FRAME DEFORMATI ===
# output_dir = "deformed_cylinder_vtk"
# os.makedirs(output_dir, exist_ok=True)
#
# for t in range(T):
#     meshio.write_points_cells(
#         f"{output_dir}/cylinder_deformed_t{t:03d}.vtk",
#         points=P_cyl_deformed[:, :, t],
#         cells={"triangle": T_cyl}
#     )