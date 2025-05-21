from scipy.io import loadmat
from funzioni import resample_u
from funzioni import point2trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Carica il file .mat
mat_data1 = loadmat('faces_fitto.mat')
mat_data2 = loadmat('vertices_fitto.mat')
mat_data3 = loadmat('f1.mat')
mat_data4 = loadmat('v1.mat')

# Visualizza le chiavi del dizionario
print(mat_data1.keys())
print(mat_data2.keys())
print(mat_data3.keys())
print(mat_data4.keys())

# Supponiamo che la matrice si chiami 'v2' nel file .mat
f2 = mat_data1['faces'] - 1
v2 = mat_data2['vertices']
f1 = mat_data3['f'] - 1
v1 = mat_data4['v']

# Controlla il tipo di dati
print(type(f2))  # Solitamente sarà un numpy.ndarray
print(type(v2))  # Solitamente sarà un numpy.ndarray
print(type(f1))  # Solitamente sarà un numpy.ndarray
print(type(v1))  # Solitamente sarà un numpy.ndarray

# Usa v2 come una normale matrice NumPy
print(f2.shape)  # Dimensioni della matrice
print(v2.shape)  # Contenuto della matrice
print(f1.shape)  # Dimensioni della matrice
print(v1.shape)  # Contenuto della matrice


# Inizializzazione di matrix3 come array di zeri
matrix3 = np.zeros((v2.shape[0], v2.shape[1], v1.shape[2]))

# Impostazione di tr come f1 e p come v2
tr = f1
p = v2[:, :, 0]

# Ciclo per il calcolo e la proiezione
for i in range(v1.shape[2] - 1):
    vr = v1[:,:,i]        # Estrai la mesh dal frame i
    ur = v1[:,:,i+1] - v1[:,:,i]  # Calcola la variazione tra il frame i+1 e i
    pp, upp = resample_u(vr, tr, ur, p)  # Usa la funzione per interpolare e proiettare i punti
    matrix3[:,:,i] = pp       # Salva la proiezione dei punti nel frame i di matrix3
    p = upp + pp              # Aggiorna p come la somma tra gli spostamenti e la proiezione
    matrix3[:,:,i+1] = p      # Salva la posizione aggiornata nel frame i+1


# Example data for a triangular mesh (replace this with your own data)
# Vertices (3D points)
p = v2[:, :, 0]
T_vertices = v1[:, :, 0]
T_faces = f1

# Faces (triangular elements, indices of vertices)
faces = f2
vertices, _ = point2trimesh(T_faces, T_vertices, p)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the triangulated surface using plot_trisurf directly
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color='cyan', edgecolor='black', alpha=0.5)

# Optional: customize the plot
ax.set_box_aspect([1,1,1])  # Equal aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trimesh Surface')

# Show the plot
plt.show()
