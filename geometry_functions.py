# geometry_functions.py
import numpy as np
from sklearn.decomposition import PCA


def find_sides(pid, p, tri, lin=None):
    """
    Find neighboring points and connecting vectors for a point in a triangulated mesh.
    
    Parameters:
    pid : int
        Point ID to analyze
    p : ndarray
        Point coordinates array
    tri : ndarray
        Triangulation connectivity matrix
    lin : ndarray, optional
        Pre-ordered list of neighbors
        
    Returns:
    tuple:
        - vicini: ordered list of neighbor points
        - lati: vectors from pid to neighbors
        - valenza: number of neighbors
    """
    
    if lin is None:
        # Find triangles containing pid
        a = np.where(tri == pid)[0]
        vicini = tri[a].flatten()
        # Remove pid and duplicates
        vicini = np.unique(vicini[vicini != pid])
        #print(vicini)
    else:
        vicini = lin

    valenza = len(vicini)
    # Create vectors from pid to neighbors
    lati = p[vicini] - np.tile(p[pid], (valenza, 1))

    if lin is None:
        # Order vectors using PCA
        pca = PCA(n_components=2)
        latirot = pca.fit_transform(lati)
        theta = np.arctan2(latirot[:, 1], latirot[:, 0])
        idx = np.argsort(theta)
        lati = lati[idx]
        vicini = vicini[idx]

    return vicini, lati, valenza
