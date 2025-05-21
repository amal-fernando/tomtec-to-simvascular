import scipy.io
import pyvista as pv
import numpy as np


def get_bounds(nome_file):
    # FUNZIONE CHE DATO IN INGRESSO UNA MESH DI SUPERFICIE IN FORMATO .STL RESTITUISCE LE SUPERFICI INFLOW, OUTFLOW E WALL DELLA MESH COME .VTP
    mesh = pv.read(nome_file)

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
                    if np.dot(normals[current], normals[neighbor]) > threshold: # np.linalg.norm(normals[current] - normals[neighbor]) < tolerance:
                        flat_regions[neighbor] = region_id
                        stack.append(neighbor)

    # Scansione di tutti i triangoli
    for i in range(mesh.n_cells):
        if flat_regions[i] == -1:  # Se non assegnato a una regione
            flood_fill(i, region_id)
            region_id += 1

    # Aggiungi i dati della regione alla mesh
    mesh.cell_data["ModelFaceID"] = flat_regions + 1
    # mesh.plot(scalars="Flat Regions", cmap="viridis")
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

    # Save the regions as .vtp files
    mesh.save("surf_mesh.vtp")
    mesh_inflow_poly.save("inlet.vtp")
    mesh_outflow_poly.save("outlet.vtp")
    mesh_walls_poly.save("wall.vtp")

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
    plotter.show()

    print('Done')
    return mesh_inflow_poly, mesh_outflow_poly, mesh_walls_poly

def write_motion(displacement, time, face, surf):
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

    with open(f"{face}.dat", "w") as f:
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

    print(f"Displacement saved to {face}.dat")
    return

