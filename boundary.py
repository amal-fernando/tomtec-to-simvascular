import numpy as np
import pyvista as pv
import cv2
import tkinter as tk
from tkinter import filedialog
from funzioni import pv_to_np, np_to_pv
from scipy.spatial import cKDTree
import numpy as np
import colorcet as cc

glasbey = cc.glasbey  # lista di 256 colori RGB

def get_bounds_v0(mesh):
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
    tolerance = np.sqrt(2 * (1 - np.cos(np.radians(8))))  # Tolleranza per angolo di 10 gradi (Distanza euclidea)
    threshold = np.cos(np.radians(8)) # Tolleranza per angolo di 10 gradi (Prodotto scalare)

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
    mesh.plot(scalars="ModelFaceID", cmap="glasbey")
    regions, counts = np.unique(flat_regions, return_counts=True)
    print("Regions and their sizes:", list(zip(regions.tolist(), counts.tolist())))

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
            print(f"Region {region_id} circularity score: {circularity:.3f}")

            if 0.9 < circularity < 1.1:  # Circular region
                if inflow_region is None:
                    inflow_region = cell_ids
                else:
                    outflow_region = cell_ids

    mesh_inflow = mesh.extract_cells(inflow_region).cell_data["vtkOriginalCellIds"] if inflow_region is not None else None
    mesh_outflow = mesh.extract_cells(outflow_region).cell_data["vtkOriginalCellIds"] if outflow_region is not None else None

    if inflow_region is None:
        print("Inflow surface not detected.")
    if outflow_region is None:
        print("Outflow surface not detected.")

    return inflow_region, outflow_region,mesh_inflow, mesh_outflow

def get_bounds_v1(mesh):
    # FUNZIONE CHE DATO IN INGRESSO UNA MESH DI SUPERFICIE IN FORMATO .STL RESTITUISCE LE SUPERFICI INFLOW, OUTFLOW E WALL DELLA MESH COME .VTP

       # Extract the points (vertices) of the mesh
    points = mesh.points

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
    tolerance = np.sqrt(2 * (1 - np.cos(np.radians(18))))  # Tolleranza per angolo di 10 gradi (Distanza euclidea)
    threshold = np.cos(np.radians(18)) # Tolleranza per angolo di 10 gradi (Prodotto scalare)

    # Etichettatura per le regioni piatte
    flat_regions = np.full(mesh.n_cells, -1, dtype=int)  # -1 significa non assegnato
    region_id = 0
    region_sizes = {}
    region_members = {}

    # Funzione per trovare regioni connesse
    def flood_fill(cell_id, region_id):
        """Assegna una regione a tutti i triangoli connessi e con normali simili alla media della regione"""
        stack = [cell_id]
        flat_regions[cell_id] = region_id
        region_normals = [normals[cell_id]]
        region_cells = [cell_id]

        while stack:
            current = stack.pop()

            for neighbor in mesh.cell_neighbors(current):
                if flat_regions[neighbor] == -1:  # Se non assegnato

                    # Calcola la normale media corrente della regione
                    avg_normal = np.mean(region_normals, axis=0)
                    avg_normal /= np.linalg.norm(avg_normal)  # Normalizza

                    # Verifica rispetto alla normale media della regione
                    dot_prod = np.dot(avg_normal, normals[neighbor])
                    angular_diff = np.linalg.norm(avg_normal - normals[neighbor])

                    if dot_prod > threshold and angular_diff < tolerance:
                        flat_regions[neighbor] = region_id
                        stack.append(neighbor)
                        region_normals.append(normals[neighbor])
                        region_cells.append(neighbor)

        # # Se troppo piccola, resetta la regione
        # if len(region_cells) < 200:
        #     for cell in region_cells:
        #         flat_regions[cell] = -1
        # else:
        #     region_sizes[region_id] = len(region_cells)
        #     region_members[region_id] = region_cells


    # Scansione di tutti i triangoli
    for i in range(mesh.n_cells):
        if flat_regions[i] == -1:  # Se non assegnato a una regione
            flood_fill(i, region_id)
            region_id += 1


    # Aggiungi i dati della regione alla mesh
    mesh.cell_data["ModelFaceID"] = flat_regions + 1
    mesh.plot(scalars="ModelFaceID", cmap="glasbey")
    regions, counts = np.unique(flat_regions, return_counts=True)
    print("Regions and their sizes:", list(zip(regions.tolist(), counts.tolist())))

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
            print(f"Region {region_id} circularity score: {circularity:.3f}")

            if 0.9 < circularity < 1.1:  # Circular region
                if inflow_region is None:
                    inflow_region = cell_ids
                else:
                    outflow_region = cell_ids

    mesh_inflow = mesh.extract_cells(inflow_region) if inflow_region is not None else None
    mesh_outflow = mesh.extract_cells(outflow_region) if outflow_region is not None else mesh_inflow

    if inflow_region is None:
        print("Inflow surface not detected.")
    if outflow_region is None:
        print("Outflow surface not detected.")

    return mesh_inflow, mesh_outflow


def extract_surface_region_from_old(new_mesh, old_surface, distance_threshold=0.2):
    """
    Estrae dalla nuova mesh la regione di superficie che si trova a distanza < threshold
    dalla vecchia superficie (anche se curva).
    """
    # Calcola distanza implicita da ogni punto della nuova mesh alla vecchia superficie
    new_with_dist = new_mesh.compute_implicit_distance(old_surface)
    dist = np.abs(new_with_dist.point_data['implicit_distance'])

    # Seleziona punti vicini
    close_pts = dist < distance_threshold

    # Estrai regione corrispondente
    region = new_mesh.extract_points(close_pts, adjacent_cells=False)

    # Connettività: prendi solo la componente principale
    region = region.connectivity().threshold(0)
    return region.extract_surface()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to select coarse mesh file
    file_path = filedialog.askopenfilename(
        title="Select P000x.vtp file"
    )
    mesh = pv.read(file_path)  # o la tua mesh PyVista
    surf_mesh = mesh.extract_geometry()
    surf_mesh.save("mesh-complete.exterior.vtp")

    # inflow_region, outflow_region, inflow_v0, outflow_v0 = get_bounds_v0(surf_mesh)
    # inflow_v1, outflow_v1 = get_bounds_v1(surf_mesh)

    # np.save("outflow.npy", inflow_v0)
    # np.save("inflow.npy", outflow_v0)


    # inlet_mesh = inflow_v0 if inflow_v0 is not None else inflow_v1
    # outlet_mesh = outflow_v0 if outflow_v0 is not None else outflow_v1

    inflow = np.load("outflow.npy", allow_pickle=True)
    outflow = np.load("inflow.npy", allow_pickle=True)

    inlet_mesh = surf_mesh.extract_cells(inflow).extract_surface()
    outlet_mesh = surf_mesh.extract_cells(outflow).extract_surface()

    # # Esporta le mesh di inlet e outlet come numpy array
    # if inlet_mesh is not None:
    #     in_points, in_faces = pv_to_np(inlet_mesh)
    #     np.save("inlet_points.npy", in_points)
    #     np.save("inlet_faces.npy", in_faces)
    # if outlet_mesh is not None:
    #     out_points, out_faces = pv_to_np(outlet_mesh)
    #     np.save("outlet_points.npy", out_points)
    #     np.save("outlet_faces.npy", out_faces)

    # in_faces = np.hstack([np.full((in_faces.shape[0],1),3),in_faces])
    # out_faces = np.hstack([np.full((out_faces.shape[0],1),3),out_faces])

    # inlet_mesh = surf_mesh.extract_cells(in_faces)
    # outlet_mesh = surf_mesh.extract_cells(out_faces)


    if inlet_mesh is not None and outlet_mesh is not None:
        print("Inlet e outlet trovati!")
        # Crea il plotter
        plotter = pv.Plotter()

        # Aggiungi la mesh completa (superficie) in grigio semi-trasparente
        plotter.add_mesh(surf_mesh, color='lightgray', opacity=0.3, label='Surface')

        # Aggiungi inlet e outlet se esistono
        if inlet_mesh is not None:
            plotter.add_mesh(inlet_mesh, color='red', label='Inlet')

        if outlet_mesh is not None:
            plotter.add_mesh(outlet_mesh, color='blue', label='Outlet')

        # Aggiungi legenda e mostra il plot
        plotter.add_legend()
        plotter.show()
    else:
        print("Non è stato possibile identificare inlet/outlet automaticamente.")

    # Open file dialog to select fine mesh file
    file_path = filedialog.askopenfilename(
        title="Select mesh.vtu file"
    )
    mesh_new = pv.read(file_path)  # o la tua mesh PyVista
    surf_mesh_new = mesh_new.extract_geometry()
    surf_mesh_new.save("mesh-complete.exterior.vtp")


    inlet_new = extract_surface_region_from_old(surf_mesh_new, inlet_mesh, distance_threshold=0.2)
    outlet_new = extract_surface_region_from_old(surf_mesh_new, outlet_mesh, distance_threshold=0.2)

    pv.global_theme.allow_empty_mesh = True
    p = pv.Plotter()
    p.add_mesh(surf_mesh_new, color="lightgray", opacity=0.3, show_edges=True, label='Surface New')
    p.add_mesh(inlet_new, color="red", opacity=0.8, label='Inlet New', show_edges=True)
    p.add_mesh(outlet_new, color="blue", opacity=0.8, label='Outlet New', show_edges=True)
    p.add_legend()
    p.show()


    print("Script completed successfully.")