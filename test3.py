import os
import tetgen
from pyvista import CellType
import numpy as np
import pyvista as pv
import xml.etree.ElementTree as ET

# Save the interpolated coordinates to a .vtu file
# Define input and output folders
output_dir = "output_volume_meshes/"
os.makedirs(output_dir, exist_ok=True)

# Load your data
# vertices = matrixFourier # np.load("vertices.npy")  # Shape: Nx3xT
# connectivity = f2 # np.load("connectivity.npy")  # Shape: Mx3

# Load the remeshed mesh
mesh = pv.read("remeshed_mesh.stl")

v2 = np.array(mesh.points)
connectivity = mesh.faces.reshape(-1, 4)[:,1:]  # Remove first column (number of vertices)
vertices = v2[:, :, np.newaxis]

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
        tet.tetrahedralize(tetgen_options) # order=1, mindihedral=20, minratio=1.5)
    except RuntimeError as e:
        print(f"Failed to tetrahedralize: {e}")
        continue

    grid = tet.grid
    grid.plot(show_edges=True)

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
    plotter.add_mesh(surface_mesh, 'r', 'wireframe')
    plotter.add_legend([[' Input Mesh ', 'r'],
                        [' Tessellated Mesh ', 'black']])
    plotter.show()

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