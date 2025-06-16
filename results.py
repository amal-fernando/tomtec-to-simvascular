import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pyvista as pv
import os
import tqdm.auto as tqdm


print("Script started")
# Initialize tkinter but hide main window
root = tk.Tk()
root.withdraw()

# Open file dialog to select header file
file_path = filedialog.askopenfilename(
    title="Select first result_xxx.vtu file"
)

base_name = os.path.basename(file_path)
radice_dataset = base_name[:-7]  # Remove the last 4 characters (e.g., ".vtu")
path= os.path.dirname(file_path)
n_files = len([f for f in os.listdir(path) if f.startswith(radice_dataset) and f.endswith('.vtu')])
multplier = int(base_name[len(radice_dataset):-4])  # Extract the multiplier from the filename

os.makedirs(os.path.join(path, "boundary_result"), exist_ok=True)
# Create a directory for saving results
boundary_result_path = os.path.join(path, "boundary_result")
inlet_path = os.path.join(os.path.dirname(path), "mesh", "mesh-surfaces", "inlet.vtp")
outlet_path = os.path.join(os.path.dirname(path), "mesh", "mesh-surfaces", "outlet.vtp")

# Initialize inlet and outlet masks (node IDs)
inlet = pv.read(inlet_path)
outlet = pv.read(outlet_path)
inletIDs = np.array(inlet.point_data["GlobalNodeID"])
outletIDs = outlet.point_data["GlobalNodeID"]

for i in range(1, n_files + 1):
    file_name = f"{radice_dataset}{i*multplier:03d}.vtu"
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist. Skipping.")
        continue

    # Read the grid
    grid = pv.read(full_path)

    grid.point_data["GlobalNodeID"] = np.arange(grid.points.shape[0]) + 1
    # grid.cell_data["GlobalElementID"] = np.arange(grid.n_cells) + 1

    # Create mask for inlet and outlet points
    surf = grid.extract_surface()
    inlet_mask = np.isin(surf.point_data["GlobalNodeID"], inletIDs)
    outlet_mask = np.isin(surf.point_data["GlobalNodeID"], outletIDs)

    # Extract inlet and outlet result points from the surface
    inlet_result = surf.extract_points(inlet_mask, adjacent_cells=False)
    outlet_result = surf.extract_points(outlet_mask, adjacent_cells=False)

    inlet_result = inlet_result.extract_surface()
    outlet_result = outlet_result.extract_surface()

    # Plot the inlet and outlet points
    # p = pv.Plotter()
    # p.add_mesh(surf, color='lightgray', show_edges=True, label='Surface', opacity=0.5)
    # p.add_mesh(inlet_result, color='blue', point_size=5, render_points_as_spheres=True, label='Inlet')
    # p.add_mesh(outlet_result, color='red', point_size=5, render_points_as_spheres=True, label='Outlet')
    # p.add_legend()
    # p.show()

    inlet_result.save(os.path.join(boundary_result_path, f"{radice_dataset}inlet_{i*multplier:03d}.vtp"))
    outlet_result.save(os.path.join(boundary_result_path, f"{radice_dataset}outlet_{i*multplier:03d}.vtp"))

print("Done")