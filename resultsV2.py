import numpy as np
import pyvista as pv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import time

def process_file(i, radice_dataset, multiplier, path_str, inletIDs, outletIDs, boundary_result_path_str):
    import pyvista as pv
    from pathlib import Path
    import numpy as np

    path = Path(path_str)
    boundary_result_path = Path(boundary_result_path_str)
    current_file = path / f"{radice_dataset}{i * multiplier:03d}.vtu"

    if not current_file.exists():
        return f"Warning: {current_file.name} not found. Skipping."

    grid = pv.read(current_file)
    grid.point_data["GlobalNodeID"] = np.arange(grid.points.shape[0]) + 1

    surf = grid.extract_surface()
    node_ids = surf.point_data["GlobalNodeID"]

    inlet_result = surf.extract_points(np.isin(node_ids, inletIDs), adjacent_cells=False).extract_surface()
    outlet_result = surf.extract_points(np.isin(node_ids, outletIDs), adjacent_cells=False).extract_surface()

    inlet_result.save(boundary_result_path / f"{radice_dataset}inlet_{i * multiplier:03d}.vtp")
    outlet_result.save(boundary_result_path / f"{radice_dataset}outlet_{i * multiplier:03d}.vtp")
    return f"Processed timestep {i}"

if __name__ == "__main__":
    print("Script started")
    start_time = time.time()  # ⏱️

    # Selezione file iniziale
    tk.Tk().withdraw()
    file_path = Path(filedialog.askopenfilename(title="Select first result_xxx.vtu file"))
    if not file_path.exists():
        raise FileNotFoundError("No file selected or file does not exist.")

    path = file_path.parent
    base_name = file_path.name
    radice_dataset = base_name[:-7]
    multiplier = int(base_name[len(radice_dataset):-4])
    boundary_result_path = path / "boundary_result"
    boundary_result_path.mkdir(exist_ok=True)

    # Conteggio file
    n_files = len([f for f in path.iterdir() if f.name.startswith(radice_dataset) and f.suffix == '.vtu'])

    # Caricamento maschere
    mesh_path = path.parent / "mesh" / "mesh-surfaces"
    inletIDs = np.array(pv.read(mesh_path / "inlet.vtp").point_data["GlobalNodeID"])
    outletIDs = np.array(pv.read(mesh_path / "outlet.vtp").point_data["GlobalNodeID"])

    # ⚠️ Gli array devono essere convertiti in tipi serializzabili (list) per multiprocessing
    inletIDs_list = inletIDs.tolist()
    outletIDs_list = outletIDs.tolist()

    # Parallel execution
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_file, i, radice_dataset, multiplier,
                str(path), inletIDs_list, outletIDs_list,
                str(boundary_result_path)
            )
            for i in range(1, n_files + 1)
        ]
        results = [f.result() for f in tqdm(futures, total=n_files)]

    for r in results:
        if r.startswith("Warning"):
            print(r)

    end_time = time.time()  # ⏱️
    print(f"Done in {end_time - start_time:.2f} seconds.")
