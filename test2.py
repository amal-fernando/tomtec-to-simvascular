import scipy.io
import pyvista as pv
import numpy as np
import gmsh
import meshio
import math
import os
import sys
import trimesh
from numpy.ma.core import append

# Load the .mat file of sparse matrix
data = scipy.io.loadmat('v1.mat')
v = data['v']
data = scipy.io.loadmat('f1.mat')
f = data['f']-1

def compute_triangle_metrics(A, B, C):
    # Compute the edge lengths
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    # Compute the semi-perimeter and area using Heron's formula
    s = (a + b + c) / 2.0
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))  # ensure non-negative

    # Compute inradius and circumradius
    if s > 0:
        inradius = area / s
    else:
        inradius = 0
    if area > 0:
        circumradius = (a * b * c) / (4 * area)
    else:
        circumradius = np.inf

    radius_ratio = 2*inradius / circumradius if circumradius > 0 else 0

    # Compute internal angles using the law of cosines
    def safe_arccos(x):
        return np.arccos(np.clip(x, -1.0, 1.0))

    angle_A = safe_arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    angle_B = safe_arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
    angle_C = safe_arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

    # Convert angles to degrees
    angle_A_deg = np.degrees(angle_A)
    angle_B_deg = np.degrees(angle_B)
    angle_C_deg = np.degrees(angle_C)

    min_angle = min(angle_A_deg, angle_B_deg, angle_C_deg)
    max_angle = max(angle_A_deg, angle_B_deg, angle_C_deg)

    # Skewness: maximum deviation from the ideal angle (60 degrees)
    skewness = max(abs(angle_A_deg - 60), abs(angle_B_deg - 60), abs(angle_C_deg - 60))

    # Compute altitudes for aspect ratio calculation
    # Altitude corresponding to side a is (2*area / a) (and similarly for b and c)
    h_a = 2 * area / a if a > 0 else 0
    h_b = 2 * area / b if b > 0 else 0
    h_c = 2 * area / c if c > 0 else 0
    min_altitude = min(h_a, h_b, h_c) if (h_a and h_b and h_c) else 0
    longest_edge = max(a, b, c)
    aspect_ratio = (2*min_altitude) / (np.sqrt(3)*longest_edge) if min_altitude > 0 else np.inf

    a, b, c = round(float(a), 3), round(float(b), 3), round(float(c), 3)

    return {
        'radius_ratio': radius_ratio,  # 1 is ideal
        'min_angle': min_angle,
        'max_angle': max_angle,
        'skewness': skewness,  # deviation from 60°
        'aspect_ratio': aspect_ratio,
        'area': area,
        'side_lengths': (a, b, c)
    }


def evaluate_mesh(vertices, triangles):
    metrics_list = []

    radius_ratios = []
    aspect_ratios = []
    min_angles = []
    max_angles = []
    skewness_values = []
    areas = []
    side_lengths = []

    for i, triangle in enumerate(triangles):
        A, B, C = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        metrics = compute_triangle_metrics(A, B, C)
        metrics_list.append(metrics)

        # Store values for global analysis
        radius_ratios.append(metrics['radius_ratio'])
        aspect_ratios.append(metrics['aspect_ratio'])
        min_angles.append(metrics['min_angle'])
        max_angles.append(metrics['max_angle'])
        skewness_values.append(metrics['skewness'])
        areas.append(metrics['area'])
        side_lengths.append(metrics['side_lengths'])

    # Compute global statistics
    global_metrics = {
        "avg_radius_ratio": np.mean(radius_ratios),
        "min_radius_ratio": np.min(radius_ratios),
        "max_radius_ratio": np.max(radius_ratios),

        "avg_aspect_ratio": np.mean(aspect_ratios),
        "min_aspect_ratio": np.min(aspect_ratios),
        "max_aspect_ratio": np.max(aspect_ratios),

        "avg_min_angle": np.mean(min_angles),
        "min_min_angle": np.min(min_angles),
        "max_min_angle": np.max(min_angles),

        "avg_max_angle": np.mean(max_angles),
        "min_max_angle": np.min(max_angles),
        "max_max_angle": np.max(max_angles),

        "avg_skewness": np.mean(skewness_values),
        "max_skewness": np.max(skewness_values),

        "avg_area": np.mean(areas),
        "min_area": np.min(areas),
        "max_area": np.max(areas),

        "avg_side_length": np.mean(side_lengths),
    }

    return metrics_list, global_metrics

vertices = v[:, :, 0]
triangles = f

metrics_list, global_metrics = evaluate_mesh(vertices, triangles)

print("=== Global Mesh Metrics ===")
for key, value in global_metrics.items():
    print(f"{key}: {value:.3f}")

print("Recommended mesh quality thresholds:")
print("  Radius Ratio: 1")
print("  Aspect Ratio: 1")
print("  Skewness: 0°")
print("  Min Angle: 60°")
print("  Max Angle: 60°")
print("---------------------------------------------------\n")


'''
print("\n=== Per-Triangle Metrics ===")
for i, metrics in enumerate(metrics_list):
    print(f"Triangle {i}:")
    print(f"  Quality Ratio: {metrics['radius_ratio']:.3f}, 1 is ideal")
    print(f"  Min Angle: {metrics['min_angle']:.2f}°")
    print(f"  Max Angle: {metrics['max_angle']:.2f}°")
    print(f"  Skewness: {metrics['skewness']:.2f}°, deviation from 60°")
    print(f"  Aspect Ratio: {metrics['aspect_ratio']:.2f}")
    print(f"  Area: {metrics['area']:.3f}")
    print(f"  Side Lengths: {metrics['side_lengths']}\n")
    print("---------------------------------------------------\n")
'''

if global_metrics['avg_aspect_ratio'] < 0.9 or global_metrics['avg_skewness'] < 50:
    print("\nThe mesh quality is poor.")
    print("Remeshing should be done.")

else:
    print("\nThe mesh quality is good.")
    print("No need to remesh.")

# Remeshing the mesh

# Ensure vertices are float64
vertices = np.array(vertices, dtype=np.float64)

# Ensure triangles are int32 (not uint16 or uint8)
faces = np.array(triangles, dtype=np.int32)  # Explicit conversion

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

    # Create a volume from all the surfaces
    s = gmsh.model.getEntities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])

    gmsh.model.geo.synchronize()

    # We specify element sizes imposed by a size field, just because we can :-)
    f = gmsh.model.mesh.field.add("MathEval")
    if gmsh.onelab.getNumber('Parameters/Apply funny mesh size field?')[0]:
        gmsh.model.mesh.field.setString(f, "F", "2*Sin((x+y)/5) + 3")
    else:
        gmsh.model.mesh.field.setString(f, "F", "1")
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize()
    gmsh.write('remesh.vtk')

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

# Load your mesh file
mesh_file = "remesh.vtk"  # Replace with your actual file
grid = pv.read(mesh_file)

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


# Extract node coordinates
# Extract points (node coordinates)
points = np.array(grid.points)
print("Points:\n", points)


# Filter the cells for Tetrahedra (type 10) and Triangles (type 5)
tetra_cells = grid.cells_dict[10]  # Tetrahedra (volume)
triangle_cells = grid.cells_dict[5]  # Triangles (surface)


# Print results
print("Tetrahedron Connectivity:\n", tetra_cells)
print("Triangle Connectivity:\n", triangle_cells)


print('Done')


#
# gmsh.initialize()
#
# gmsh.clear()
#
# gmsh.option.setNumber("General.Terminal", 1) # Suppress terminal output
#
# gmsh.merge("mesh.stl")  # Load the mesh directly
#
# # gmsh.fltk.run()  # Opens the GUI to visualize the new mesh
#
# gmsh.model.mesh.classifySurfaces(15*np.pi/180, True, True, 180) # Angle thresholdfor surface reconstruction
#
# gmsh.model.mesh.createGeometry()
# gmsh.model.geo.synchronize()
#
# # # Get all surfaces
# # surfaces = gmsh.model.getEntities(dim=2)
# #
# # for dim, surf_tag in surfaces:
# #     # Get a point on the surface to reparametrize (choose the first node)
# #     nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(dim=2, tag=surf_tag)
# #
# #     if len(nodeCoords) >= 3:  # Ensure there is at least one point
# #         x, y, z = nodeCoords[:3]  # Take the first node coordinates
# #         parametricCoord = gmsh.model.getParametrization(dim, surf_tag, [x, y, z])
# #
# #         # Reparametrize on the surface
# #         gmsh.model.reparametrizeOnSurface(dim, surf_tag, parametricCoord, surf_tag)
# #
# # # gmsh.model.occ.synchronize()  # Ensure OCC sees the created geometry
# # #
# # # # Step 2: Extract surfaces
# # # surfaces = gmsh.model.getEntities(2)  # Get all surfaces (dim=2)
# # #
# # # # Step 3: Perform Boolean Union (fuse all surfaces together)
# # # if len(surfaces) > 1:
# # #     union_result, _ = gmsh.model.occ.fuse(surfaces, [])
# # #     gmsh.model.occ.synchronize()
# #
# # gmsh.fltk.run()  # Opens the GUI to visualize the new mesh
# #
#
# # Set mesh options (adjust values based on your needs)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)  # Minimum element size
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)  # Maximum element size
# gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay-based remeshing
# gmsh.option.setNumber("Mesh.Smoothing", 0)  # Apply 2 smoothing iterations
# gmsh.option.setNumber("Mesh.Optimize", 1)  # Enable optimization
# gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)  # Optimization threshold (0-1)
#
# # Generate the new mesh
# gmsh.model.mesh.generate(2)
#
# # Save the remeshed output
# gmsh.write("remeshed_mesh.stl")
#
# gmsh.fltk.run()  # Opens the GUI to visualize the new mesh
# gmsh.finalize()  # Clean up
#
# # Load the remeshed mesh
# mesh = pv.read("remeshed_mesh.stl")
#
# plotter = pv.Plotter()
# plotter.add_mesh(mesh, color="tan", show_edges=True)
# plotter.show()
#
# points = np.array(mesh.points)
# faces = mesh.faces.reshape(-1, 4)[:,1:]
#
# metrics_list, global_metrics = evaluate_mesh(points, faces)
#
# print("=== Global Mesh Metrics ===")
# for key, value in global_metrics.items():
#     print(f"{key}: {value:.3f}")
#
#
# print('Done')
