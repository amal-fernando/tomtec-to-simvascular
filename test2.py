def gmsh_remesh(mesh, target_size=0.5):
    import meshio
    import gmsh
    import numpy as np
    import pyvista as pv
    import math
    import os

    # Convert pyvista mesh to numpy
    vertices = np.array(mesh.points, dtype=np.float64)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # solo triangoli

    # Save temporary STL file
    meshio.write("mesh.stl", meshio.Mesh(points=vertices, cells=[("triangle", faces)]))

    gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.merge("mesh.stl")

    angle = 40 # min: 15, max: 120
    forceParametrizablePatches = 0 # 0: False, 1: True
    includeBoundary = True
    curveAngle = 180

    gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary,
                                     forceParametrizablePatches,
                                     curveAngle * math.pi / 180.)
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size)
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay-based remeshing
    gmsh.option.setNumber("Mesh.Smoothing", 0)  # Apply 2 smoothing iterations
    gmsh.option.setNumber("Mesh.Optimize", 1)  # Enable optimization
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)  # Optimization threshold (0-1)

    gmsh.model.mesh.generate(2)
    gmsh.write("remesh.stl")
    gmsh.finalize()

    return pv.read("remesh.stl")

'''
def gmsh_remesh(mesh):
    # Remeshing the mesh
    vertices, faces = pv_to_np(mesh)
    # Ensure vertices are float64
    vertices = np.array(vertices, dtype=np.float64)

    # Ensure triangles are int32 (not uint16 or uint8)
    faces = np.array(faces, dtype=np.int32)  # Explicit conversion

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
        #
        # # Create a volume from all the surfaces
        # s = gmsh.model.getEntities(2)
        # l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
        # gmsh.model.geo.addVolume([l])
        #
        # gmsh.model.geo.synchronize()
        #
        # # We specify element sizes imposed by a size field, just because we can :-)
        # f = gmsh.model.mesh.field.add("MathEval")
        # if gmsh.onelab.getNumber('Parameters/Apply funny mesh size field?')[0]:
        #     gmsh.model.mesh.field.setString(f, "F", "2*Sin((x+y)/5) + 3")
        # else:
        #     gmsh.model.mesh.field.setString(f, "F", "1")
        # gmsh.model.mesh.field.setAsBackgroundMesh(f)
        #
        # gmsh.model.mesh.generate(3)
        # gmsh.model.mesh.optimize()
        #
        gmsh.model.geo.synchronize()
        gmsh.write('remesh.stl')


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

    grid = pv.read('remesh.stl')
    
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
    
    return grid
'''