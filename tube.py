import gmsh

def cylinder (R_cyl=5, H_cyl=60, lc=1.0):
    """
    Create a cylinder surface mesh using GMSH and save it as an STL file.

    Parameters:
    R_cyl (float): Cylinder radius.
    H_cyl (float): Cylinder height.
    lc (float): Mesh element size.
    """

    # Initialize GMSH
    gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("cylinder_surface")

    # Create a cylinder surface (no thickness)
    gmsh.model.occ.addCylinder(0, 0, -H_cyl/2, 0, 0, H_cyl, R_cyl, tag=1) # Cylinder with center at (0,0,-H/2) and radius R_cyl
    gmsh.model.occ.remove([(3,1)])  # Remove volume to keep only the surface
    gmsh.model.occ.synchronize() # Synchronize the model to ensure all operations are applied

    # Mesh it as a surface (2D elements)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc*0.1)  # Minimum element size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)  # Maximum element size
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay-based remeshing

    gmsh.model.mesh.generate(2) # Generate the mesh for 2D elements
    # gmsh.fltk.run() # Decomment to open the GMSH GUI to visualize the mesh
    gmsh.option.setNumber("Mesh.Binary", 1)  # Save mesh in binary format
    gmsh.write("cylinder_surface.stl")  # Save the mesh as STL file

    gmsh.finalize()  # Finalize GMSH

    return "cylinder_surface.stl"