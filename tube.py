import gmsh
import meshio
import pyvista as pv
import imageio
import numpy as np
from test import get_bounds

'''Create a cylinder surface mesh'''

gmsh.initialize()
gmsh.clear()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("cylinder_surface")

# Parameters
R_cyl = 5
H_cyl = 60
lc = 0.005  # Mesh element size

# Create a cylinder surface (no thickness)
gmsh.model.occ.addCylinder(0, 0, -H_cyl/2, 0, 0, H_cyl, R_cyl, tag=1)
gmsh.model.occ.remove([(3,1)])  # Remove volume to keep only the surface
gmsh.model.occ.synchronize()

# Mesh it as a surface (2D elements)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)  # Minimum element size
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)  # Maximum element size
gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay-based remeshing

gmsh.model.mesh.generate(2)
gmsh.fltk.run()
gmsh.option.setNumber("Mesh.Binary", 1)  # imposta output binario
gmsh.write("cylinder_surface.stl")

# # Define physical groups to ensure all elements belong to a set
# gmsh.model.addPhysicalGroup(2, [1], tag=1)  # Tagging the cylinder surface (2D elements)
# gmsh.model.setPhysicalName(2, 1, "CylinderSurface")
#
# gmsh.model.addPhysicalGroup(3, [1], tag=2)  # Tagging the volume (3D elements)
# gmsh.model.setPhysicalName(3, 2, "CylinderVolume")
#
# gmsh.model.mesh.generate(3)
# gmsh.fltk.run()
# gmsh.write("cylinder_volume.msh")

gmsh.finalize()

# Output the inflow, outflow and wall surfaces
# catSurf("cylinder_surface.stl")

# # Read the .msh file
# mesh = meshio.read("cylinder_volume.stl")
#
# # Write to .vtu format
# meshio.write("full_mesh.vtu", mesh)
#
# '''Deform the cylinder surface mesh'''
# print("Deforming the cylinder surface mesh...")
#

