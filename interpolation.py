# Copyright (c) Stanford University, The Regents of the University of
#               California, and others.
#
# All Rights Reserved.
#
# See Copyright-SimVascular.txt for additional details.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
sys.path.append(os.path.join(os.path.dirname(
__file__), "../src"))
import numpy as np
import collections
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import io_utils
import utils
import glob


"""
Functions to write interpolated surface meshes for perscribed wall motion

"""

def cubic_spline_ipl(time, t_m, dt_m, boundary_queue):
    """
    Cubic Hermite spline interpolation for nodes movement
    see https://en.wikipedia.org/wiki/Cubic_Hermite_spline

    Args:
        time: time index in range(num_itpls)+1
        t_m: initial time point
        dt_m: number of iterpolations
        boundary_queue: list of VTK PolyData
    
    Returns:
        coords: coordinates of the interpolated mesh
    """

    boundary0 = vtk_to_numpy(boundary_queue[0].GetPoints().GetData())
    boundary1 = vtk_to_numpy(boundary_queue[1].GetPoints().GetData())
    boundary2 = vtk_to_numpy(boundary_queue[2].GetPoints().GetData())
    boundary3 = vtk_to_numpy(boundary_queue[3].GetPoints().GetData())

    dim = boundary0.shape[-1]

    t_a = (float(time) - t_m)/dt_m
    h_00 = 2*t_a*t_a*t_a - 3*t_a*t_a + 1
    h_10 = t_a*t_a*t_a - 2*t_a*t_a + t_a
    h_01 = - 2*t_a*t_a*t_a + 3*t_a*t_a
    h_11 = t_a*t_a*t_a - t_a*t_a

    v_m = (boundary2-boundary0)/dt_m/2
    v_m1 = (boundary3-boundary1)/dt_m/2
    coords = h_00*boundary1 + h_01*boundary2 + h_10*v_m*dt_m + h_11*v_m1*dt_m
    return coords

def find_index_in_array(x, y):
    """
    For x being a list containing y, find the index of each element of y in x
    """
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    indices = xsorted[ypos]
    return indices

def move_mesh(fns, start_point, intpl_num, num_cycle):
    total_num_phase = len(fns)
    total_steps = total_num_phase * (intpl_num+1)*num_cycle
    initialized = False
    poly_template = io_utils.read_vtk_mesh(fns[start_point])
    ref_coords = vtk_to_numpy(poly_template.GetPoints().GetData())
    store = np.zeros((poly_template.GetNumberOfPoints(), 3, total_steps+1)) 
    count = 0
    # First cycle
    for msh_idx in list(range(start_point, total_num_phase))+ list(range(0, start_point)):
        if not initialized:
            boundary_queue = collections.deque(4*[None], 4)
            boundary_queue.append(io_utils.read_vtk_mesh(fns[(msh_idx+total_num_phase-1)%total_num_phase]))
            boundary_queue.append(io_utils.read_vtk_mesh(fns[msh_idx]))
            boundary_queue.append(io_utils.read_vtk_mesh(fns[(msh_idx+1)%total_num_phase]))
            boundary_queue.append(io_utils.read_vtk_mesh(fns[(msh_idx+2)%total_num_phase]))
            initialized = True
        else:
            boundary_queue.append(io_utils.read_vtk_mesh(fns[(msh_idx+2)%total_num_phase]))

        for i_idx in range(intpl_num+1):
            new_coords = cubic_spline_ipl(i_idx, 0, intpl_num+1, boundary_queue)
            displacement = new_coords- ref_coords
            store[:, :, count] = displacement
            count+=1
    # The rest cycles are copies of first cycle
    for c in range(1,num_cycle):
        s = c*total_num_phase * (intpl_num+1)
        e = s + total_num_phase * (intpl_num+1)
        store[:,:,s:e] = store[:, :,0:count]

    return store

def write_motion(mesh_dir,  start_point, intpl_num, output_dir, num_cycle, duration, debug=False, mode='displacement'):
    fns = utils.natural_sort(glob.glob(os.path.join(mesh_dir, "*.vtp")))
    total_num_phase = len(fns)
    total_steps = num_cycle* total_num_phase * (intpl_num+1)+1
    initialized = False
    time_pts = np.linspace(0,num_cycle*duration, total_steps)
    
    poly_template = io_utils.read_vtk_mesh(fns[start_point])
    
    displacements = move_mesh(fns, start_point, intpl_num, num_cycle)
    if debug:
        import vtk
        debug_dir = os.path.join(output_dir,"Debug")
        try:
            os.makedirs(debug_dir)
        except Exception as e: print(e)
        coords = vtk_to_numpy(poly_template.GetPoints().GetData())
        poly = vtk.vtkPolyData()
        poly.DeepCopy(poly_template)
        for ii in range(displacements.shape[-1]):
            poly.GetPoints().SetData(numpy_to_vtk(displacements[:,:,ii]+coords))
            fn_debug = os.path.join(debug_dir, "debug%05d.vtp" %ii)
            io_utils.write_vtk_polydata(poly, fn_debug)

    node_ids = vtk_to_numpy(poly_template.GetPointData().GetArray('GlobalNodeID'))
    face_ids = vtk_to_numpy(poly_template.GetCellData().GetArray('ModelFaceID'))
    #write time steps and node numbers
    for face in np.unique(face_ids):
        if mode=='displacement':
            fn = os.path.join(output_dir, '%d_displacement.dat' % face)
        elif mode=='velocity':
            fn = os.path.join(output_dir, '%d_velocity.dat' % face)
        else:
            raise ValueError('Unsupported boundary type {}; should be displacement or velocity.'.format(mode))
        face_poly = utils.threshold_polydata(poly_template, 'ModelFaceID', (face,face))
        f = open(fn, 'w')
        f.write('{} {} {}\n'.format(3, total_steps,face_poly.GetNumberOfPoints()))
        for t in time_pts:
            f.write('{}\n'.format(t))
        #f.write('{}\n'.format(face_poly.GetNumberOfPoints()))
        face_ids = vtk_to_numpy(face_poly.GetPointData().GetArray('GlobalNodeID'))
        node_id_index = find_index_in_array(node_ids, face_ids)
        for i in node_id_index:
            disp = displacements[i, :, :]
            f.write('{}\n'.format(node_ids[i]))
            for j in range(total_steps):
                if mode=='displacement':
                    f.write('{} {} {}\n'.format(disp[0,j], disp[1,j],disp[2,j]))
                elif mode=='velocity':
                    f.write('{} {} {}\n'.format(disp[0,j]/(time_pts[1]-time_pts[0]), disp[1,j]/(time_pts[1]-time_pts[0]),disp[2,j]/(time_pts[1]-time_pts[0])))
        f.close()




if __name__=='__main__':
    import time
    import argparse
    start = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', help="Path to the surface meshes")
    parser.add_argument('--output_dir', help="Path to the volume meshes")
    parser.add_argument('--num_interpolation', type=int, help="Number of interpolations")
    parser.add_argument('--num_cycle', type=int, help="Number of cardiac cycles")
    parser.add_argument('--duration', type=float, help="Cycle duration in seconds")
    parser.add_argument('--phase', default=-1, type=int, help="Id of the phase to generate volume mesh")
    parser.add_argument('--boundary_type', default='displacement', help='Type of the boundary condition, displacement or velocity')
    args = parser.parse_args()
    output_dir = os.path.join(args.output_dir, 'mesh-complete')
    try:
       os.makedirs(output_dir)
    except Exception as e: print(e)

    write_motion(args.input_dir,  args.phase ,args.num_interpolation, output_dir, args.num_cycle, args.duration, debug=False, mode=args.boundary_type)
    end = time.time()
    print("Time spent: ", end-start)
