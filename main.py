# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410

import sys

import petsc4py

petsc4py.init(sys.argv)
import numpy as np
import stokes_flow as sf
from petsc4py import PETSc
from time import time
from sf_error import sf_error


# from memory_profiler import profile


# @profile
def standard_method():
    OptDB = PETSc.Options()
    deltaLength = OptDB.getReal('d', 0.5)
    epsilon = OptDB.getReal('e', 0.25)
    u = OptDB.getReal('u', 1)
    filename = OptDB.getString('f', 'sphere_2')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    debug_mode = OptDB.getBool('debug', False)
    u_multiple = OptDB.getInt('m', 1)
    matrix_method = OptDB.getString('sm', 'rs')

    if u_multiple != 1 and solve_method != 'lsqr':
        ierr = 51
        err_msg = 'Only lsqr method is avalable when the number of velocity nodes is greater than force nodes. '
        raise sf_error(ierr, err_msg)
    problem_dic = {'rs': sf.stokesFlowProblem,
                   'sf': sf.stokesFlowProblem,
                   'sf_debug': sf.stokesFlowProblem,
                   'ps': sf.stokesFlowProblem, }
    obj_dic = {'rs': sf.stokesFlowObject,
               'sf': sf.surf_forceObj,
               'sf_debug': sf.surf_forceObj,
               'ps': sf.pointSourceObj, }

    n_obj = OptDB.getInt('n', 1)
    n_obj_x = OptDB.getInt('nx', n_obj)
    n_obj_y = OptDB.getInt('ny', n_obj)
    distance = OptDB.getReal('dist', 3)
    distance_x = OptDB.getReal('distx', distance)
    distance_y = OptDB.getReal('disty', distance)
    move_delta = np.array([distance_x, distance_y, 1])

    field_range = np.array([[-3, -3, -3], [n_obj_x - 1, n_obj_y - 1, 0] * move_delta + [3, 3, 3]])
    n_grid = np.array([n_obj_x, n_obj_y, 1]) * 20
    problem_arguments = {'method': matrix_method,
                         'delta': deltaLength * epsilon}

    # create problem
    t0 = time()
    problem = problem_dic[matrix_method]()
    obj1 = obj_dic[matrix_method](filename=filename + '.mat')
    obj_list = [obj1]
    problem.add_obj(obj1)
    for i in range(1, n_obj_x * n_obj_y):
        ix = i // n_obj_x
        iy = i % n_obj_x
        move_dist = np.array([ix, iy, 0]) * move_delta
        obj2 = obj_list[0].copy()
        obj_list.append(obj2)
        obj2.move(move_dist)
        problem.add_obj(obj2)
    problem.create_matrix(**problem_arguments)
    n_nodes = problem.get_n_u_nodes()
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print('n_obj_x: %d, n_obj_y, %d'
              % (n_obj_x, n_obj_x))
        print('move_x: %f, move_y: %f'
              % (distance_x, distance_y))
        print('delta: %f, number of nodes: %d' % (deltaLength * epsilon, n_nodes))
        print('solve method: %s, precondition method: %s' % (solve_method, precondition_method))
        print('output path: ' + filename)
        print('MPI size: %d' % size)
    t1 = time()
    if rank == 0:
        print('create matrix use: %f (s)' % (t1 - t0))

    # problem.saveM(filename + 'M_rs_petsc')
    t0 = time()
    problem.solve(solve_method, precondition_method)
    t1 = time()
    if rank == 0:
        print('solve matrix equation use: %f (s)' % (t1 - t0))

    t0 = time()
    problem.vtk_force('%sForce_%2d_%2d' % (filename, n_obj_x, n_obj_y))
    t1 = time()
    if rank == 0:
        print('write force file use: %f (s)' % (t1 - t0))

    if not debug_mode:
        t0 = time()
        problem.vtk_velocity_rectangle('%sVelocity_%2d_%2d' % (filename, n_obj_x, n_obj_y), field_range, n_grid)
        t1 = time()
        if rank == 0:
            print('write velocity file use: %f (s)' % (t1 - t0))

    if rank == 0:
        for i0, obj0 in enumerate(problem.get_obj_list()):
            force_sphere = obj0.get_force()
            print('---->>>Resultant of %s at x axis is %f'
                  % (obj0.get_obj_name(), np.sum(force_sphere[::3])))

    pass


if __name__ == '__main__':
    standard_method()
