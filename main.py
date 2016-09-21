# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410

import petsc4py, sys
petsc4py.init(sys.argv)
import numpy as np
import stokes_flow as sf
from petsc4py import PETSc
from time import time
# from memory_profiler import profile


# @profile
def my_func():
    # create problem
    solve_method = 'gmres'
    precondition_method = 'none'
    # field_range = np.array([[-1, -1, -1], [1, 1, 3]])
    # n_grid = np.array([30, 30, 90])
    # delta = 6.e-4
    # filename = "./helixInfor"
    # n_obj = 1
    # field_range = np.array([[-15, -15, -18], [15, 15, 30]])
    # n_grid = np.array([30, 30, 90])
    # delta = 6.e-4
    # filename = "./bacteria"
    # n_obj = 1

    OptDB = PETSc.Options()
    filename = OptDB.getString('f', "./sphere")
    n_obj = OptDB.getInt('n', 1)
    n_obj_x = OptDB.getInt('nx', n_obj)
    n_obj_y = OptDB.getInt('ny', n_obj)
    move = OptDB.getReal('m', 3)
    move_x = OptDB.getReal('m', move)
    move_y = OptDB.getReal('m', move)
    move_delta = np.array([move_x, move_y, 1])
    delta = OptDB.getReal('delta', 3.e-2)
    field_range = np.array([[-3, -3, -3], [n_obj_x-1, n_obj_y-1, 0]*move_delta+[3, 3, 3]])
    n_grid = np.array([n_obj_x, n_obj_y, 1]) * 30
    # problem_arguments = {'method': 'rs',
    #                      'delta': delta}
    problem_arguments = {'method': 'rs_petsc',
                         'delta': delta}
    # problem_arguments = {'method': 'sf'}

    t0 = time()
    problem = sf.StokesFlowComponent()
    obj1 = sf.StokesFlowObject(filename=filename + '.mat')
    obj_list = [obj1]
    problem.add_obj(obj1)
    for i in range(1, n_obj_x * n_obj_y):
        ix = i // n_obj_x
        iy = i % n_obj_x
        move_dist = np.array([ix, iy, 0]) * move_delta
        # comm = PETSc.COMM_WORLD.tompi4py()
        # rank = comm.Get_rank()
        # if rank == 0:
        #     print(move_dist)
        obj2 = obj_list[0].copy()
        obj_list.append(obj2)
        obj2.move(move_dist)
        problem.add_obj(obj2)

    problem.create_matrix(**problem_arguments)
    t1 = time()
    print('create matrix use: %f (s)'%(t1-t0))

    # problem.saveM(filename + 'M_rs_petsc')
    t0 = time()
    problem.solve(solve_method, precondition_method)
    t1 = time()
    print('solve matrix equation use: %f (s)'%(t1-t0))

    t0 = time()
    problem.vtk_force('%sForce_%2d_%2d'%(filename, n_obj_x, n_obj_y))
    t1 = time()
    print('write force file use: %f (s)'%(t1-t0))

    t0 = time()
    problem.vtk_velocity('%sVelocity_%2d_%2d'%(filename, n_obj_x, n_obj_y), field_range, n_grid)
    t1 = time()
    print('write velocity file use: %f (s)'%(t1-t0))
    pass

if __name__ == '__main__':
  my_func()