# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410

import petsc4py, sys
petsc4py.init(sys.argv)
import numpy as np
import stokes_flow as sf
from petsc4py import PETSc
import geo
from time import time
# import os
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
from mpi4py import MPI
from memory_profiler import profile


@profile
def my_func():
    # create problem

    OptDB = PETSc.Options()
    length = OptDB.getReal('l', 2)
    radius = OptDB.getReal('r', 1)
    aRinv = OptDB.getReal('a', 2)
    deltaLength = OptDB.getReal('d', 0.1)
    epsilon = OptDB.getReal('e', 0.25)
    u = OptDB.getReal('u', 1)
    # pathname = OptDB.getString('p', './sphereInTunnel')
    filename = OptDB.getString('f', 'sphereInTunnel')
    # if pathname[0:2] != './':
    #     pathname = './' + pathname
    # if pathname[:-1] == '/':
    #     pathname = pathname[:-1]
    # if not os.path.exists(pathname):
    #     print('create path: ' + pathname)
    #     os.makedirs(pathname)
    # pathname = pathname + '/'
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plt = OptDB.getBool('plot', False)

    nSphere = int(16*radius*radius/deltaLength/deltaLength)
    nodesSphere, uSphere = geo.sphere(nSphere, radius, radius, [u, 0, 0, 0, 0, 0])
    nodesTunnel, uTunnel = geo.tunnel(deltaLength, length, radius*aRinv)
    nTunnel = nodesTunnel.shape[0]
    field_range = np.array([[-radius*0.1, radius, -radius], [radius*0.1, radius*aRinv, radius]])
    n_grid = np.array([5, (aRinv-1)*20, 20], dtype='int')
    problem_arguments = {'method': 'rs_petsc',
                         'delta': deltaLength*epsilon}

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print('tunnel length: %f, tunnel radius: %f, sphere radius: %f'
              %(length, radius*aRinv, radius))
        print('delta length: %f, epsilon: %f, velocity: %f'
              %(deltaLength, epsilon, u))
        print('Number of nodes for tunnel and sphere are %d and %d, respectively.'%(nTunnel, nSphere))
        print('solve method: %s, precondition method: %s'%(solve_method, precondition_method))
        print('output path: ' + filename)
        print('MPI size: %d'%size)

    nodes = np.concatenate((nodesSphere, nodesTunnel), axis=0)
    u = np.concatenate((uSphere, uTunnel), axis=0).flatten()

    # if plt:
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     ax.scatter(nodesSphere[:, 0], nodesSphere[:, 1], nodesSphere[:, 2], c='r', marker='o')
    #     ax.scatter(nodesTunnel[:, 0], nodesTunnel[:, 1], nodesTunnel[:, 2], c='b', marker='o')
    #     ax.quiver(nodes[:, 0], nodes[:, 1], nodes[:, 2],
    #               u[:, 0], u[:, 1], u[:, 2],
    #               color='r', length=deltaLength*2)
    #     ax.axis(v='equal')
    #     plt.show()

    t0 = time()
    problem = sf.StokesFlowComponent()
    obj1 = sf.StokesFlowObject()
    obj1.set_data(nodes, u)
    problem.add_obj(obj1)
    problem.create_matrix(**problem_arguments)
    t1 = time()
    if rank == 0:
        print('create matrix use: %f (s)'%(t1-t0))

    t0 = time()
    # problem.saveM(filename + 'M_rs_petsc')
    problem.solve(solve_method, precondition_method)
    t1 = time()
    if rank == 0:
        print('solve matrix equation use: %f (s)'%(t1-t0))

    t0 = time()
    problem.vtk_force(filename + 'Force')
    t1 = time()
    if rank == 0:
        print('write force file use: %f (s)'%(t1-t0))

    t0 = time()
    problem.vtk_velocity(filename + 'velocity', field_range, n_grid)
    t1 = time()
    if rank == 0:
        print('write velocity file use: %f (s)'%(t1-t0))
    pass


if __name__ == '__main__':
  my_func()