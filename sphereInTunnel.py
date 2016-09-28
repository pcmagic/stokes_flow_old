# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410

import petsc4py, sys
petsc4py.init(sys.argv)
import numpy as np
import stokes_flow as sf
from petsc4py import PETSc
import geo
from time import time
from sf_error import sf_error
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
from memory_profiler import profile


# @profile
def standard_method():
    # create problem

    OptDB = PETSc.Options()
    length = OptDB.getReal('l', 2)
    radius = OptDB.getReal('r', 1)
    aRinv = OptDB.getReal('a', 2)
    deltaLength = OptDB.getReal('d', 0.5)
    epsilon = OptDB.getReal('e', 0.25)
    u = OptDB.getReal('u', 1)
    filename = OptDB.getString('f', 'sphereInTunnel')
    solve_method = OptDB.getString('s', 'lsqr')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
    u_multiple = OptDB.getInt('m', 1)
    if u_multiple != 1 and solve_method != 'lsqr':
        ierr = 51
        err_msg = 'Only lsqr method is avalable when the number of velocity nodes is greater than force nodes. '
        raise sf_error(ierr, err_msg)


    deltaLength0 = deltaLength
    deltaLength = deltaLength / np.sqrt(u_multiple)
    u_nSphere = int(16*radius*radius/deltaLength/deltaLength)
    u_nodesSphere, uSphere, normSphere = geo.sphere(u_nSphere, radius, radius, [u, 0, 0, 0, 0, 0])
    u_nodesTunnel, uTunnel, normTunnel = geo.tunnel(deltaLength, length, radius*aRinv)
    f_nodesSphere = u_nodesSphere[::u_multiple, :]
    f_nodesTunnel = u_nodesTunnel[::u_multiple, :]
    f_nSphere = f_nodesSphere.shape[0]
    f_nTunnel = f_nodesTunnel.shape[0]
    field_range = np.array([[-length/2, radius, -radius], [length/2, radius*aRinv, radius]])
    n_grid = np.array([length, (aRinv-1), 1], dtype='int') * 2
    problem_arguments = {'method': 'rs_petsc',
                         'delta': deltaLength*epsilon}

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print('tunnel length: %f, tunnel radius: %f, sphere radius: %f'
              %(length, radius*aRinv, radius))
        print('delta length: %f, epsilon: %f, velocity: %f, multiple coefficient: %d'
              %(deltaLength0, epsilon, u, u_multiple))
        print('Number of force nodes for tunnel and sphere are %d and %d, respectively.'%(f_nTunnel, f_nSphere))
        print('solve method: %s, precondition method: %s'%(solve_method, precondition_method))
        print('output path: ' + filename)
        print('MPI size: %d'%size)

    # if plot:
    #     f_nodes = np.concatenate((f_nodesSphere, f_nodesTunnel), axis=0)
    #     u_nodes = np.concatenate((u_nodesSphere, u_nodesTunnel), axis=0)
    #     u = np.concatenate((uSphere, uTunnel), axis=0).flatten()
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     ax.scatter(u_nodesSphere[:, 0], u_nodesSphere[:, 1], u_nodesSphere[:, 2], c='b', marker='o')
    #     ax.scatter(f_nodesSphere[:, 0], f_nodesSphere[:, 1], f_nodesSphere[:, 2], c='r', marker='o')
    #     # ax.scatter(u_nodesTunnel[:, 0], u_nodesTunnel[:, 1], u_nodesTunnel[:, 2], c='b', marker='o')
    #     # ax.scatter(f_nodesTunnel[:, 0], f_nodesTunnel[:, 1], f_nodesTunnel[:, 2], c='r', marker='o')
    #     # ax.quiver(u_nodes[:, 0], u_nodes[:, 1], u_nodes[:, 2],
    #     #           u[0::3], u[1::3], u[2::3],
    #     #           color='r', length=deltaLength*2)
    #     ax.axis(v='equal')
    #     plt.show()

    t0 = time()
    problem = sf.StokesFlowProblem()
    obj_sphere = sf.StokesFlowObject()
    obj_sphere.set_data(f_nodesSphere, u_nodesSphere, uSphere.flatten())
    problem.add_obj(obj_sphere)
    obj_tunnel = sf.StokesFlowObject()
    obj_tunnel.set_data(f_nodesTunnel, u_nodesTunnel, uTunnel.flatten())
    problem.add_obj(obj_tunnel)
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

    if rank == 0:
        force_sphere = obj_sphere.get_force()
        print('---->>>Resultant at x axis is %f' % (np.sum(force_sphere[::3])))
    pass

# @profile
def surf_force_method():
    # create problem

    OptDB = PETSc.Options()
    length = OptDB.getReal('l', 2)
    radius = OptDB.getReal('r', 1)
    aRinv = OptDB.getReal('a', 2)
    deltaLength = OptDB.getReal('d', 0.5)
    epsilon = OptDB.getReal('e', 0.25)
    u = OptDB.getReal('u', 1)
    filename = OptDB.getString('f', 'sphereInTunnel')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)

    nSphere = int(16*radius*radius/deltaLength/deltaLength)
    nodesSphere, uSphere, normSphere = geo.sphere(nSphere, radius, radius, [u, 0, 0, 0, 0, 0])
    nodesTunnel, uTunnel, normTunnel = geo.tunnel(deltaLength, length, radius*aRinv)
    nTunnel = nodesTunnel.shape[0]
    field_range = np.array([[-length/2, radius, -radius], [length/2, radius*aRinv, radius]])
    n_grid = np.array([length, (aRinv-1), 1], dtype='int') * 10
    problem_arguments = {'method': 'sf',
                         'd_radia': deltaLength / 2 }

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print('surface force method')
        print('tunnel length: %f, tunnel radius: %f, sphere radius: %f'
              %(length, radius*aRinv, radius))
        print('delta length: %f, epsilon: %f, velocity: %f'
              %(deltaLength, epsilon, u))
        print('Number of nodes for tunnel and sphere are %d and %d, respectively.'%(nTunnel, nSphere))
        print('solve method: %s, precondition method: %s'%(solve_method, precondition_method))
        print('output path: ' + filename)
        print('MPI size: %d'%size)

    # if plot:
    #     nodes = np.concatenate((nodesSphere, nodesTunnel), axis=0)
    #     u = np.concatenate((uSphere, uTunnel), axis=0).flatten()
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
    problem = sf.surf_forceProblem()
    obj_sphere = sf.surf_forceObj()
    obj_sphere.set_data(nodesSphere, nodesSphere, uSphere.flatten(), normSphere)
    problem.add_obj(obj_sphere)
    obj_tunnel = sf.surf_forceObj()
    obj_tunnel.set_data(nodesTunnel, nodesTunnel, uTunnel.flatten(), normTunnel)
    problem.add_obj(obj_tunnel)
    problem.create_matrix(**problem_arguments)
    t1 = time()
    if rank == 0:
        print('create matrix use: %f (s)'%(t1-t0))

    t0 = time()
    problem.saveM(filename + 'M_sf')
    if rank == 0:
        print('save finished. ')
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

    if rank == 0:
        force_sphere = obj_sphere.get_force()
        print('---->>>Resultant at x axis is %f' % (np.sum(force_sphere[::3])))
    pass

if __name__ == '__main__':
    OptDB = PETSc.Options()
    solve_method = OptDB.getString('sm', 'rs')
    if solve_method == 'rs':
        standard_method()
    elif solve_method == 'sf':
        surf_force_method()