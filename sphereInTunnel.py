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
from memory_profiler import profile


# @profile
def main_fun():
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
    debug_mode = OptDB.getBool('debug', False)
    u_multiple = OptDB.getInt('m', 1)
    matrix_method = OptDB.getString('sm', 'rs')

    if u_multiple != 1 and solve_method != 'lsqr':
        ierr = 51
        err_msg = 'Only lsqr method is avalable when the number of velocity nodes is greater than force nodes. '
        raise sf_error(ierr, err_msg)
    problem_dic = {'rs':sf.stokesFlowProblem,
                   'sf':sf.surf_forceProblem}
    obj_dic = {'rs':sf.stokesFlowObject,
               'sf':sf.surf_forceObj}

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
    problem_arguments = {'method': matrix_method,
                         'delta': deltaLength*epsilon,          # for rs method
                         'd_radia': deltaLength / 2}            # for sf method

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print('tunnel length: %f, tunnel radius: %f, sphere radius: %f'
              %(length, radius*aRinv, radius))
        print('delta length: %f, velocity: %f, multiple coefficient: %d'
              %(deltaLength0, u, u_multiple))
        if matrix_method == 'rs':
            print('create matrix method: %s, epsilon: %f'
                  %(matrix_method, epsilon))
        elif matrix_method == 'sf':
            print('create matrix method: %s'%matrix_method)
        else:
            print('please specify how to descript the metrix create method')
        print('Number of force nodes for tunnel and sphere are %d and %d, respectively.'
              %(f_nTunnel, f_nSphere))
        print('solve method: %s, precondition method: %s'
              %(solve_method, precondition_method))
        print('output path: ' + filename)
        print('MPI size: %d'%size)

    if plot:
        # from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        u_nodes = np.concatenate((u_nodesSphere, u_nodesTunnel), axis=0)
        u = np.concatenate((uSphere, uTunnel), axis=0).flatten()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(u_nodesSphere[:, 0], u_nodesSphere[:, 1], u_nodesSphere[:, 2], c='b', marker='o')
        ax.scatter(f_nodesSphere[:, 0], f_nodesSphere[:, 1], f_nodesSphere[:, 2], c='r', marker='o')
        ax.scatter(u_nodesTunnel[:, 0], u_nodesTunnel[:, 1], u_nodesTunnel[:, 2], c='b', marker='o')
        ax.scatter(f_nodesTunnel[:, 0], f_nodesTunnel[:, 1], f_nodesTunnel[:, 2], c='r', marker='o')
        ax.quiver(u_nodes[:, 0], u_nodes[:, 1], u_nodes[:, 2],
                  u[0::3], u[1::3], u[2::3],
                  color='r', length=deltaLength*2)
        ax.axis(v='equal')
        plt.show()

    t0 = time()
    problem = problem_dic[matrix_method]()
    obj_sphere = obj_dic[matrix_method]()
    para = {'norm': normSphere}
    obj_sphere.set_data(f_nodesSphere, u_nodesSphere, uSphere.flatten(), **para)
    problem.add_obj(obj_sphere)
    obj_tunnel = obj_dic[matrix_method]()
    para = {'norm': normTunnel}
    obj_tunnel.set_data(f_nodesTunnel, u_nodesTunnel, uTunnel.flatten(), **para)
    problem.add_obj(obj_tunnel)
    problem.create_matrix(**problem_arguments)
    t1 = time()
    if rank == 0:
        print('create matrix use: %f (s)'%(t1-t0))

    t0 = time()
    # problem.viewM(**problem_arguments)
    M = problem.get_M()
    # problem.saveM(filename + 'M_rs_petsc')
    # if rank == 0:
    #     print('save finished. ')
    problem.solve(solve_method, precondition_method)
    t1 = time()
    if rank == 0:
        print('solve matrix equation use: %f (s)'%(t1-t0))

    t0 = time()
    problem.vtk_force(filename + 'Force')
    t1 = time()
    if rank == 0:
        print('write force file use: %f (s)'%(t1-t0))

    if not debug_mode:
        t0 = time()
        problem.vtk_velocity(filename + 'velocity', field_range, n_grid)
        t1 = time()
        if rank == 0:
            print('write velocity file use: %f (s)'%(t1-t0))

    if rank == 0:
        force_sphere = obj_sphere.get_force()
        print('---->>>Resultant at x axis is %f' % (np.sum(force_sphere[::3])))

    return M


# @profile
def view_matrix(m, **kwargs):
    args = {'vmin': None,
            'vmax': None,
            'title': ' ',
            'cmap': None}
    for key, value in args.items():
        if key in kwargs:
            args[key] = kwargs[key]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cax = ax.matshow(m,
                     origin='lower',
                     vmin=args['vmin'],
                     vmax=args['vmax'],
                     cmap=plt.get_cmap(args['cmap']))
    fig.colorbar(cax)
    plt.title(args['title'])
    plt.show()


if __name__ == '__main__':
    # main_fun()

    OptDB = PETSc.Options()
    OptDB.setValue('sm', 'rs')
    m_rs = main_fun()
    OptDB.setValue('sm', 'sf')
    m_sf = main_fun()
    # delta_m = np.abs(m_rs - m_sf)
    # # view_matrix(np.log10(delta_m), 'rs_m - sf_m')
    # percentage = delta_m / (np.maximum(np.abs(m_rs), np.abs(m_sf)) + 1e-100)
    #
    # view_args = {'vmin': -10,
    #              'vmax': 0,
    #              'title': 'log10_abs_rs',
    #              'cmap': 'gray'}
    # view_matrix(np.log10(np.abs(m_rs) + 1e-100), **view_args)
    #
    # view_args = {'vmin': -10,
    #              'vmax': 0,
    #              'title': 'log10_abs_sf',
    #              'cmap': 'gray'}
    # view_matrix(np.log10(np.abs(m_sf) + 1e-100), **view_args)
    #
    # view_args = {'vmin': 0,
    #              'vmax': 1,
    #              'title': 'percentage',
    #              'cmap': 'gray'}
    # view_matrix(percentage, **view_args)
    #
    # view_args = {'vmin': 0,
    #              'vmax': -10,
    #              'title': 'log10_percentage',
    #              'cmap': 'gray'}
    # view_matrix(np.log10(percentage + 1e-100), **view_args)