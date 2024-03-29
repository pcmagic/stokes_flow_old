# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410

import petsc4py
import sys

petsc4py.init(sys.argv)
import numpy as np
import stokes_flow as sf
from petsc4py import PETSc
import geo
from time import time
from sf_error import sf_error


# @profile
def main_fun():
    OptDB = PETSc.Options()
    length = OptDB.getReal('l', 2)
    radius = OptDB.getReal('r', 1)
    aRinv = OptDB.getReal('a', 2)
    deltaLength = OptDB.getReal('d', 1)
    epsilon = OptDB.getReal('e', 0.25)
    u = OptDB.getReal('u', 1)
    filename = OptDB.getString('f', 'sphereInTunnel')
    sphere_mesh_name = OptDB.getString('sphere', '..')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
    debug_mode = OptDB.getBool('debug', False)
    u_multiple = OptDB.getInt('m', 1)
    matrix_method = OptDB.getString('sm', 'rs')
    ps_para1 = OptDB.getReal('ps_para', 1.25)   # scale factor of radiues for velocity and force nodes.

    problem_dic = {'rs': sf.stokesFlowProblem,
                   'sf': sf.stokesFlowProblem,
                   'sf_debug': sf.stokesFlowProblem,
                   'ps': sf.pointSourceProblem, }
    obj_dic = {'rs': sf.stokesFlowObject,
               'sf': sf.surf_forceObj,
               'sf_debug': sf.surf_forceObj,
               'ps': sf.pointSourceObj, }
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    deltaLength0 = deltaLength
    deltaLength = deltaLength / np.sqrt(u_multiple)
    # Sphere geo
    u_nSphere = int(16 * radius * radius / deltaLength / deltaLength)
    if sphere_mesh_name != '..':
        f_nodesSphere, u_nodesSphere, origon = geo.mat_nodes(sphere_mesh_name)
        uSphere = geo.mat_velocity(sphere_mesh_name)
        normSphere = geo.norm_sphere(f_nodesSphere)
        if rank == 0:
            print('---->>>imput sphere mesh from %s.mat' % (sphere_mesh_name))
    else:
        u_nodesSphere, uSphere, normSphere = geo.sphere(u_nSphere, radius, radius, [u, 0, 0, 0, 0, 0])
        f_nodesSphere = u_nodesSphere[::u_multiple, :]
    f_nSphere = f_nodesSphere.shape[0]
    u_nSphere = u_nodesSphere.shape[0]
    uSphere = uSphere.flatten()

    # Tunnel geo
    u_nodesTunnel, uTunnel, normTunnel = geo.tunnel(deltaLength, length, radius * aRinv)
    if matrix_method == 'ps':
        ps_para3 = u_nSphere / f_nSphere
        ps_para2 = np.sqrt(ps_para3 * ps_para1)
        f_nodesTunnel, _, _ = geo.tunnel(deltaLength * ps_para2, length, radius * aRinv * ps_para1)
    else:
        f_nodesTunnel = u_nodesTunnel[::u_multiple, :]
    uTunnel = uTunnel.flatten()
    f_nTunnel = f_nodesTunnel.shape[0]
    u_nTunnel = u_nodesTunnel.shape[0]

    field_range = np.array([[-length / 2, radius, -radius], [length / 2, radius * aRinv, radius]])
    n_grid = np.array([length, (aRinv - 1), 1], dtype='int') * 20
    problem_arguments = {'method': matrix_method,
                         'delta': deltaLength * epsilon,  # for rs method
                         'd_radia': deltaLength / 2}  # for sf method

    if rank == 0:
        print('tunnel length: %f, tunnel radius: %f, delta length: %f, multiple coefficient: %d'
              % (length, radius * aRinv, deltaLength0, u_multiple))
        if sphere_mesh_name == '..':
            print('sphere radius: %f, velocity: %f, multiple coefficient: %d'
                  % (radius, u, u_multiple))
        else:
            print('imput sphere mesh from %s.mat' % (sphere_mesh_name))
        if matrix_method == 'rs':
            print('create matrix method: %s, epsilon: %f'
                  % (matrix_method, epsilon))
        elif matrix_method == 'sf':
            print('create matrix method: %s' % matrix_method)
        elif matrix_method == 'ps':
            print('create matrix method: %s' % matrix_method)
        else:
            print('please specify how to descript the metrix create method')

        print('Number of force nodes for tunnel and sphere are %d and %d, respectively.'
              % (f_nTunnel, f_nSphere))
        print('Number of velocity nodes for tunnel and sphere are %d and %d, respectively.'
              % (u_nTunnel, u_nSphere))

        print('solve method: %s, precondition method: %s'
              % (solve_method, precondition_method))
        print('output path: ' + filename)
        print('MPI size: %d' % size)

    if u_multiple != 1 and solve_method != 'lsqr':
        ierr = 51
        err_msg = 'Only lsqr method is avalable when the number of velocity nodes is greater than force nodes. '
        raise sf_error(ierr, err_msg)

    if plot:
        if rank == 0:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            u_nodes = np.concatenate((u_nodesSphere, u_nodesTunnel), axis=0)
            u = np.concatenate((uSphere, uTunnel), axis=0)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(u_nodesSphere[:, 0], u_nodesSphere[:, 1], u_nodesSphere[:, 2], c='b', marker='o')
            ax.scatter(f_nodesSphere[:, 0], f_nodesSphere[:, 1], f_nodesSphere[:, 2], c='r', marker='o')
            ax.scatter(u_nodesTunnel[:, 0], u_nodesTunnel[:, 1], u_nodesTunnel[:, 2], c='b', marker='o')
            ax.scatter(f_nodesTunnel[:, 0], f_nodesTunnel[:, 1], f_nodesTunnel[:, 2], c='r', marker='o')
            ax.quiver(u_nodes[:, 0], u_nodes[:, 1], u_nodes[:, 2],
                      u[0::3], u[1::3], u[2::3],
                      color='r', length=deltaLength * 2)
            ax.axis(v='equal')
            plt.show()

    t0 = time()
    problem = problem_dic[matrix_method]()
    obj_sphere = obj_dic[matrix_method]()
    para = {'norm': normSphere,
            'name': 'sphereObj', }
    obj_sphere.set_data(f_nodesSphere, u_nodesSphere, uSphere, **para)
    problem.add_obj(obj_sphere)
    obj_tunnel = obj_dic[matrix_method]()
    para = {'norm': normTunnel,
            'name': 'tunnelObj', }
    obj_tunnel.set_data(f_nodesTunnel, u_nodesTunnel, uTunnel, **para)
    problem.add_obj(obj_tunnel)
    problem.create_matrix(**problem_arguments)
    t1 = time()
    if rank == 0:
        print('create matrix use: %f (s)' % (t1 - t0))

    t0 = time()
    # problem.view_M(**problem_arguments)
    M = problem.get_M()
    # problem.saveM(filename + 'M_rs_petsc')
    # if rank == 0:
    #     print('save finished. ')
    problem.solve(solve_method, precondition_method)
    t1 = time()
    if rank == 0:
        print('solve matrix equation use: %f (s)' % (t1 - t0))

    t0 = time()
    problem.vtk_obj(filename)
    t1 = time()
    if rank == 0:
        print('write force file use: %f (s)' % (t1 - t0))

    if not debug_mode:
        t0 = time()
        u_nodesSphere_check, uSphere_check, _ = \
            geo.sphere(u_nodesSphere.shape[0] + 1, radius, radius, [u, 0, 0, 0, 0, 0])
        f_nodesSphere_check = u_nodesSphere_check
        obj_check = sf.stokesFlowObject()
        obj_check.set_data(f_nodesSphere_check, u_nodesSphere_check, uSphere_check)
        problem.vtk_check_obj(filename + '_sphereCheck', obj_check)
        u_nodesTunnel_check = u_nodesTunnel + deltaLength / 2
        uTunnel_check = uTunnel
        f_nodesTunnel_check = u_nodesTunnel_check
        obj_check = sf.stokesFlowObject()
        obj_check.set_data(f_nodesTunnel_check, u_nodesTunnel_check, uTunnel_check)
        problem.vtk_check_obj(filename + '_tunnelCheck', obj_check)
        t1 = time()
        if rank == 0:
            print('write check file use: %f (s)' % (t1 - t0))

        t0 = time()
        problem.vtk_velocity_rectangle(filename + '_velocity', field_range, n_grid)
        t1 = time()
        if rank == 0:
            print('write velocity file use: %f (s)' % (t1 - t0))

    if rank == 0:
        force_sphere = obj_sphere.get_force_x()
        print('---->>>Resultant at x axis is %f' % (np.sum(force_sphere)))

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
    main_fun()

    # OptDB = PETSc.Options()
    # OptDB.setValue('sm', 'rs')
    # m_rs = main_fun()
    # OptDB.setValue('sm', 'sf')
    # m_sf = main_fun()
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
