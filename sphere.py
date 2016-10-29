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


# @profile
def main_fun():
    OptDB = PETSc.Options()
    radius = OptDB.getReal('r', 1)
    deltaLength = OptDB.getReal('d', 0.5)
    epsilon = OptDB.getReal('e', 0.25)
    u = OptDB.getReal('u', 1)
    filename = OptDB.getString('f', 'sphere')
    sphere_mesh_name = OptDB.getString('sphere', '..')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
    debug_mode = OptDB.getBool('debug', False)
    u_multiple = OptDB.getInt('m', 1)
    matrix_method = OptDB.getString('sm', 'rs')

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
    u_nSphere = int(16 * radius * radius / deltaLength / deltaLength)
    if sphere_mesh_name != '..':
        f_nodesSphere, u_nodesSphere, origon = geo.mat_nodes(sphere_mesh_name)
        uSphere = geo.mat_velocity(sphere_mesh_name)
        normSphere = geo.norm_sphere(f_nodesSphere)
    else:
        u_nodesSphere, uSphere, normSphere = geo.sphere(u_nSphere, radius, radius, [u, 0, 0, 0, 0, 0])
        f_nodesSphere = u_nodesSphere[::u_multiple, :]
    uSphere = uSphere.flatten()
    f_nSphere = f_nodesSphere.shape[0]
    u_nSphere = u_nodesSphere.shape[0]

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
                         'delta': deltaLength * epsilon,  # for rs method
                         'd_radia': deltaLength / 2}  # for sf method

    if rank == 0:
        if sphere_mesh_name == '..':
            print('sphere radius: %f, delta length: %f, velocity: %f, multiple coefficient: %d'
                  % (radius, deltaLength0, u, u_multiple))
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
        print('Number of force and velocity nodes are %d and %d, respectively.'
              % (f_nSphere, u_nSphere))
        print('solve method: %s, precondition method: %s'
              % (solve_method, precondition_method))
        print('output path: ' + filename)
        print('MPI size: %d' % size)

    if plot:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(u_nodesSphere[:, 0], u_nodesSphere[:, 1], u_nodesSphere[:, 2], c='b', marker='o')
        ax.scatter(f_nodesSphere[:, 0], f_nodesSphere[:, 1], f_nodesSphere[:, 2], c='r', marker='o')
        ax.quiver(u_nodesSphere[:, 0], u_nodesSphere[:, 1], u_nodesSphere[:, 2],
                  uSphere[0::3], uSphere[1::3], uSphere[2::3],
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
    problem.create_matrix(**problem_arguments)
    M = problem.get_M()
    t1 = time()
    if rank == 0:
        print('create matrix use: %f (s)' % (t1 - t0))

    t0 = time()
    # problem.view_M(**problem_arguments)
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
        problem.vtk_check_obj(filename + 'Check_sphere', obj_check)
        t1 = time()
        if rank == 0:
            print('write check file use: %f (s)' % (t1 - t0))

        t0 = time()
        problem.vtk_velocity_rectangle('%sVelocity_%2d_%2d' % (filename, n_obj_x, n_obj_y), field_range, n_grid)
        t1 = time()
        if rank == 0:
            print('write velocity file use: %f (s)' % (t1 - t0))

    if rank == 0:
        force_sphere = obj_sphere.get_force_x()
        print('---->>>Resultant at x axis is %f' % (np.sum(force_sphere)))

    return M


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
