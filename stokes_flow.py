# functions for solving stokes flow using regularised stokeslets (and its improved) method.
# Zhang Ji, 20160409

import copy

import numpy as np
import scipy.io as sio
from evtk.hl import pointsToVTK, gridToVTK
from petsc4py import PETSc

from sf_error import sf_error


class stokesFlowProblem:
    def __init__(self):
        self._obj_list = []  # contain objects
        self._n_obj = 0  # number of objects.
        self._method = ' '  # solving method,
        self._kwargs = {}  # kwargs associate with solving method,
        self._force = np.zeros([0])  # force information
        self._force_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self._velocity = np.zeros([0])  # velocity information
        self._re_velocity = np.zeros([0])  # resolved velocity information
        self._f_node_index_list = [0, ]  # list of node index for objs. IMPORTANT: only store first index of objs.
        self._f_index_list = [0, ]  # list of force index for objs. IMPORTANT: only store first index of objs.
        self._u_node_index_list = [0, ]  # list of node index for objs. IMPORTANT: only store first index of objs.
        self._u_index_list = [0, ]  # list of velocity index for objs. IMPORTANT: only store first index of objs.
        self._M_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)  # M matrix
        self._dimension = -1  # 2 dimension or 3 dimension
        self._finish_solve = False

        import StokesFlowMethod
        self._method_dict = {'sf': StokesFlowMethod.surf_force_matrix_3d,
                             'sf_debug': StokesFlowMethod.surf_force_matrix_3d_debug,
                             'rs': StokesFlowMethod.regularized_stokeslets_matrix_3d_petsc,
                             'ps': StokesFlowMethod.point_source_matrix_3d_petsc, }
        self._check_args_dict = {'sf': StokesFlowMethod.check_surf_force_matrix_3d,
                                 'sf_debug': StokesFlowMethod.check_surf_force_matrix_3d,
                                 'rs': StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
                                 'ps': StokesFlowMethod.check_point_source_matrix_3d_petsc, }

    def add_obj(self, obj):
        """
        Add a new object to the problem.

        :type obj: stokesFlowObject
        :param obj: added object
        :return: none.
        """
        if obj.get_index() != -1:
            ierr = 202
            err_msg = 'the object have been added to the problem once before. '
            raise sf_error(ierr, err_msg)
        self._n_obj += 1
        self._obj_list.append(obj)
        obj.set_index(self._n_obj)

        new_node_index = self._f_node_index_list[-1] + obj.get_n_f_node()
        new_para_index = self._f_index_list[-1] + obj.get_n_force()
        self._f_node_index_list.append(new_node_index)
        self._f_index_list.append(new_para_index)

        new_u_node_index = self._u_node_index_list[-1] + obj.get_n_u_node()
        new_u_index = self._u_index_list[-1] + obj.get_n_velocity()
        self._u_node_index_list.append(new_u_node_index)
        self._u_index_list.append(new_u_index)

        self._velocity = np.hstack((self._velocity, obj.get_velocity()))

    def __repr__(self):
        return 'StokesFlowProblem'

    def create_matrix(self, method, **kwargs):
        """

        :rtype: object
        """
        if len(self._obj_list) == 0:
            ierr = 201
            err_msg = 'at least one object is necessary. '
            raise sf_error(ierr, err_msg)

        # set method.
        self._method = method
        self._check_args_dict[method](**kwargs)
        self._kwargs = kwargs

        # create matrix
        if not self._M_petsc.isAssembled():
            self._M_petsc.setSizes(((None, self._u_index_list[-1]), (None, self._f_index_list[-1])))
            self._M_petsc.setType('dense')
            self._M_petsc.setFromOptions()
            self._M_petsc.setUp()
        for i, obj1 in enumerate(self._obj_list):
            velocity_index_begin = self._f_index_list[i]
            for j, obj2 in enumerate(self._obj_list):
                force_index_begin = self._f_index_list[j]
                force_index_end = self._f_index_list[j + 1]
                temp_m_petsc = self._method_dict[method](obj1, obj2, **kwargs)
                temp_m = temp_m_petsc.getDenseArray()
                temp_m_start, temp_m_end = temp_m_petsc.getOwnershipRange()
                for k in range(temp_m_start, temp_m_end):
                    a = 1
                    self._M_petsc.setValues(velocity_index_begin + k,
                                            np.arange(force_index_begin, force_index_end, dtype='int32'),
                                            temp_m[k - temp_m_start, :])
        self._M_petsc.assemble()

    def solve(self, solve_method, precondition_method):
        force_petsc, velocity_petsc = self._M_petsc.createVecs()
        velocity_petsc[:] = self._velocity[:]
        velocity_petsc.assemble()
        force_petsc.set(0)

        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        ksp.setType(solve_method)
        ksp.getPC().setType(precondition_method)
        ksp.setOperators(self._M_petsc)
        ksp.setFromOptions()
        ksp.solve(velocity_petsc, force_petsc)

        self._force_petsc = force_petsc
        scatter, self._force = PETSc.Scatter.toAll(force_petsc)
        scatter.scatter(force_petsc, self._force, False, PETSc.Scatter.Mode.FORWARD)
        for i, obj in enumerate(self._obj_list):
            obj.set_force(self._force[self._f_index_list[i]:self._f_index_list[i + 1]])

        re_velocity_petsc = self._M_petsc.createVecLeft()
        re_velocity_petsc.set(0)
        self._M_petsc.mult(self._force_petsc, re_velocity_petsc)
        self._re_velocity_petsc = re_velocity_petsc
        scatter, self._re_velocity = PETSc.Scatter.toAll(re_velocity_petsc)
        scatter.scatter(re_velocity_petsc, self._re_velocity, False, PETSc.Scatter.Mode.FORWARD)
        for i, obj in enumerate(self._obj_list):
            obj.set_re_velocity(self._re_velocity[self._u_index_list[i]:self._u_index_list[i + 1]])
        self._finish_solve = True

        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            print("The last KSP residual norm: %e"%(ksp.getResidualNorm()))

    def vtk_force(self, filename):
        if not self._finish_solve:
            ierr = 305
            err_msg = 'call solve() method first. '
            raise sf_error(ierr, err_msg)
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()

        if rank == 0:
            force_x = self._force[0::3].copy()
            force_y = self._force[1::3].copy()
            force_z = self._force[2::3].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])
            nodes = np.ones([self._f_node_index_list[-1], 3], order='F')
            for i, obj in enumerate(self._obj_list):
                nodes[self._f_node_index_list[i]:self._f_node_index_list[i + 1], :] = obj.get_f_nodes()
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"force": (force_x, force_y, force_z),
                              "velocity": (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z)})
            del force_x, force_y, force_z, \
                velocity_x, velocity_y, velocity_z, \
                velocity_err_x, velocity_err_y, velocity_err_z, \
                nodes

    def vtk_check_obj(self, filename, obj):
        """
        check velocity at the surface of objects.

        :type filename: str
        :param filename: output file name
        :type obj: stokesFlowObject
        :param obj: check object (those have similar geometry with problem but different node positions.)
        :return: none.
        """
        if not self._finish_solve:
            ierr = 305
            err_msg = 'call solve() method first. '
            raise sf_error(ierr, err_msg)
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        method = self._method
        kwargs = self._kwargs

        n_u_nodes = obj.get_n_velocity()
        m_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        m_petsc.setSizes(((None, n_u_nodes), (None, self._f_index_list[-1])))
        m_petsc.setType('dense')
        m_petsc.setFromOptions()
        m_petsc.setUp()
        for i, obj1 in enumerate(self._obj_list):
            force_index_begin = self._f_index_list[i]
            force_index_end = self._f_index_list[i + 1]
            temp_m_petsc = self._method_dict[method](obj, obj1, **kwargs)
            temp_m = temp_m_petsc.getDenseArray()
            temp_m_start, temp_m_end = temp_m_petsc.getOwnershipRange()
            for k in range(temp_m_start, temp_m_end):
                m_petsc.setValues(k,
                                  np.arange(force_index_begin, force_index_end, dtype='int32'),
                                  temp_m[k - temp_m_start, :])
        m_petsc.assemble()
        u_petsc = m_petsc.createVecLeft()
        u_petsc.set(0)
        m_petsc.mult(self._force_petsc, u_petsc)
        scatter, u = PETSc.Scatter.toZero(u_petsc)
        scatter.scatter(u_petsc, u, False, PETSc.Scatter.Mode.FORWARD)

        if rank == 0:
            u_exact = obj.get_velocity()
            velocity_x = u[0::3].copy() - u_exact[0::3]
            velocity_y = u[1::3].copy() - u_exact[1::3]
            velocity_z = u[2::3].copy() - u_exact[2::3]
            nodes = obj.get_u_nodes()
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"velocity_err": (velocity_x, velocity_y, velocity_z)})

    def vtk_velocity_rectangle(self, filename: str,
                               field_range: np.ndarray,
                               n_grid: np.ndarray):
        """

        :type self: stokesFlowProblem
        :param self: self
        :type filename: str
        :param filename: output file name.
        :type: field_range: np.array
        :param field_range: range of output velocity field.
        :type: n_grid: np.array
        :param n_grid: number of cells at each direction.
        """

        self._M_petsc.destroy()
        n_range = field_range.shape
        if n_range[0] > n_range[1]:
            field_range = field_range.transpose()
            n_range = field_range.shape
        if n_range != (2, 3):
            ierr = 310
            err_msg = 'maximum and minimum coordinates for the rectangular velocity field are necessary, ' + \
                      'i.e. range = [[0,0,0],[10,10,10]]. '
            raise sf_error(ierr, err_msg)
        if not self._finish_solve:
            ierr = 305
            err_msg = 'call solve() method first. '
            raise sf_error(ierr, err_msg)
        # set method.
        method = self._method
        kwargs = self._kwargs

        n_grid = n_grid.ravel()
        if n_grid.shape != (3,):
            ierr = 311
            err_msg = 'mesh number of each axis for the rectangular velocity field is necessary, ' + \
                      'i.e. n_grid = [100, 100, 100]. '
            raise sf_error(ierr, err_msg)

        # solve velocity at cell center.
        min_range = np.min(field_range, 0)
        max_range = np.max(field_range, 0)
        n_node = n_grid[1] * n_grid[2]
        n_para = 3 * n_node
        full_region_x = np.linspace(min_range[0], max_range[0], n_grid[0])
        full_region_y = np.linspace(min_range[1], max_range[1], n_grid[1])
        full_region_z = np.linspace(min_range[2], max_range[2], n_grid[2])
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()

        # to handle big problem, solve velocity field at every splice along x axis.
        u_x = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
        u_y = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
        u_z = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
        for i0, current_region_x in enumerate(full_region_x):
            [temp_x, temp_y, temp_z] = np.meshgrid(current_region_x, full_region_y, full_region_z, indexing='ij')
            velocity_nodes = np.c_[temp_x.ravel(), temp_y.ravel(), temp_z.ravel()]
            obj0 = stokesFlowObject()
            obj0.set_data(velocity_nodes, velocity_nodes, np.zeros(velocity_nodes.size))
            m_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
            m_petsc.setSizes(((None, n_para), (None, self._f_index_list[-1])))
            m_petsc.setType('dense')
            m_petsc.setFromOptions()
            m_petsc.setUp()
            for i, obj1 in enumerate(self._obj_list):
                force_index_begin = self._f_index_list[i]
                force_index_end = self._f_index_list[i + 1]
                temp_m_petsc = self._method_dict[method](obj0, obj1, **kwargs)
                temp_m = temp_m_petsc.getDenseArray()
                temp_m_start, temp_m_end = temp_m_petsc.getOwnershipRange()
                for k in range(temp_m_start, temp_m_end):
                    m_petsc.setValues(k,
                                      np.arange(force_index_begin, force_index_end, dtype='int32'),
                                      temp_m[k - temp_m_start, :])
            m_petsc.assemble()
            u_petsc = m_petsc.createVecLeft()
            u_petsc.set(0)
            m_petsc.mult(self._force_petsc, u_petsc)
            scatter, u = PETSc.Scatter.toZero(u_petsc)
            scatter.scatter(u_petsc, u, False, PETSc.Scatter.Mode.FORWARD)
            if rank == 0:
                u_x[i0, :, :] = u[0::3].reshape((1, n_grid[1], n_grid[2]))
                u_y[i0, :, :] = u[1::3].reshape((1, n_grid[1], n_grid[2]))
                u_z[i0, :, :] = u[2::3].reshape((1, n_grid[1], n_grid[2]))

        if rank == 0:
            # output data
            delta = (max_range - min_range) / n_grid
            full_region_x = np.linspace(min_range[0] - delta[0] / 2, max_range[0] + delta[0] / 2, n_grid[0] + 1)
            full_region_y = np.linspace(min_range[1] - delta[1] / 2, max_range[1] + delta[1] / 2, n_grid[1] + 1)
            full_region_z = np.linspace(min_range[2] - delta[2] / 2, max_range[2] + delta[2] / 2, n_grid[2] + 1)
            gridToVTK(filename, full_region_x, full_region_y, full_region_z,
                      cellData={"velocity": (u_x, u_y, u_z)})

    def vtk_obj(self, filename):
        for obj1 in self._obj_list:
            obj1.vtk(filename)

    def saveM(self, filename: str = '..', ):
        viewer = PETSc.Viewer().createASCII(filename + '.txt', 'w', comm=PETSc.COMM_WORLD)
        viewer(self._M_petsc)
        viewer.destroy()

    def view_log_M(self, **kwargs):
        m = self._M_petsc.getDenseArray()
        view_args = {'vmin': -10,
                     'vmax': 0,
                     'title': 'log10_abs_' + kwargs['method'],
                     'cmap': 'gray'}
        self._view_matrix(np.log10(np.abs(m) + 1e-100), **view_args)

    def view_M(self, **kwargs):
        m = self._M_petsc.getDenseArray()
        view_args = {'vmin': None,
                     'vmax': None,
                     'title': kwargs['method'],
                     'cmap': 'gray'}
        self._view_matrix(m, **view_args)

    def _view_matrix(self, m, **kwargs):
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

    def get_M(self, **kwargs):
        M = self._M_petsc.getDenseArray().copy()
        return M

    def get_n_f_nodes(self):
        return self._f_node_index_list[-1]

    def get_n_u_nodes(self):
        return self._u_node_index_list[-1]

    def get_obj_list(self):
        return self._obj_list


class stokesFlowObject:
    # general class of object, contain general properties of objcet.
    def __init__(self, filename: str = '..'):
        """
         :type filename str
        :param filename: name of mat file containing object information
        """
        self._index = -1  # index of object
        self._f_nodes = np.zeros([0])  # global coordinates of force nodes
        self._u_nodes = np.zeros([0])  # global coordinates of velocity nodes
        self._velocity = np.zeros([0])  # velocity information
        self._re_velocity = np.zeros([0])  # resolved information
        self._force = np.zeros([0])  # force information
        self._origin = np.zeros([0])  # global coordinate of origin point
        self._local_f_nodes = np.zeros([0])  # local coordinates of force nodes
        self._local_u_nodes = np.zeros([0])  # local coordinates of velocity nodes
        self._type = 'uninitialized'  # object type
        self._name = '...'  # object name

        if filename == '..':
            return
        self.import_mat(filename)

    def __repr__(self):
        return self._type + ': index. %d' % self._index

    def import_mat(self, filename: str):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        """
        mat_contents = sio.loadmat(filename)
        velocity = mat_contents['U'].astype(np.float)
        origin = mat_contents['origin'].astype(np.float)
        f_nodes = mat_contents['f_nodes'].astype(np.float)
        u_nodes = mat_contents['u_nodes'].astype(np.float)
        para = {'origin': origin}
        self.set_data(f_nodes, u_nodes, velocity, **para)

    def import_nodes(self, filename: str):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        """
        mat_contents = sio.loadmat(filename)
        origin = mat_contents['origin'].astype(np.float)
        f_nodes = mat_contents['f_nodes'].astype(np.float)
        u_nodes = mat_contents['u_nodes'].astype(np.float)
        para = {'origin': origin}
        self.set_data(f_nodes, u_nodes, **para)

    def set_data(self,
                 f_nodes: np.array,
                 u_nodes: np.array,
                 velocity: np.array,
                 **kwargs):
        self.set_nodes(f_nodes, u_nodes, **kwargs)
        self.set_velocity(velocity, **kwargs)

    def set_nodes(self,
                  f_nodes: np.array,
                  u_nodes: np.array,
                  **kwargs):
        need_args = []
        for key in need_args:
            if not key in kwargs:
                ierr = 401
                err_msg = 'information about ' + key + \
                          ' is nesscery for surface force method. '
                raise sf_error(ierr, err_msg)

        args = {'origin': np.array([0, 0, 0]),
                'name': '...'}
        for key, value in args.items():
            if not key in kwargs:
                kwargs[key] = args[key]

        self._f_nodes = f_nodes
        self._u_nodes = u_nodes
        self._force = np.zeros(self._f_nodes.size)
        self._origin = kwargs['origin']
        self._name = kwargs['name']
        self._local_f_nodes = self._f_nodes - self._origin
        self._local_u_nodes = self._u_nodes - self._origin
        self._type = 'general obj'

        # TODO: processing value of delta
        # delta = mat_contents['delta']
        # delta = delta[0, 0]
        # return nodes, velocity, delta

    def set_velocity(self,
                     velocity: np.array,
                     **kwargs):
        need_args = []
        for key in need_args:
            if not key in kwargs:
                ierr = 401
                err_msg = 'information about ' + key + \
                          ' is nesscery for surface force method. '
                raise sf_error(ierr, err_msg)

        args = {}
        for key, value in args.items():
            if not key in kwargs:
                kwargs[key] = args[key]

        self._velocity = velocity.reshape(velocity.size)

    def copy(self):
        """
        copy a new object.
        """
        obj2 = copy.deepcopy(self)
        obj2.set_index(-1)
        return obj2

    def get_index(self):
        return self._index

    def get_type(self):
        return self._type

    def get_obj_name(self):
        return self._type + ' (index %d)' % self._index

    def set_index(self, new_index):
        self._index = new_index

    def move(self, delta_origin):
        self._origin += delta_origin
        self._f_nodes += delta_origin
        self._u_nodes += delta_origin

    def get_origin(self):
        return self._origin

    def set_origin(self, new_origin):
        self._origin = new_origin
        self._f_nodes = self._local_f_nodes + new_origin
        self._u_nodes = self._local_u_nodes + new_origin

    def get_f_nodes(self):
        return self._f_nodes

    def get_u_nodes(self):
        return self._u_nodes

    def get_force(self):
        return self._force

    def get_force_x(self):
        return self._force[::3]

    def set_force(self, force):
        self._force = force

    def get_velocity(self):
        return self._velocity

    def set_re_velocity(self, re_velocity):
        self._re_velocity = re_velocity

    def get_n_f_node(self):
        return self._f_nodes.shape[0]

    def get_n_u_node(self):
        return self._u_nodes.shape[0]

    def get_n_force(self):
        return self._force.size

    def get_n_velocity(self):
        return self._velocity.size

    def vtk(self, filename):
        if self._name == '...':
            return

        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            force_x = self._force[0::3].copy()
            force_y = self._force[1::3].copy()
            force_z = self._force[2::3].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])

            f_filename = filename + '_' + self._name + '_force'
            pointsToVTK(f_filename, self._f_nodes[:, 0], self._f_nodes[:, 1], self._f_nodes[:, 2],
                        data={"force": (force_x, force_y, force_z), } )
            u_filename = filename + '_' + self._name + '_velocity'
            pointsToVTK(u_filename, self._u_nodes[:, 0], self._u_nodes[:, 1], self._u_nodes[:, 2],
                        data={"velocity": (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z), })

            del force_x, force_y, force_z, \
                velocity_x, velocity_y, velocity_z, \
                velocity_err_x, velocity_err_y, velocity_err_z, \


class surf_forceObj(stokesFlowObject):
    def __init__(self, filename: str = '..'):
        super(surf_forceObj, self).__init__(filename)
        self._norm = np.zeros([0])  # information about normal vector at each point.

    def import_mat(self, filename: str):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        """
        mat_contents = sio.loadmat(filename)
        velocity = mat_contents['U'].astype(np.float)
        origin = mat_contents['origin'].astype(np.float)
        f_nodes = mat_contents['f_nodes'].astype(np.float)
        u_nodes = mat_contents['u_nodes'].astype(np.float)
        norm = mat_contents['norm'].astype(np.float)
        para = {'norm': norm,
                'origin': origin}
        self.set_data(f_nodes, u_nodes, velocity, **para)

    def import_nodes(self, filename: str):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        """
        mat_contents = sio.loadmat(filename)
        origin = mat_contents['origin'].astype(np.float)
        f_nodes = mat_contents['f_nodes'].astype(np.float)
        u_nodes = mat_contents['u_nodes'].astype(np.float)
        norm = mat_contents['norm'].astype(np.float)
        para = {'norm': norm,
                'origin': origin}
        self.set_nodes(f_nodes, u_nodes, **para)

    def set_data(self,
                 f_nodes: np.array,
                 u_nodes: np.array,
                 velocity: np.array,
                 **kwargs):
        self.set_nodes(f_nodes, u_nodes, **kwargs)
        self.set_velocity(velocity, **kwargs)

    def set_nodes(self,
                  f_nodes: np.array,
                  u_nodes: np.array,
                  **kwargs):
        need_args = []
        for key in need_args:
            if not key in kwargs:
                ierr = 401
                err_msg = 'information about ' + key + \
                          ' is nesscery for surface force method. '
                raise sf_error(ierr, err_msg)

        args = {'origin': np.array([0, 0, 0])}
        for key, value in args.items():
            if not key in kwargs:
                kwargs[key] = args[key]

        super(surf_forceObj, self).set_nodes(f_nodes, u_nodes, **kwargs)
        self._norm = kwargs['norm']
        self._type = 'surface force obj'

    def get_norm(self):
        return self._norm


class pointSourceObj(stokesFlowObject):
    def set_nodes(self,
                  f_nodes: np.array,
                  u_nodes: np.array,
                  **kwargs):
        super(pointSourceObj, self).set_nodes(f_nodes, u_nodes, **kwargs)
        self._type = 'point source obj'

    def get_n_force(self):
        return self._f_nodes.shape[0] * 4

    def get_force_x(self):
        return self._force[::4]

    def vtk(self, filename):
        if self._name == '...':
            return

        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            force_x = self._force[0::4].copy()
            force_y = self._force[1::4].copy()
            force_z = self._force[2::4].copy()
            pointSource = self._force[3::4].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])

            f_filename = filename + '_' + self._name + '_force'
            pointsToVTK(f_filename, self._f_nodes[:, 0], self._f_nodes[:, 1], self._f_nodes[:, 2],
                        data={"force": (force_x, force_y, force_z),
                              "point_source": pointSource} )
            u_filename = filename + '_' + self._name + '_velocity'
            pointsToVTK(u_filename, self._u_nodes[:, 0], self._u_nodes[:, 1], self._u_nodes[:, 2],
                        data={"velocity": (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z), })

            del force_x, force_y, force_z, \
                velocity_x, velocity_y, velocity_z, \
                velocity_err_x, velocity_err_y, velocity_err_z, \



class pointSourceProblem(stokesFlowProblem):
    def vtk_force(self, filename):
        if not self._finish_solve:
            ierr = 305
            err_msg = 'call solve() method first. '
            raise sf_error(ierr, err_msg)
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()

        if rank == 0:
            force_x = self._force[0::4].copy()
            force_y = self._force[1::4].copy()
            force_z = self._force[2::4].copy()
            pointSource = self._force[3::4].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])
            nodes = np.ones([self._f_node_index_list[-1], 3], order='F')
            for i, obj in enumerate(self._obj_list):
                nodes[self._f_node_index_list[i]:self._f_node_index_list[i + 1], :] = obj.get_f_nodes()
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"force": (force_x, force_y, force_z),
                              "velocity": (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z),
                              "point_source": pointSource})
            del force_x, force_y, force_z, \
                velocity_x, velocity_y, velocity_z, \
                velocity_err_x, velocity_err_y, velocity_err_z, \
                nodes, pointSource