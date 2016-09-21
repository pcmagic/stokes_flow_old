# functions for solving stokes flow using regularised stokeslets (and its improved) method.
# Zhang Ji, 20160409

import copy
import numpy as np
import scipy.io as sio
from evtk.hl import pointsToVTK, gridToVTK
from petsc4py import PETSc
from sf_error import sf_error

class StokesFlowComponent:
    import StokesFlowMethod
    method_dict = {'rs': StokesFlowMethod.regularized_stokeslets_matrix_3d,
                   'sf': StokesFlowMethod.surface_force_matrix_3d,
                   'rs_petsc': StokesFlowMethod.regularized_stokeslets_matrix_3d_petsc,}
    check_args_dict = {'rs': StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
                       'sf': StokesFlowMethod.check_surface_force_matrix_3d,
                       'rs_petsc': StokesFlowMethod.check_regularized_stokeslets_matrix_3d,}

    # try:
    #     self.__method = method_dict[method]
    # except KeyError:
    #     ierr = 203
    #     err_msg = 'Wrong method name. '
    #     raise sf_error(ierr, err_msg)

    def __init__(self):
        self.__obj_list = []  # contain objects
        self.__n_obj = 0  # number of objects.
        self.__method = ' '  # solving method,
        self.__kwargs = {}  # kwargs associate with solving method,
        self.__force = np.nan  # force information
        self.__force_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.__velocity = np.nan  # velocity information
        self.__re_velocity = np.nan  #check velocity field.
        self.__node_index_list = []  # list of node index for objs. IMPORTANT: only store first index of objs.
        self.__para_index_list = []  # list of force index for objs. IMPORTANT: only store first index of objs.
        self.__M_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)  # M matrix
        self.__dimension = -1  # 2 dimension or 3 dimension
        self.__finish_solve = False

    def add_obj(self, obj):
        """
        Add a new object to the problem.

        :type obj: StokesFlowObject
        :param obj: added object
        :return: none.
        """
        if obj.get_index() != -1:
            ierr = 202
            err_msg = 'the object have been added to the problem once before. '
            raise sf_error(ierr, err_msg)
        self.__n_obj += 1
        self.__obj_list.append(obj)
        obj.set_index(self.__n_obj)

    def __repr__(self):
        return 'StokesFlowComponent'

    def create_matrix(self, method, **kwargs):
        """

        :rtype: object
        """
        if len(self.__obj_list) == 0:
            ierr = 201
            err_msg = 'at least one object is necessary. '
            raise sf_error(ierr, err_msg)

        # set method.
        self.__method = method
        self.check_args_dict[method](**kwargs)
        self.__kwargs = kwargs

        # processing node and force index lists.
        self.__node_index_list.append(0)
        self.__para_index_list.append(0)
        for obj in self.__obj_list:
            new_node_index = self.__node_index_list[-1] + obj.get_n_node()
            new_para_index = self.__para_index_list[-1] + obj.get_n_para()
            self.__node_index_list.append(new_node_index)
            self.__para_index_list.append(new_para_index)

        # processing velocity
        self.__velocity = np.ones([self.__para_index_list[-1]])
        for i, obj in enumerate(self.__obj_list):
            self.__velocity[self.__para_index_list[i]:self.__para_index_list[i + 1]] \
                = obj.get_velocity()

        # create matrix
        self.__M_petsc.setSizes(((None, self.__para_index_list[-1]), (None, self.__para_index_list[-1])))
        self.__M_petsc.setType('dense')
        self.__M_petsc.setFromOptions()
        self.__M_petsc.setUp()
        for i, obj1 in enumerate(self.__obj_list):
            velocity_nodes = obj1.get_nodes()
            velocity_index_begin = self.__para_index_list[i]
            velocity_index_end = self.__para_index_list[i + 1]
            for j, obj2 in enumerate(self.__obj_list):
                force_nodes = obj2.get_nodes()
                force_index_begin = self.__para_index_list[j]
                force_index_end = self.__para_index_list[j + 1]
                temp_m_petsc = self.method_dict[method](velocity_nodes, force_nodes, **kwargs)
                temp_m = temp_m_petsc.getDenseArray()
                temp_m_start, temp_m_end = temp_m_petsc.getOwnershipRange()
                for k in range(temp_m_start, temp_m_end):
                    self.__M_petsc.setValues(velocity_index_begin+k,
                                             np.arange(force_index_begin, force_index_end, dtype='int32'),
                                             temp_m[k-temp_m_start, :])
        self.__M_petsc.assemble()

    def solve(self, solve_method, precondition_method):
        velocity_petsc, force_petsc = self.__M_petsc.createVecs()
        velocity_petsc[:] = self.__velocity[:]
        velocity_petsc.assemble()
        force_petsc.set(0)

        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        ksp.setType(solve_method)
        ksp.getPC().setType(precondition_method)
        ksp.setOperators(self.__M_petsc)
        ksp.setFromOptions()
        ksp.solve(velocity_petsc, force_petsc)
        scatter, self.__force = PETSc.Scatter.toAll(force_petsc)
        scatter.scatter(force_petsc, self.__force, False, PETSc.Scatter.Mode.FORWARD)
        self.__force_petsc = force_petsc
        for i, obj in enumerate(self.__obj_list):
            obj.set_force(self.__force[self.__para_index_list[i]:self.__para_index_list[i + 1]])

        re_velocity_petsc = self.__M_petsc.createVecLeft()
        re_velocity_petsc.set(0)
        self.__M_petsc.mult(force_petsc, re_velocity_petsc)
        scatter, self.__re_velocity = PETSc.Scatter.toAll(re_velocity_petsc)
        scatter.scatter(re_velocity_petsc, self.__re_velocity, False, PETSc.Scatter.Mode.FORWARD)
        self.__finish_solve = True

    def vtk_force(self, filename):
        if not (self.__finish_solve):
            ierr = 305
            err_msg = 'call solve() method first. '
            raise sf_error(ierr, err_msg)
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            force_x = self.__force[0::3].copy()
            force_y = self.__force[1::3].copy()
            force_z = self.__force[2::3].copy()
            force_total = (force_x ** 2 + force_y ** 2 + force_z ** 2) ** 0.5
            velocity_x = self.__velocity[0::3].copy()
            velocity_y = self.__velocity[1::3].copy()
            velocity_z = self.__velocity[2::3].copy()
            velocity_err_x = np.abs(self.__re_velocity[0::3] - self.__velocity[0::3])
            velocity_err_y = np.abs(self.__re_velocity[1::3] - self.__velocity[1::3])
            velocity_err_z = np.abs(self.__re_velocity[2::3] - self.__velocity[2::3])
            velocity_total = (velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2) ** 0.5
            velocity_err_total = (velocity_err_x ** 2 + velocity_err_y ** 2 + velocity_err_z ** 2) ** 0.5
            nodes = np.ones([self.__node_index_list[-1], 3], order='F')
            for i, obj in enumerate(self.__obj_list):
                nodes[self.__node_index_list[i]:self.__node_index_list[i + 1], :] = obj.get_nodes()
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"force_x": force_x,
                              "force_y": force_y,
                              "force_z": force_z,
                              "force_total": force_total,
                              "velocity_x": velocity_x,
                              "velocity_y": velocity_y,
                              "velocity_z": velocity_z,
                              "velocity_total": velocity_total,
                              "velocity_err_x": velocity_err_x,
                              "velocity_err_y": velocity_err_y,
                              "velocity_err_z": velocity_err_z,
                              "velocity_err_total": velocity_err_total})
            del force_x, force_y, force_z, force_total, \
                velocity_x, velocity_y, velocity_z, velocity_total, \
                nodes

    def vtk_velocity(self, filename: str,
                     field_range: np.ndarray,
                     n_grid: np.ndarray):
        """

        :type self: StokesFlowComponent
        :param self: self
        :type filename: str
        :param filename: output file name.
        :type: range: np.array
        :param range: range of output velocity field.
        """

        n_range = field_range.shape
        if n_range[0] > n_range[1]:
            field_range = field_range.transpose()
            n_range = field_range.shape
        if n_range != (2, 3):
            ierr = 310
            err_msg = 'maximum and minimum coordinates for the rectangular velocity field are necessary, ' + \
                      'i.e. range = [[0,0,0],[10,10,10]]. '
            raise sf_error(ierr, err_msg)
        if not (self.__finish_solve):
            ierr = 305
            err_msg = 'call solve() method first. '
            raise sf_error(ierr, err_msg)
        # set method.
        method = self.__method
        kwargs = self.__kwargs

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
            m_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
            m_petsc.setSizes(((None, n_para), (None, self.__para_index_list[-1])))
            m_petsc.setType('dense')
            m_petsc.setFromOptions()
            m_petsc.setUp()
            for i, obj1 in enumerate(self.__obj_list):
                force_nodes = obj1.get_nodes()
                force_index_begin = self.__para_index_list[i]
                force_index_end = self.__para_index_list[i + 1]
                temp_m_petsc = self.method_dict[method](velocity_nodes, force_nodes, **kwargs)
                temp_m = temp_m_petsc.getDenseArray()
                temp_m_start, temp_m_end = temp_m_petsc.getOwnershipRange()
                for k in range(temp_m_start, temp_m_end):
                    m_petsc.setValues(k, np.arange(force_index_begin, force_index_end, dtype='int32'), temp_m[k-temp_m_start, :])
            m_petsc.assemble()
            u_petsc = m_petsc.createVecLeft()
            u_petsc.set(0)
            m_petsc.mult(self.__force_petsc, u_petsc)
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

    def saveM(self, filename: str = '..',):
        viewer = PETSc.Viewer().createASCII(filename + '.txt', 'w', comm=PETSc.COMM_WORLD)
        viewer(self.__M_petsc)
        viewer.destroy()


class StokesFlowObject:
    # general class of object, contain general properties of objcet.
    def __init__(self, filename: str = '..'):
        """
         :type filename str
        :param filename: name of mat file containing object information
         :type index int
        :param index: object index
        """
        self.__index = -1  # index of object
        self.__nodes = np.nan  # global coordinates of nodes
        self.__velocity = np.nan  # velocity information
        self.__force = np.nan  # force information
        self.__origin = np.nan  # global coordinate of origin point
        self.__local_nodes = np.nan  # local coordinates of nodes
        self.__type = 'uninitialized'  # object name

        if filename == '..':
            return
        self.import_mat(filename)

    def __repr__(self):
        return self.__type + ': index. %d' % self.__index

    def import_mat(self, filename: str):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        :type index: int
        :param index: object index
        """
        mat_contents = sio.loadmat(filename)
        velocity = mat_contents['U']
        origin = mat_contents['origin']

        # TODO: check the formats of U and origin
        self.__nodes = mat_contents['nodes'].astype(np.float64)
        self.__velocity = velocity.flatten(1).astype(np.float64)
        self.__force = self.__velocity.copy()
        self.__force.fill(0)
        self.__origin = origin.flatten(1).astype(np.float64)
        self.__local_nodes = self.__nodes - self.__origin
        self.__type = 'general obj'

        # TODO: processing value of delta
        # delta = mat_contents['delta']
        # delta = delta[0, 0]
        # return nodes, velocity, delta

    def set_data(self,
                 nodes: np.array,
                 velocity: np.array,
                 origin: np.array = np.array([0, 0, 0])):
        self.__nodes = nodes
        self.__velocity = velocity
        self.__force = self.__velocity.copy()
        self.__force.fill(0)
        self.__origin = origin
        self.__local_nodes = self.__nodes - self.__origin
        self.__type = 'general obj'

        # TODO: processing value of delta
        # delta = mat_contents['delta']
        # delta = delta[0, 0]
        # return nodes, velocity, delta

    def copy(self):
        """
        copy a new object.

        :type new_index: int
        :param new_index: object index
        """
        obj2 = copy.deepcopy(self)
        obj2.set_index(-1)
        return obj2

    def get_index(self):
        return self.__index

    def set_index(self, new_index):
        self.__index = new_index

    def move(self, delta_origin):
        self.__origin += delta_origin
        self.__nodes += delta_origin

    def get_origin(self):
        return self.__origin

    def set_origin(self, new_origin):
        self.__origin = new_origin
        self.__nodes = self.__local_nodes + new_origin

    def get_nodes(self):
        return self.__nodes

    def get_force(self):
        return self.__force

    def set_force(self, force):
        self.__force = force

    def get_velocity(self):
        return self.__velocity

    def get_n_node(self):
        return len(self.__nodes)

    def get_n_para(self):
        return len(self.__force)
