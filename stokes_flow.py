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
                   'sf': StokesFlowMethod.surface_force_matrix_3d}
    check_args_dict = {'rs': StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
                       'sf': StokesFlowMethod.check_surface_force_matrix_3d}

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
        self.__velocity = np.nan  # velocity information
        self.__node_index_list = []  # list of node index for objs. IMPORTANT: only store first index of objs.
        self.__para_index_list = []  # list of force index for objs. IMPORTANT: only store first index of objs.
        self.__M = np.ndarray([0, 0])  # M matrix
        self.__dimention = -1  # 2 dimention or 3 dimention
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

    def create_mtrix(self, method, **kwargs):
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
        self.__M = np.ndarray([self.__para_index_list[-1], self.__para_index_list[-1]])
        for i, obj1 in enumerate(self.__obj_list):
            velocity_nodes = obj1.get_nodes()
            velocity_index_begin = self.__para_index_list[i]
            velocity_index_end = self.__para_index_list[i + 1]
            for j, obj2 in enumerate(self.__obj_list):
                force_nodes = obj2.get_nodes()
                force_index_begin = self.__para_index_list[j]
                force_index_end = self.__para_index_list[j + 1]
                self.__M[velocity_index_begin:velocity_index_end, force_index_begin:force_index_end] \
                    = self.method_dict[method](velocity_nodes, force_nodes, **kwargs)

    def solve(self, solve_method, precondition_method):
        pc_rs_m = PETSc.Mat().createDense(size=self.__M.shape, array=self.__M)
        pc_velocity = PETSc.Vec().createWithArray(self.__velocity)
        pc_force = pc_rs_m.getVecRight()
        pc_force.set(0)

        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setType(solve_method)
        ksp.getPC().setType(precondition_method)
        ksp.setOperators(pc_rs_m)
        ksp.setFromOptions()
        ksp.solve(pc_velocity, pc_force)
        self.__force = pc_force.getArray()
        for i, obj in enumerate(self.__obj_list):
            obj.set_force(self.__force[self.__para_index_list[i]:self.__para_index_list[i + 1]])
        self.__finish_solve = True

    def vtk_force(self, filename):
        if not (self.__finish_solve):
            ierr = 305
            err_msg = 'call solve() method first. '
            raise sf_error(ierr, err_msg)

        force_x = self.__force[0::3].ravel()
        force_y = self.__force[1::3].ravel()
        force_z = self.__force[2::3].ravel()
        force_total = (force_x ** 2 + force_y ** 2 + force_z ** 2) ** 0.5
        velocity_x = self.__velocity[0::3].ravel()
        velocity_y = self.__velocity[1::3].ravel()
        velocity_z = self.__velocity[2::3].ravel()
        velocity_total = (velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2) ** 0.5
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
                          "velocity_total": velocity_total})
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
        n_node = n_grid[0] * n_grid[1] * n_grid[2]
        n_para = 3 * n_node
        full_region_x = np.linspace(min_range[0], max_range[0], n_grid[0])
        full_region_y = np.linspace(min_range[1], max_range[1], n_grid[1])
        full_region_z = np.linspace(min_range[2], max_range[2], n_grid[2])
        [temp_x, temp_y, temp_z] = np.meshgrid(full_region_x, full_region_y, full_region_z, indexing='ij')
        velocity_nodes = np.c_[temp_x.ravel(), temp_y.ravel(), temp_z.ravel()]
        m = np.ndarray([n_para, self.__para_index_list[-1]])
        for i, obj1 in enumerate(self.__obj_list):
            force_nodes = obj1.get_nodes()
            force_index_begin = self.__para_index_list[i]
            force_index_end = self.__para_index_list[i + 1]
            m[0:n_para, force_index_begin:force_index_end] \
                = self.method_dict[method](velocity_nodes, force_nodes, **kwargs)
        u = np.dot(m, self.__force)
        u_x = u[0::3].ravel().reshape(n_grid, order='A')
        u_y = u[1::3].ravel().reshape(n_grid, order='A')
        u_z = u[2::3].ravel().reshape(n_grid, order='A')

        # output data
        delta = (max_range - min_range) / n_grid
        full_region_x = np.linspace(min_range[0] - delta[0] / 2, max_range[0] + delta[0] / 2, n_grid[0] + 1)
        full_region_y = np.linspace(min_range[1] - delta[1] / 2, max_range[1] + delta[1] / 2, n_grid[1] + 1)
        full_region_z = np.linspace(min_range[2] - delta[2] / 2, max_range[2] + delta[2] / 2, n_grid[2] + 1)
        [temp_x, temp_y, temp_z] = np.meshgrid(full_region_x, full_region_y, full_region_z, indexing='ij')
        gridToVTK(filename, temp_x, temp_y, temp_z,
                  cellData={"velocity": (u_x, u_y, u_z)})


class StokesFlowObject:
    # general class of object, contain general properties of objcet.
    def __init__(self, father_obj: StokesFlowComponent, filename: str = '..', index: int = -1):
        """

        :type father_obj StokesFlowComponent
        :param father_obj: point to main obj, the problem.
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
        self.__father = father_obj  # father object

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
