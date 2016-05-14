# functions for solving stokes flow using regularised stokeslets (and its improved) method.
# Zhang Ji, 20160409

import numpy as np
import scipy.io as sio
import copy

from sf_error import sf_error


class StokesFlowComponent:
    def __init__(self):
        self.__indexs = []  # contain indexs of objects
        self.__objs = []  # contain objects
        self.__method = []  # solving method
        self.__nodes = np.empty(shape=0)  # global coordinate
        self.__force = np.nan  # force information
        self.__velocity = np.nan  # velocity information

    def add_obj(self, obj):
        """
        Add a new object to the problem.

        :type obj: StokesFlowObject
        :param obj: added object
        :return: none.
        """
        self.__indexs.append(obj.get_index())
        self.__objs.append(obj)


    def index_exist(self, index):
        """

        :type index int
        :param index: obj index
        :rtype: bool
        :return: if the index is exist.
        """
        return index in self.__indexs

    def __repr__(self):
        return 'StokesFlowComponent'

    def collect_nodes(self):
        for obj in self.__objs:
            self.__nodes = np.vstack((self.__nodes,obj.get_nodes()))

    def create_mtrix(self):
        return

    def set_method(self, method):
        self.__method = method

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
        self.__index = np.nan  # index of object
        self.__nodes = np.nan  # global coordinates of nodes
        self.__velocity = np.nan  # velocity information
        self.__force = np.nan  # force information
        self.__origin = np.nan  # global coordinate of origin point
        self.__local_nodes = np.nan  # local coordinates of nodes
        self.__type = 'uninitialized'  # object name
        self.__node_index = np.nan  # index of nodes in the hole problem
        self.__freedom_index = np.nan  # index of freedom in the hole problem
        self.__father = father_obj  # father object

        if filename == '..':
            return
        self.import_mat(filename, index)

    def __repr__(self):
        return self.__type + ': index. %d' % self.__index

    def import_mat(self, filename: str, index: int):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        :type index: int
        :param index: object index
        """
        self.check_index(index)
        mat_contents = sio.loadmat(filename)
        velocity = mat_contents['U']
        origin = mat_contents['origin']

        # TODO: check the formats of U and origin
        self.__index = index
        self.__nodes = mat_contents['nodes']
        self.__velocity = velocity.flatten(1)
        self.__force = self.__velocity.copy()
        self.__force.fill(0)
        self.__origin = origin.flatten(1)
        self.__local_nodes = self.__nodes - self.__origin
        self.__type = 'general obj'

        # TODO: processing value of delta
        # delta = mat_contents['delta']
        # delta = delta[0, 0]
        # return nodes, velocity, delta

    def copy(self, new_index: int):
        """
        copy a new object.

        :type new_index: int
        :param new_index: object index
        """
        self.check_index(new_index)
        obj2 = copy.deepcopy(self)
        obj2.set_index(new_index)
        return obj2

    def check_index(self, index: int):
        """
        check weather the index is legal

        :type index: int
        :param index: object index
        """
        if not (isinstance(index, int)):
            ierr = 100
            err_msg = 'the index of object must be integer. '
            raise sf_error(ierr, err_msg)
        if index < 1:
            ierr = 101
            err_msg = 'the index of object must equal or great then 1. '
            raise sf_error(ierr, err_msg)
        if self.__father.index_exist(index):
            ierr = 102
            err_msg = 'the index of object must unique. '
            raise sf_error(ierr, err_msg)

    def get_index(self):
        return self.__index

    def set_index(self, new_index):
        self.__index = new_index

    def move(self, new_origin):
        self.__origin = new_origin
        self.__nodes = self.__local_nodes + new_origin

    def get_origin(self):
        return self.__origin

    def get_nodes(self):
        return self.__nodes

def regularized_stokeslets_matrix_3d(vnodes: np.ndarray,  # nodes contain velocity information
                                     fnodes: np.ndarray,  # nodes contain force information
                                     delta: float):  # correction factor
    # Solve m matrix using regularized stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets[J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.

    n_vnode = vnodes.shape
    if n_vnode[0] < n_vnode[1]:
        vnodes = vnodes.transpose()
        n_vnode = vnodes.shape
    n_fnode = fnodes.shape
    if n_fnode[0] < n_fnode[1]:
        fnodes = fnodes.transpose()
        n_fnode = fnodes.shape

    m = np.zeros((n_vnode[0] * 3, n_fnode[0] * 3))
    for i0 in range(n_vnode[0]):
        delta_xi = fnodes - vnodes[i0]
        temp1 = delta_xi ** 2
        delta_r2 = temp1.sum(axis=1) + delta ** 2  # delta_r2 = r^2+e^2
        delta_r3 = delta_r2 ** 1.5  # delta_r3 = (r^2+e^2)^1.5
        temp2 = (delta_r2 + delta ** 2) / delta_r3  # temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
        m[3 * i0, 0::3] = temp2 + delta_xi[:, 0] * delta_xi[:, 0] / delta_r3  # Mxx
        m[3 * i0 + 1, 1::3] = temp2 + delta_xi[:, 1] * delta_xi[:, 1] / delta_r3  # Myy
        m[3 * i0 + 2, 2::3] = temp2 + delta_xi[:, 2] * delta_xi[:, 2] / delta_r3  # Mzz
        m[3 * i0, 1::3] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3  # Mxy
        m[3 * i0, 2::3] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3  # Mxz
        m[3 * i0 + 1, 2::3] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3  # Myz
        m[3 * i0 + 1, 0::3] = m[3 * i0, 1::3]  # Myx
        m[3 * i0 + 2, 0::3] = m[3 * i0, 2::3]  # Mzx
        m[3 * i0 + 2, 1::3] = m[3 * i0 + 1, 2::3]  # Mzy
    return m  # 'regularized stokeslets matrix, U = M * F'
