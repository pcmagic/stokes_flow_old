# functions for solving stokes flow using regularised stokeslets (and its improved) method.
# Zhang Ji, 20160409

import scipy.io as sio
import numpy as np


# Import geometries from Matlab file.
def import_mat():
    mat_contents = sio.loadmat('helixInfor.mat')
    nodes = mat_contents['nodes']
    velocity = mat_contents['U']
    velocity = velocity.flatten(1)
    delta = mat_contents['delta']
    delta = delta[0, 0]
    return nodes, velocity, delta


def regularized_stokeslets_matrix_3d(vnodes: np.ndarray = 'nodes contain velocity information',
                                     fnodes: np.ndarray = 'nodes contain force information',
                                     delta: float = 'correction factor'):
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
