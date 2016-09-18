import numpy as np
from petsc4py import PETSc
from sf_error import sf_error
from mpi4py import MPI


def regularized_stokeslets_matrix_3d(vnodes: np.ndarray,  # nodes contain velocity information
                                     fnodes: np.ndarray,  # nodes contain force information
                                     delta: float):  # correction factor
    # Solve m matrix using regularized stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets[J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        n_vnode = vnodes.shape
        if n_vnode[0] < n_vnode[1]:
            vnodes = vnodes.transpose()
            n_vnode = vnodes.shape
        n_fnode = fnodes.shape
        if n_fnode[0] < n_fnode[1]:
            fnodes = fnodes.transpose()
            n_fnode = fnodes.shape

        split = np.array_split(vnodes, size)
        split_size_in = [len(split[i])*3 for i in range(len(split))]
        split_disp_in = np.insert(np.cumsum(split_size_in), 0, 0)[0:-1]
        split_size_out = [len(split[i])*3*n_fnode[0]*3 for i in range(len(split))]
        split_disp_out = np.insert(np.cumsum(split_size_out), 0, 0)[0:-1]
    else:
        fnodes = None
        vnodes = None
        split = None
        split_disp_in = None
        split_size_in = None
        split_disp_out = None
        split_size_out = None
        n_fnode = None
        n_vnode = None

    split_size_in = comm.bcast(split_size_in, root=0)
    split_disp_in = comm.bcast(split_disp_in, root=0)
    split_size_out = comm.bcast(split_size_out, root=0)
    split_disp_out = comm.bcast(split_disp_out, root=0)
    vnodes_local = np.zeros(split_size_in[rank], dtype='float64')
    comm.Scatterv([vnodes, split_size_in, split_disp_in, MPI.DOUBLE], vnodes_local, root=0)
    vnodes_local = vnodes_local.reshape((3, -1)).T
    n_vnode_local = len(vnodes_local)
    n_fnode_local = comm.bcast(n_fnode, root=0)
    if rank == 0:
        fnodes_local = fnodes
    else:
        fnodes_local = np.zeros(n_fnode_local, dtype='float64')
    comm.Bcast(fnodes_local, root=0)

    m_local = np.zeros((n_vnode_local*3, n_fnode_local[0]*3))
    for i0 in range(n_vnode_local):
        delta_xi = fnodes_local - vnodes_local[i0]
        temp1_local = delta_xi ** 2
        delta_2 = np.square(delta)  # delta_2 = e^2
        delta_r2 = temp1_local.sum(axis=1) + delta_2  # delta_r2 = r^2+e^2
        delta_r3 = delta_r2 * np.sqrt(delta_r2)  # delta_r3 = (r^2+e^2)^1.5
        temp2 = (delta_r2 + delta_2 ) / delta_r3  # temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
        m_local[3 * i0, 0::3] = temp2 + np.square(delta_xi[:, 0]) / delta_r3  # Mxx
        m_local[3 * i0 + 1, 1::3] = temp2 + np.square(delta_xi[:, 1]) / delta_r3  # Myy
        m_local[3 * i0 + 2, 2::3] = temp2 + np.square(delta_xi[:, 2]) / delta_r3  # Mzz
        m_local[3 * i0 + 1, 0::3] = m_local[3 * i0, 1::3] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3  # Mxy
        m_local[3 * i0 + 2, 0::3] = m_local[3 * i0, 2::3] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3  # Mxz
        m_local[3 * i0 + 2, 1::3] = m_local[3 * i0 + 1, 2::3] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3  # Myz

    if rank == 0:
        m = np.zeros((n_vnode[0]*3, n_fnode[0]*3))
    else:
        m = None
    comm.Gatherv(m_local, [m, split_size_out, split_disp_out, MPI.DOUBLE], root=0)
    return m  # ' regularized stokeslets matrix, U = M * F '

def regularized_stokeslets_matrix_3d_petsc(vnodes: np.ndarray,  # nodes contain velocity information
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

    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    m.setSizes(((None, n_vnode[0]*3), (None, n_fnode[0]*3)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    fnodes_pc, vnodes_pc = m.createVecs()         # F and U vectors in petsc form.
    fnodes_pc = fnodes.reshape([-1, 1])[:]
    vnodes_pc = vnodes.reshape([-1, 1])[:]
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):
        delta_xi = fnodes - vnodes[i0//3]
        temp1 = delta_xi ** 2
        delta_2 = np.square(delta)  # delta_2 = e^2
        delta_r2 = temp1.sum(axis=1) + delta_2  # delta_r2 = r^2+e^2
        delta_r3 = delta_r2 * np.sqrt(delta_r2)  # delta_r3 = (r^2+e^2)^1.5
        temp2 = (delta_r2 + delta_2) / delta_r3  # temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
        temp2 = (delta_r2 + delta_2) / delta_r3  # temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
        if i0 % 3 == 0:       # x axis
            m[i0, 0::3] = temp2 + np.square(delta_xi[:, 0]) / delta_r3  # Mxx
            m[i0, 1::3] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3  # Mxy
            m[i0, 2::3] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3  # Mxz
        elif i0 % 3 == 1:     # y axis
            m[i0, 0::3] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3  # Mxy
            m[i0, 1::3] = temp2 + np.square(delta_xi[:, 1]) / delta_r3  # Myy
            m[i0, 2::3] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3  # Myz
        # elif i0 % 3 == 2:     # z axis
        else:     # z axis
            m[i0, 0::3] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3  # Mxz
            m[i0, 1::3] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3  # Myz
            m[i0, 2::3] = temp2 + np.square(delta_xi[:, 2]) / delta_r3  # Mzz
    m.assemble()

    return m  # ' regularized stokeslets matrix, U = M * F '


def check_regularized_stokeslets_matrix_3d(**kwargs):
    if not ('delta' in kwargs):
        ierr = 301
        err_msg = 'the reguralized stokeslets method needs parameter, delta. '
        raise sf_error(ierr, err_msg)
    if len(kwargs) > 1:
        ierr = 302
        err_msg = 'only one parameter, delta, is accepted for the reguralized stokeslets method. '
        raise sf_error(ierr, err_msg)


#TODO: theory of the method have some problem, now.
def surface_force_matrix_3d(vnodes: np.ndarray,  # nodes contain velocity information
                            fnodes: np.ndarray,  # nodes contain force information
                            d_radia: float):  # the radia of the integral surface.

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
        temp0 = delta_xi ** 2
        delta_r2 = temp0.sum(axis=1)  # delta_r2 = r^2
        delta_r = delta_r2 ** 0.5  # delta_r = r
        temp1 = 1 / (8 * np.pi * delta_r)
        temp2 = 1 / (8 * np.pi * delta_r * delta_r2)
        m[3 * i0, 0::3] = temp1 + delta_xi[:, 0] * delta_xi[:, 0] * temp2  # Mxx
        m[3 * i0 + 1, 1::3] = temp1 + delta_xi[:, 1] * delta_xi[:, 1] * temp2  # Myy
        m[3 * i0 + 2, 2::3] = temp1 + delta_xi[:, 2] * delta_xi[:, 2] * temp2  # Mzz
        m[3 * i0, 1::3] = delta_xi[:, 0] * delta_xi[:, 1] * temp2  # Mxy
        m[3 * i0, 2::3] = delta_xi[:, 0] * delta_xi[:, 2] * temp2  # Mxz
        m[3 * i0 + 1, 2::3] = delta_xi[:, 1] * delta_xi[:, 2] * temp2  # Myz
        m[3 * i0 + 1, 0::3] = m[3 * i0, 1::3]  # Myx
        m[3 * i0 + 2, 0::3] = m[3 * i0, 2::3]  # Mzx
        m[3 * i0 + 2, 1::3] = m[3 * i0 + 1, 2::3]  # Mzy
    for i0 in range(n_vnode[0]):
        m[3 * i0, 3 * i0] = (1 + np.pi) * np.pi * delta_r ** 2
        m[3 * i0 + 1, 3 * i0 + 1] = (1 + np.pi) * np.pi * delta_r ** 2
        m[3 * i0 + 2, 3 * i0 + 2] = np.pi * delta_r ** 2

    return m  # 'regularized stokeslets matrix, U = M * F'


def check_surface_force_matrix_3d(**kwargs):
    if len(kwargs) > 0:
        ierr = 303
        err_msg = 'no parameter, is accepted for the surface force stokeslets method. '
        raise sf_error(ierr, err_msg)
