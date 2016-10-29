import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import stokes_flow as sf
from sf_error import sf_error


def regularized_stokeslets_matrix_3d(vnodes: np.ndarray,  # nodes contain velocity information
                                     fnodes: np.ndarray,  # nodes contain force information
                                     delta: float):  # correction factor
    # Solve m matrix using regularized stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets[J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.

    comm = PETSc.COMM_WORLD.tompi4py()
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
        split_size_in = [len(split[i]) * 3 for i in range(len(split))]
        split_disp_in = np.insert(np.cumsum(split_size_in), 0, 0)[0:-1]
        split_size_out = [len(split[i]) * 3 * n_fnode[0] * 3 for i in range(len(split))]
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

    m_local = np.zeros((n_vnode_local * 3, n_fnode_local[0] * 3))
    for i0 in range(n_vnode_local):
        delta_xi = fnodes_local - vnodes_local[i0]
        temp1_local = delta_xi ** 2
        delta_2 = np.square(delta)  # delta_2 = e^2
        delta_r2 = temp1_local.sum(axis=1) + delta_2  # delta_r2 = r^2+e^2
        delta_r3 = delta_r2 * np.sqrt(delta_r2)  # delta_r3 = (r^2+e^2)^1.5
        temp2 = (delta_r2 + delta_2) / delta_r3  # temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
        m_local[3 * i0, 0::3] = temp2 + np.square(delta_xi[:, 0]) / delta_r3  # Mxx
        m_local[3 * i0 + 1, 1::3] = temp2 + np.square(delta_xi[:, 1]) / delta_r3  # Myy
        m_local[3 * i0 + 2, 2::3] = temp2 + np.square(delta_xi[:, 2]) / delta_r3  # Mzz
        m_local[3 * i0 + 1, 0::3] = m_local[3 * i0, 1::3] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3  # Mxy
        m_local[3 * i0 + 2, 0::3] = m_local[3 * i0, 2::3] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3  # Mxz
        m_local[3 * i0 + 2, 1::3] = m_local[3 * i0 + 1, 2::3] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3  # Myz

    if rank == 0:
        m = np.zeros((n_vnode[0] * 3, n_fnode[0] * 3))
    else:
        m = None
    comm.Gatherv(m_local, [m, split_size_out, split_disp_out, MPI.DOUBLE], root=0)
    return m  # ' regularized stokeslets matrix, U = M * F '


def regularized_stokeslets_matrix_3d_petsc(obj1: sf.stokesFlowObject,  # objct contain velocity information
                                           obj2: sf.stokesFlowObject,  # objct contain force information
                                           **kwargs):
    # Solve m matrix using regularized stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets[J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.

    delta = kwargs['delta']  # correction factor
    vnodes = obj1.get_f_nodes()
    fnodes = obj2.get_f_nodes()
    n_vnode = vnodes.shape
    if n_vnode[0] < n_vnode[1]:
        vnodes = vnodes.transpose()
        n_vnode = vnodes.shape
    n_fnode = fnodes.shape
    if n_fnode[0] < n_fnode[1]:
        fnodes = fnodes.transpose()
        n_fnode = fnodes.shape

    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    m.setSizes(((None, n_vnode[0] * 3), (None, n_fnode[0] * 3)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):
        delta_xi = fnodes - vnodes[i0 // 3]
        temp1 = delta_xi ** 2
        delta_2 = np.square(delta)  # delta_2 = e^2
        delta_r2 = temp1.sum(axis=1) + delta_2  # delta_r2 = r^2+e^2
        delta_r3 = delta_r2 * np.sqrt(delta_r2)  # delta_r3 = (r^2+e^2)^1.5
        temp2 = (delta_r2 + delta_2) / delta_r3  # temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
        if i0 % 3 == 0:  # x axis
            m[i0, 0::3] = (temp2 + np.square(delta_xi[:, 0]) / delta_r3) / (8 * np.pi)  # Mxx
            m[i0, 1::3] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3 / (8 * np.pi)  # Mxy
            m[i0, 2::3] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Mxz
        elif i0 % 3 == 1:  # y axis
            m[i0, 0::3] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3 / (8 * np.pi)  # Mxy
            m[i0, 1::3] = (temp2 + np.square(delta_xi[:, 1]) / delta_r3) / (8 * np.pi)  # Myy
            m[i0, 2::3] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Myz
        else:  # z axis
            m[i0, 0::3] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Mxz
            m[i0, 1::3] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Myz
            m[i0, 2::3] = (temp2 + np.square(delta_xi[:, 2]) / delta_r3) / (8 * np.pi)  # Mzz
    m.assemble()

    return m  # ' regularized stokeslets matrix, U = M * F '


def surf_force_matrix_3d_debug(obj1: sf.stokesFlowObject,  # objct contain velocity information
                               obj2: sf.stokesFlowObject,  # objct contain force information
                               **kwargs):
    # Solve m matrix using regularized stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets[J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.

    delta = kwargs['delta']  # correction factor
    d_radia = kwargs['d_radia']  # the radial of the integral surface.
    vnodes = obj1.get_f_nodes()
    fnodes = obj2.get_f_nodes()
    n_vnode = vnodes.shape
    if n_vnode[0] < n_vnode[1]:
        vnodes = vnodes.transpose()
        n_vnode = vnodes.shape
    n_fnode = fnodes.shape
    if n_fnode[0] < n_fnode[1]:
        fnodes = fnodes.transpose()
        n_fnode = fnodes.shape

    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    m.setSizes(((None, n_vnode[0] * 3), (None, n_fnode[0] * 3)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):
        delta_xi = fnodes - vnodes[i0 // 3]
        temp1 = delta_xi ** 2
        delta_2 = np.square(delta)  # delta_2 = e^2
        delta_r2 = temp1.sum(axis=1) + delta_2  # delta_r2 = r^2+e^2
        delta_r3 = delta_r2 * np.sqrt(delta_r2)  # delta_r3 = (r^2+e^2)^1.5
        temp2 = (delta_r2 + delta_2) / delta_r3  # temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
        if i0 % 3 == 0:  # x axis
            m[i0, 0::3] = (temp2 + np.square(delta_xi[:, 0]) / delta_r3) / (8 * np.pi)  # Mxx
            m[i0, 1::3] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3 / (8 * np.pi)  # Mxy
            m[i0, 2::3] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Mxz
        elif i0 % 3 == 1:  # y axis
            m[i0, 0::3] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3 / (8 * np.pi)  # Mxy
            m[i0, 1::3] = (temp2 + np.square(delta_xi[:, 1]) / delta_r3) / (8 * np.pi)  # Myy
            m[i0, 2::3] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Myz
        else:  # z axis
            m[i0, 0::3] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Mxz
            m[i0, 1::3] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Myz
            m[i0, 2::3] = (temp2 + np.square(delta_xi[:, 2]) / delta_r3) / (8 * np.pi)  # Mzz
    if obj1 == obj2:  # self-interaction
        for i0 in range(m_start, m_end):
            i1 = i0 // 3
            norm = obj1.get_norm()[i1, :]
            if i0 % 3 == 0:  # x axis
                m[i0, i0 + 0] = (
                                    3 * np.cos(norm[0]) ** 2 + 1 / 2 * (5 + np.cos(2 * norm[1])) * np.sin(
                                        norm[0]) ** 2) / (
                                    8 * np.pi * d_radia)  # Mxx
                m[i0, i0 + 1] = (np.cos(norm[0]) * np.sin(norm[0]) * np.sin(norm[1]) ** 2) / (
                    8 * np.pi * d_radia)  # Mxy
                m[i0, i0 + 2] = (-np.cos(norm[1]) * np.sin(norm[0]) * np.sin(norm[1])) / (8 * np.pi * d_radia)  # Mxz
            elif i0 % 3 == 1:  # y axis
                m[i0, i0 - 1] = (np.cos(norm[0]) * np.sin(norm[0]) * np.sin(norm[1]) ** 2) / (
                    8 * np.pi * d_radia)  # Mxy
                m[i0, i0 + 0] = (1 / 8 * (
                    22 - 2 * np.cos(2 * norm[0]) + np.cos(2 * (norm[0] - norm[1])) + 2 * np.cos(2 * norm[1]) + np.cos(
                        2 * (norm[0] + norm[1])))) / (8 * np.pi * d_radia)  # Myy
                m[i0, i0 + 1] = (np.cos(norm[0]) * np.cos(norm[1]) * np.sin(norm[1])) / (8 * np.pi * d_radia)  # Myz
            else:  # z axis
                m[i0, i0 - 2] = (-np.cos(norm[1]) * np.sin(norm[0]) * np.sin(norm[1])) / (8 * np.pi * d_radia)  # Mxz
                m[i0, i0 - 1] = (np.cos(norm[0]) * np.cos(norm[1]) * np.sin(norm[1])) / (8 * np.pi * d_radia)  # Myz
                m[i0, i0 + 0] = (1 / 2 * (5 - np.cos(2 * norm[1]))) / (8 * np.pi * d_radia)  # Mzz
    m.assemble()
    return m  # ' regularized stokeslets matrix, U = M * F '


def check_regularized_stokeslets_matrix_3d(**kwargs):
    if not ('delta' in kwargs):
        ierr = 301
        err_msg = 'the reguralized stokeslets method needs parameter, delta. '
        raise sf_error(ierr, err_msg)


def surf_force_matrix_3d(obj1: sf.surf_forceObj,  # objct contain velocity information
                         obj2: sf.surf_forceObj,  # objct contain force information
                         **kwargs):
    # Solve m matrix using surface force distribution method
    # U = M * F.
    # details see my notes, 面力分布法
    # Zhang Ji, 20160928

    d_radia = kwargs['d_radia']  # the radial of the integral surface.
    vnodes = obj1.get_f_nodes()
    fnodes = obj2.get_f_nodes()
    n_vnode = vnodes.shape
    if n_vnode[0] < n_vnode[1]:
        vnodes = vnodes.transpose()
        n_vnode = vnodes.shape
    n_fnode = fnodes.shape
    if n_fnode[0] < n_fnode[1]:
        fnodes = fnodes.transpose()
        n_fnode = fnodes.shape

    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    m.setSizes(((None, n_vnode[0] * 3), (None, n_fnode[0] * 3)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):  # interaction between different nodes.
        delta_xi = fnodes - vnodes[i0 // 3]  # [delta_x, delta_y, delta_z]
        temp1 = delta_xi ** 2
        delta_r2 = temp1.sum(axis=1)  # r^2
        if obj1 == obj2:  # self-interaction will be solved later
            delta_r2[i0 // 3] = 1
        delta_r1 = delta_r2 ** 0.5  # r^1
        delta_r3 = delta_r2 * delta_r1  # r^3
        temp2 = 1 / delta_r1  # 1/r
        if i0 % 3 == 0:  # x axis
            m[i0, 0::3] = (temp2 + delta_xi[:, 0] * delta_xi[:, 0] / delta_r3) / (8 * np.pi)  # Mxx
            m[i0, 1::3] = (delta_xi[:, 0] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Mxy
            m[i0, 2::3] = (delta_xi[:, 0] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mxz
        elif i0 % 3 == 1:  # y axis
            m[i0, 0::3] = (delta_xi[:, 0] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Mxy
            m[i0, 1::3] = (temp2 + delta_xi[:, 1] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Myy
            m[i0, 2::3] = (delta_xi[:, 1] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Myz
        else:  # z axis
            m[i0, 0::3] = (delta_xi[:, 0] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mxz
            m[i0, 1::3] = (delta_xi[:, 1] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Myz
            m[i0, 2::3] = (temp2 + delta_xi[:, 2] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mzz
    if obj1 == obj2:  # self-interaction
        for i0 in range(m_start, m_end):
            i1 = i0 // 3
            norm = obj1.get_norm()[i1, :]
            if i0 % 3 == 0:  # x axis
                m[i0, i0 + 0] = (
                                    3 * np.cos(norm[0]) ** 2 + 1 / 2 * (5 + np.cos(2 * norm[1])) * np.sin(
                                        norm[0]) ** 2) / (
                                    8 * np.pi * d_radia)  # Mxx
                m[i0, i0 + 1] = (np.cos(norm[0]) * np.sin(norm[0]) * np.sin(norm[1]) ** 2) / (
                    8 * np.pi * d_radia)  # Mxy
                m[i0, i0 + 2] = (-np.cos(norm[1]) * np.sin(norm[0]) * np.sin(norm[1])) / (8 * np.pi * d_radia)  # Mxz
            elif i0 % 3 == 1:  # y axis
                m[i0, i0 - 1] = (np.cos(norm[0]) * np.sin(norm[0]) * np.sin(norm[1]) ** 2) / (
                    8 * np.pi * d_radia)  # Mxy
                m[i0, i0 + 0] = (1 / 8 * (
                    22 - 2 * np.cos(2 * norm[0]) + np.cos(2 * (norm[0] - norm[1])) + 2 * np.cos(2 * norm[1]) + np.cos(
                        2 * (norm[0] + norm[1])))) / (8 * np.pi * d_radia)  # Myy
                m[i0, i0 + 1] = (np.cos(norm[0]) * np.cos(norm[1]) * np.sin(norm[1])) / (8 * np.pi * d_radia)  # Myz
            else:  # z axis
                m[i0, i0 - 2] = (-np.cos(norm[1]) * np.sin(norm[0]) * np.sin(norm[1])) / (8 * np.pi * d_radia)  # Mxz
                m[i0, i0 - 1] = (np.cos(norm[0]) * np.cos(norm[1]) * np.sin(norm[1])) / (8 * np.pi * d_radia)  # Myz
                m[i0, i0 + 0] = (1 / 2 * (5 - np.cos(2 * norm[1]))) / (8 * np.pi * d_radia)  # Mzz
    m.assemble()
    # import matplotlib.pyplot as plt
    # M = m.getDenseArray()
    # fig, ax = plt.subplots()
    # cax = ax.matshow(M, origin='lower')
    # fig.colorbar(cax)
    # plt.show()
    return m  # ' regularized stokeslets matrix, U = M * F '


def check_surf_force_matrix_3d(**kwargs):
    if not ('d_radia' in kwargs):
        ierr = 301
        err_msg = 'the surface force method needs parameter, d_radia, the radial of the integral surface. '
        raise sf_error(ierr, err_msg)


def point_source_matrix_3d_petsc(obj1: sf.stokesFlowObject,  # objct contain velocity information
                                 obj2: sf.stokesFlowObject,  # objct contain force information
                                 **kwargs):
    # Solve m matrix using regularized stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets[J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.

    vnodes = obj1.get_u_nodes()
    n_vnode = obj1.get_n_velocity()
    fnodes = obj2.get_f_nodes()
    n_fnode = obj2.get_n_force()

    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    num_unknown = 4
    m.setSizes(((None, n_vnode), (None, n_fnode)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):  # interaction between different nodes.
        delta_xi = fnodes - vnodes[i0 // num_unknown]  # [delta_x, delta_y, delta_z]
        temp1 = delta_xi ** 2
        delta_r2 = temp1.sum(axis=1)  # r^2
        delta_r1 = delta_r2 ** 0.5  # r^1
        delta_r3 = delta_r2 * delta_r1  # r^3
        temp2 = 1 / delta_r1  # 1/r
        if i0 % 3 == 0:  # velocity x axis
            m[i0, 0::num_unknown] = (temp2 + delta_xi[:, 0] * delta_xi[:, 0] / delta_r3) / (8 * np.pi)  # Mxx
            m[i0, 1::num_unknown] = (delta_xi[:, 0] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Mxy
            m[i0, 2::num_unknown] = (delta_xi[:, 0] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mxz
            m[i0, 3::num_unknown] = delta_xi[:, 0] / delta_r3 / (4 * np.pi)  # Mx_source
        elif i0 % 3 == 1:  # velocity y axis
            m[i0, 0::num_unknown] = (delta_xi[:, 0] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Mxy
            m[i0, 1::num_unknown] = (temp2 + delta_xi[:, 1] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Myy
            m[i0, 2::num_unknown] = (delta_xi[:, 1] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Myz
            m[i0, 3::num_unknown] = delta_xi[:, 1] / delta_r3 / (4 * np.pi)  # My_source
        elif i0 % 3 == 2:  # velocity z axis
            m[i0, 0::num_unknown] = (delta_xi[:, 0] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mxz
            m[i0, 1::num_unknown] = (delta_xi[:, 1] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Myz
            m[i0, 2::num_unknown] = (temp2 + delta_xi[:, 2] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mzz
            m[i0, 3::num_unknown] = delta_xi[:, 2] / delta_r3 / (4 * np.pi)  # Mz_source
    m.assemble()

    return m  # ' regularized stokeslets matrix, U = M * F '


def check_point_source_matrix_3d_petsc(**kwargs):
    if not ('delta' in kwargs):
        ierr = 301
        err_msg = 'the reguralized stokeslets method needs parameter, delta. '
        raise sf_error(ierr, err_msg)
