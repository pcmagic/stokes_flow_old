import numpy as np
from sf_error import sf_error
import scipy.io as sio


def sphere(n: int,                  # number of nodes.
           headA: float,            # major axis = 2*headA
           headC: float,            # minor axis = 2*headC
           sphereU: np.array):      # [u1, u2, u3, w1, w2, w3], velocity and angular velocity.
    jj = np.arange(n)
    xlocH = -1 + 2*jj/(n-1)
    numf = 0.5

    prefac = 3.6*np.sqrt(headC/headA)
    spherePhi = np.ones(n)
    for i0 in range(0, n):
        if i0 == 0 or i0 == n-1:
            spherePhi[i0] = 0
        else:
            tr = np.sqrt(1-xlocH[i0]**2)
            wgt = prefac * ( 1 - numf*(1-tr) ) / tr
            spherePhi[i0] = (spherePhi[i0-1] + wgt/np.sqrt(n)) % 2*np.pi

    tsin = np.sqrt(1 - xlocH**2)
    nodes = np.zeros((n, 3), order='F')
    nodes[:, 0] = headC * xlocH
    nodes[:, 1] = headA * tsin * np.cos(spherePhi)
    nodes[:, 2] = headA * tsin * np.sin(spherePhi)

    u = np.zeros((n, 3), order = 'F')
    u[:, 0] = sphereU[0] + sphereU[4]*nodes[:, 2] - sphereU[5]*nodes[:, 1]
    u[:, 1] = sphereU[1] + sphereU[5]*nodes[:, 0] - sphereU[3]*nodes[:, 2]
    u[:, 2] = sphereU[2] + sphereU[3]*nodes[:, 1] - sphereU[4]*nodes[:, 0]
    normal = norm_sphere(nodes)
    return nodes, u, normal


def norm_sphere(nodes):
    normal = np.zeros((nodes.shape[0], 2))       # {Sin[a] Sin[b], -Cos[a] Sin[b], Cos[b]} = {n1, n2, n3} is the normal vector
    normal_vector = nodes / np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2 + nodes[:, 2]**2).reshape(nodes.shape[0], 1)
    normal[:, 1] = np.arccos(normal_vector[:, 2])                           # b
    normal[:, 0] = np.arcsin(normal_vector[:, 0] / np.sin(normal[:, 1]))    # a
    return normal


def tunnel(deltaLength: float,      # length of the mesh
           length: float,           # length of the tunnel
           radius: float):          # radius of the tunnel
    a = np.arange(0, 2 * np.pi - deltaLength / radius / 2, deltaLength / radius)
    x, y = np.cos(a) * radius, np.sin(a) * radius
    z = np.arange(-length / 2, length / 2, deltaLength)
    n_a, n_z = a.size, z.size

    nodes = np.zeros((n_a*n_z, 3), order = 'F')
    nodes[:, 1] = np.tile(x, (n_z, 1)).reshape(-1, 1).flatten(order='F')
    nodes[:, 2] = np.tile(y, (n_z, 1)).reshape(-1, 1).flatten(order='F')
    nodes[:, 0] = np.tile(z, n_a).reshape(n_a, -1).flatten(order='F')

    u = np.zeros(nodes.shape, order='F')
    normal = norm_tunnel(nodes)
    return nodes, u, normal


def norm_tunnel(nodes):
    normal = np.zeros((nodes.shape[0], 2))       # {Sin[a] Sin[b], -Cos[a] Sin[b], Cos[b]} = {n1, n2, n3} is the normal vector
    normal_vector = -1 * nodes / np.sqrt(nodes[:, 1]**2 + nodes[:, 2]**2).reshape(nodes.shape[0], 1)        # -1 means swap direction
    normal[:, 1] = np.arccos(normal_vector[:, 2])   # b
    normal[:, 0] = 0                                # a
    return normal

def mat_nodes(filename: str = '..'):
    if filename == '..':
        ierr = -1
        err_msg = 'wrong mat file name. '
        raise sf_error(ierr, err_msg)

    mat_contents = sio.loadmat(filename)
    origin = mat_contents['origin'].astype(np.float)
    f_nodes = mat_contents['f_nodes'].astype(np.float, order = 'F')
    u_nodes = mat_contents['u_nodes'].astype(np.float, order = 'F')
    return f_nodes, u_nodes, origin



def mat_velocity(filename: str = '..'):
    if filename == '..':
        ierr = -1
        err_msg = 'wrong mat file name. '
        raise sf_error(ierr, err_msg)

    mat_contents = sio.loadmat(filename)
    velocity = mat_contents['U'].astype(np.float, order = 'F')
    return velocity