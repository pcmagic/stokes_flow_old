import numpy as np


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
    nodes = np.zeros((n, 3))
    nodes[:, 0] = headC * xlocH
    nodes[:, 1] = headA * tsin * np.cos(spherePhi)
    nodes[:, 2] = headA * tsin * np.sin(spherePhi)

    u = np.zeros((n, 3))
    u[:, 0] = sphereU[0] + sphereU[4]*nodes[:, 2] +- sphereU[5]*nodes[:, 1]
    u[:, 1] = sphereU[1] + sphereU[5]*nodes[:, 0] +- sphereU[3]*nodes[:, 2]
    u[:, 2] = sphereU[2] + sphereU[3]*nodes[:, 1] +- sphereU[4]*nodes[:, 0]

    return nodes, u

def tunnel(deltaLength: float,      # length of the mesh
           length: float,           # length of the tunnel
           radius: float):          # radius of the tunnel
    a = np.arange(0, 2 * np.pi, deltaLength / radius)
    x, y = np.cos(a) * radius, np.sin(a) * radius
    z = np.arange(-length / 2, length / 2, deltaLength)
    n_a, n_z = a.size, z.size

    nodes = np.zeros((n_a*n_z, 3))
    nodes[:, 1] = np.tile(x, (n_z, 1)).reshape(-1, 1).flatten(order='F')
    nodes[:, 2] = np.tile(y, (n_z, 1)).reshape(-1, 1).flatten(order='F')
    nodes[:, 0] = np.tile(z, n_a).reshape(n_a, -1).flatten(order='F')

    u = np.zeros(nodes.shape)

    return nodes, u
