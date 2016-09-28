import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

deltaLength = 0.1
length = 1
radius = 0.3
a = np.arange(0, 2 * np.pi, deltaLength / radius)
x, y = np.cos(a) * radius, np.sin(a) * radius
z = np.arange(-length / 2, length / 2, deltaLength)
n_a, n_z = a.size, z.size

nodes = np.zeros((n_a * n_z, 3))
nodes[:, 0] = np.tile(x, (n_z, 1)).reshape(-1, 1).flatten(order='F')
nodes[:, 1] = np.tile(y, (n_z, 1)).reshape(-1, 1).flatten(order='F')
nodes[:, 2] = np.tile(z, n_a).reshape(n_a, -1).flatten(order='F')

# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], linestyle='None', marker='.')
ax.axis(v='equal')
plt.show()

pass