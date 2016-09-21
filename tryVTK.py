from evtk.hl import pointsToVTK, gridToVTK
import numpy as np


n_obj_x = 1
n_obj_y = 1
move_delta = np.array([3, 3, 1])
filename = "./tryVTK"
field_range = np.array([[-3, -3, -3], [n_obj_x - 1, n_obj_y - 1, 0] * move_delta + [3, 3, 3]])
n_grid = np.array([3, 3, 3])
min_range = np.min(field_range, 0)
max_range = np.max(field_range, 0)
n_node = n_grid[0] * n_grid[1] * n_grid[2]
n_para = 3 * n_node
full_region_x = np.linspace(min_range[0], max_range[0], n_grid[0])
full_region_y = np.linspace(min_range[1], max_range[1], n_grid[1])
full_region_z = np.linspace(min_range[2], max_range[2], n_grid[2])
temp_x, temp_y, temp_z = np.meshgrid(full_region_x, full_region_y, full_region_z, indexing='ij')
# velocity_nodes = np.c_[temp_x.ravel(), temp_y.ravel(), temp_z.ravel()]

u_x = temp_x.reshape((n_grid[0], n_grid[1], n_grid[2]))
u_y = temp_y.reshape((n_grid[0], n_grid[1], n_grid[2]))
u_z = temp_z.reshape((n_grid[0], n_grid[1], n_grid[2]))
# output data
delta = (max_range - min_range) / n_grid
full_region_x = np.linspace(min_range[0] - delta[0] / 2, max_range[0] + delta[0] / 2, n_grid[0] + 1)
full_region_y = np.linspace(min_range[1] - delta[1] / 2, max_range[1] + delta[1] / 2, n_grid[1] + 1)
full_region_z = np.linspace(min_range[2] - delta[2] / 2, max_range[2] + delta[2] / 2, n_grid[2] + 1)
temp_x, temp_y, temp_z = np.meshgrid(full_region_x, full_region_y, full_region_z, indexing='ij')
gridToVTK(filename, temp_x, temp_y, temp_z,
          cellData={"velocity": (u_x, u_y, u_z)})