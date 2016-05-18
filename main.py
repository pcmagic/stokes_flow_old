# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410



# import petsc4py
# petsc4py.init()
import numpy as np

import stokes_flow as sf

move_dist = [1, 0, 0]
method = 'rs'
delta = 6.e-4
solve_method = 'gmres'
precondition_method = 'none'
field_range = np.array([[-1, -1, -1], [1, 1, 3]])
n_grid = np.array([50, 50, 150])
filename = "./helixInfor"

# nodes, velocity, delta = sf.import_mat()
problem = sf.StokesFlowComponent()
obj1 = sf.StokesFlowObject(problem, filename=filename + '.mat', index=1)
obj_list = [obj1]
problem.add_obj(obj1)
for i in range(0):
    obj2 = obj_list[-1].copy()
    obj_list.append(obj2)
    obj2.move(move_dist)
    problem.add_obj(obj2)

problem_arguments = {'method': method,
                     'delta': delta}
problem.create_mtrix(**problem_arguments)
problem.solve(solve_method, precondition_method)
problem.vtk_force(filename + 'Force')
problem.vtk_velocity(filename + 'Velocity', field_range, n_grid)

# import numpy_io as nio
# nio.write(velocity, 'velocity.txt')
# nio.write(rs_m, 'rs_m.txt')
# nio.write(force, 'force.txt')



pass
