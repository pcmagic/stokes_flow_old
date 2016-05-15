# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410



import numpy as np
from evtk.hl import pointsToVTK
# import petsc4py
# petsc4py.init()
import stokes_flow as sf

move_dist =[1, 0, 0]
method = 'rs'
delta = 6.e-4
solve_method = 'gmres'
precondition_method = 'none'
filename = "./surfaceForce"

# nodes, velocity, delta = sf.import_mat()
problem = sf.StokesFlowComponent()
obj1 = sf.StokesFlowObject(problem, filename='./helixInfor.mat', index=1)
obj_list = [obj1]
for i in range(30):
    obj_list.append(obj_list[-1].copy())
    obj_list[-1].move(move_dist)
for obj in obj_list:
    problem.add_obj(obj)

problem_arguments = {'method': method,
                     'delta': delta}
problem.create_mtrix(**problem_arguments)
problem.solve(solve_method, precondition_method)
problem.vtk_force(filename)

# import numpy_io as nio
# nio.write(velocity, 'velocity.txt')
# nio.write(rs_m, 'rs_m.txt')
# nio.write(force, 'force.txt')



pass
