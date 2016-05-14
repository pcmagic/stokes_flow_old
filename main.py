# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410



# import numpy as np
from evtk.hl import pointsToVTK
# import petsc4py
# petsc4py.init()
from petsc4py import PETSc
import stokes_flow as sf
import copy

# nodes, velocity, delta = sf.import_mat()
problem = sf.StokesFlowComponent()
obj1 = sf.StokesFlowObject(problem, filename='./helixInfor.mat', index=1)
obj2 = obj1.copy(new_index=2)
obj2.move(obj1.get_origin()+ [10, 10, 0])
problem.add_obj(obj1)
problem.add_obj(obj2)
problem.collect_nodes()

problem.set_method(sf.regularized_stokeslets_matrix_3d)
# proble
rs_m = method(nodes, nodes, delta)

pc_velocity = PETSc.Vec().createWithArray(velocity)
pc_rs_m = PETSc.Mat().createDense(size=rs_m.shape, array=rs_m)
pc_force = pc_rs_m.getVecRight()
pc_force.set(0)

ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setType('gmres')
ksp.getPC().setType('none')
ksp.setOperators(pc_rs_m)
ksp.setFromOptions()
ksp.solve(pc_velocity, pc_force)
force = pc_force.getArray()

# import numpy_io as nio
# nio.write(velocity, 'velocity.txt')
# nio.write(rs_m, 'rs_m.txt')
# nio.write(force, 'force.txt')

force_x = force[0::3].ravel()
force_y = force[1::3].ravel()
force_z = force[2::3].ravel()
force_total = (force_x ** 2 + force_y ** 2 + force_z ** 2) ** 0.5
velocity_x = velocity[0::3].ravel()
velocity_y = velocity[1::3].ravel()
velocity_z = velocity[2::3].ravel()
velocity_total = (velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2) ** 0.5
pointsToVTK("./surfaceForce", nodes[:, 0], nodes[:, 1], nodes[:, 2],
            data={"force_x": force_x,
                  "force_y": force_y,
                  "force_z": force_z,
                  "force_total": force_total,
                  "velocity_x": velocity_x,
                  "velocity_y": velocity_y,
                  "velocity_z": velocity_z,
                  "velocity_total": velocity_total})

print("")
