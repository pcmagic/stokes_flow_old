# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410


import stokes_flow as sf
# import numpy as np
nodes, velocity, delta = sf.import_mat()
rs_m = sf.regularized_stokeslets_matrix_3d(nodes, nodes, delta)

# import petsc4py
# petsc4py.init()
from petsc4py import PETSc
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
force_x = force[0::3].ravel()
force_y = force[1::3].ravel()
force_z = force[2::3].ravel()
force_total = (force_x ** 2 + force_y ** 2 + force_z ** 2) ** 0.5

# import numpy_io as nio
# nio.write(velocity, 'velocity.txt')
# nio.write(rs_m, 'rs_m.txt')
# nio.write(force, 'force.txt')

from evtk.hl import pointsToVTK
pointsToVTK("./surfaceForce", nodes[:, 0], nodes[:, 1], nodes[:, 2], \
            data={"force_x": force_x, \
                  "force_y": force_y, \
                  "force_z": force_z, \
                  "force_total": force_total})

print("")