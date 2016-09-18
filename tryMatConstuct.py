import numpy as np
from petsc4py import PETSc
import sys, petsc4py
from mpi4py import MPI


petsc4py.init(sys.argv)
mSizes = (2, 2)
mij = []

# create sub-matrices mij
for i in range(len(mSizes)):
    for j in range(len(mSizes)):
        temp_m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        temp_m.setSizes(((None, mSizes[i]), (None, mSizes[j])))
        temp_m.setType('mpidense')
        temp_m.setFromOptions()
        temp_m.setUp()
        temp_m[:, :] = np.random.random_sample((mSizes[i], mSizes[j]))
        temp_m.assemble()
        temp_m.view()
        mij.append(temp_m)

# Now we have four sub-matrices. I would like to construct them into a big matrix M.
M = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
M.setSizes(((None, np.sum(mSizes)), (None, np.sum(mSizes))))
M.setType('mpidense')
M.setFromOptions()
M.setUp()
mLocations = np.insert(np.cumsum(mSizes), 0, 0)    # mLocations = [0, mSizes]
for i in range(len(mSizes)):
    for j in range(len(mSizes)):
        temp_m = mij[i*len(mSizes)+j].getDenseArray()
        temp_m_start, temp_m_end = mij[i*len(mSizes)+j].getOwnershipRange()
        rank = MPI.COMM_WORLD.Get_rank()
        print('rank:', rank, '   ', i, '   ', j, '   ', temp_m, '   ', temp_m.shape, '   ', temp_m_start, '   ', temp_m_end)
        for k in range(temp_m_start, temp_m_end):
            # print('i: %d, j: %d, k: %d, mLocations[i]: %d'%(i, j, k, mLocations[i]))
            M.setValues(mLocations[i]+k, np.arange(mLocations[j],mLocations[j+1],dtype='int32'), temp_m[k-temp_m_start, :])
M.assemble()
M.view()