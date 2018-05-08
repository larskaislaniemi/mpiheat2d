#!/usr/bin/python3
import h5py
import numpy as np
import sys

if len(sys.argv) != 6:
    print("Usage:", sys.argv[0], " modelname ny nx Ly Lx")
    raise "argument error"

modelname = sys.argv[1]
nx = int(sys.argv[3]) # 2**11
ny = int(sys.argv[2]) # 2**11
Lx = float(sys.argv[5]) # 50e3
Ly = float(sys.argv[4]) # 50e3

numdatatype = "float64"    # change to float32 for float in heatfast.c

###############################

class WrongArgumentException(Exception):
    pass


def getValue(modelname, field, x, y, Lx, Ly):
    if modelname == "simplegradient":
        if field == "T":
            return 500.0 * y / Ly
        elif field == "Kdiff":
            return 1e-6
    elif modelname == "variedk":
        if field == "T":
            return 0.0
        elif field == "Kdiff":
            if y < 0.5*Ly:
                return 1e-6
            else:
                return 4e-6
    else:
        raise Exception("No such model")

dx = Lx / (nx-1)
dy = Ly / (ny-1)

#T = np.zeros((ny,nx), dtype=numdatatype)
x = np.linspace(0, Lx, nx).reshape(1,nx)
y = np.linspace(0, Ly, ny).reshape(ny,1)

f = h5py.File('init.h5', 'w')

dset_T     = f.create_dataset("T",     shape=(ny, nx), dtype=numdatatype)
dset_Kdiff = f.create_dataset("Kdiff", shape=(ny, nx), dtype=numdatatype)

if modelname == "simplegradient":
    dset_T[:,:] = 500.0 * y / Ly
    dset_Kdiff[:,:] = 1e-6
elif modelname == "variedk":
    dset_T[:,:] = 0.0

    idxw = np.where(y[:,0] < 0.5*Ly)
    idx = (np.amin(idxw), np.amax(idxw))
    print(idx)
    dset_Kdiff[idx[0]:idx[1]+1,:] = 1e-6

    idxw = np.where(y[:,0] >= 0.5*Ly)
    idx = (np.amin(idxw), np.amax(idxw))
    print(idx)
    dset_Kdiff[idx[0]:idx[1]+1,:] = 4e-6

else:
    raise WrongArgumentException("No such model")


#print("Start")
#for i in range(ny):
#    sys.stdout.write(".")
#    sys.stdout.flush()
#    for j in range(nx):
#        dset_T[i,j]     = getValue(sys.argv[1], "T",     dx*j, dy*i, Lx, Ly)
#        dset_Kdiff[i,j] = getValue(sys.argv[1], "Kdiff", dx*j, dy*i, Lx, Ly)
#print("Done")

f.close()
