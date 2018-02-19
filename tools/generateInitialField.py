#!/usr/bin/python3
import h5py
import numpy as np

nx = 8
ny = 16
Lx = 50e3
Ly = 50e3

numdatatype = "float64"    # change to float32 for float in heatfast.c

###############################

dx = Lx / (nx-1)
dy = Ly / (ny-1)

T = np.zeros((ny,nx), dtype=numdatatype)
x = np.linspace(0, Lx, nx).reshape(1,nx)
y = np.linspace(0, Ly, ny).reshape(ny,1)




f = h5py.File('init.h5', 'w')

dset_T = f.create_dataset("T", (ny,nx), dtype=numdatatype)
dset_T[:,:] = 500.0 * y / Ly

dset_Kdiff = f.create_dataset("Kdiff", (ny,nx), dtype=numdatatype)
dset_Kdiff[:,:] = 1e-6;

f.close()
