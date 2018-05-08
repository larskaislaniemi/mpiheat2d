Heat diffusion code for massive parallelization / hi-res.

Required libraries:

- MPI
- HDF5 (parallel support enabled)
- libconfig
- Python3 with numpy and h5py

Building:

	# mkdir build
	# cd build
	# cmake ..
	# make

To run an example case:

	Generate initial temperature field to 'init.h5'
	# cd ../data
	# ../tools/generateInitialField variedk 32 32 50e3 50e3

	Modify run parameters as needed (nx, ny, Lx, Ly should match
	those used above):
	# vim input.cfg

	Run (output goes to data.h5):
	# mpirun -np 4 ../build/heatfast

	Generate a Xdmf metadata file for the H5 file:
	# ../tools/generateXdmf.py data.h5 data.xdmf

	Open the resulting Xdmf file in, e.g., ParaView or
	other software supporting HDF5 dataformat.

