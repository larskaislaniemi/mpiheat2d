Heat diffusion code for massive parallelization / hi-res.

Required libraries:

- MPI
- HDF5 (parallel support enabled)
- libconfig

Installation:

	# mkdir build
	# cd build
	# cmake ..
	# make

Run:

	# vim ../data/input.cfg
	# mpirun -np NPROC ./heatfast ../data/input.cfg
	# h5ls data.h5
