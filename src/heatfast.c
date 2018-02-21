#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <hdf5.h>
#include <libconfig.h>
#include "heatfast.h"


int main(int argc, char **argv) {
	struct mpidata mpistate;
	mpistate.mpicomm_ndims = 2;
	mpistate.mpicomm_reorder = 1;
	mpistate.mpicomm_periods[0] = 0; 
	mpistate.mpicomm_periods[1] = 0;
	int iproc, nproc, ierr;

	struct config globalConfig;
	struct modeldata mstate;

	struct field T, Told;
	struct field Kdiff;
	struct field globalGridx, globalGridy, gridx, gridy;

	Real dy, dx;


	/* inits */
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);



	/* read config */

	if (argc != 2) {
		fprintf(stderr, "ERROR: No config file\nUsage: %s configfile\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, ERR_CMDLINE_ARGUMENT);
		exit(ERR_CMDLINE_ARGUMENT);
	}

	readConfig(&globalConfig, argv[1]);

		
	/* find good proc configuration */

	ierr = domainDecomp(globalConfig.nx, globalConfig.ny, nproc, &(globalConfig.px), &(globalConfig.py));
	if (ierr != 0) {
		MPI_Abort(MPI_COMM_WORLD, ierr);
		exit(ierr);
	}

	if (iproc == 0) {
		fprintf(stdout, "Proc configuration: nx,ny = %d,%d; np = %d; px,py = %d, %d\n", globalConfig.nx, globalConfig.ny, nproc, globalConfig.px, globalConfig.py);
	}

	mpistate.mpicomm_dims[IY] = globalConfig.py;
	mpistate.mpicomm_dims[IX] = globalConfig.px;

	MPI_Cart_create(MPI_COMM_WORLD, mpistate.mpicomm_ndims, mpistate.mpicomm_dims, mpistate.mpicomm_periods, mpistate.mpicomm_reorder, &mpistate.comm2d);
	MPI_Cart_get(mpistate.comm2d, mpistate.mpicomm_ndims, mpistate.mpicomm_dims, mpistate.mpicomm_periods, mpistate.mpicoord);

	
	
	/* init grids */

	dy = globalConfig.Ly / (globalConfig.ny-1);
	dx = globalConfig.Lx / (globalConfig.nx-1);
	globalGridx.nx = globalConfig.nx; globalGridx.ny = 1;
	globalGridy.ny = globalConfig.ny; globalGridy.nx = 1;
	globalGridx.ox = globalGridx.oy = 0;
	globalGridy.ox = globalGridy.oy = 0;
	globalGridy.f = malloc(sizeof(Real) * globalGridy.ny);
	globalGridx.f = malloc(sizeof(Real) * globalGridx.nx);
	for (int i = 0; i < globalGridy.ny; i++) globalGridy.f[i] = i*dy;
	for (int j = 0; j < globalGridx.nx; j++) globalGridx.f[j] = j*dx;

	gridx.nx = globalConfig.nx / globalConfig.px + 2; gridx.ny = 1;
	gridy.ny = globalConfig.ny / globalConfig.py + 2; gridy.nx = 1;
	gridx.ox = mpistate.mpicoord[IX] * globalConfig.nx / globalConfig.px - 1; gridx.oy = 0;
	gridy.oy = mpistate.mpicoord[IY] * globalConfig.ny / globalConfig.py - 1; gridy.ox = 0;
	gridy.f = malloc(sizeof(Real) * gridy.ny);
	gridx.f = malloc(sizeof(Real) * gridx.nx);
	for (int i = 0; i < gridy.ny; i++) gridy.f[i] = (i + gridy.oy)*dy;
	for (int j = 0; j < gridx.nx; j++) gridx.f[j] = (j + gridx.ox)*dx;

	mstate.grids = malloc(sizeof(struct field *)*2);
	mstate.grids[IX] = &gridx;
	mstate.grids[IY] = &gridy;

	mstate.globalGrids = malloc(sizeof(struct field *)*2);
	mstate.globalGrids[IX] = &globalGridx;
	mstate.globalGrids[IY] = &globalGridy;

	
	
	/* initialize fields */
	T.nx = gridx.nx; T.ny = gridy.ny; 
	T.ox = gridx.ox; T.oy = gridy.oy;
	T.f = calloc(sizeof(Real), T.nx*T.ny);
	createFromField(&T, &Told);
	nameField(&T, "T");
	nameField(&Told, "Told");

	Kdiff.nx = gridx.nx; Kdiff.ny = gridy.ny; 
	Kdiff.ox = gridx.ox; Kdiff.oy = gridy.oy;
	Kdiff.f = calloc(sizeof(Real), Kdiff.nx*Kdiff.ny);
	nameField(&Kdiff, "Kdiff");

	mstate.time = 0.0;
	mstate.timestep = 0;
	mstate.nfields = 3;
	mstate.fields = malloc(sizeof(struct field *) * 3);
	mstate.fields[0] = &T;
	mstate.fields[1] = &Told;
	mstate.fields[2] = &Kdiff;


	readHdf5(&T, &mpistate, globalConfig.initfile);
	readHdf5(&Kdiff, &mpistate, globalConfig.initfile);

	communicateHalos(&mpistate, &T);
	communicateHalos(&mpistate, &Kdiff);


	for (int iter = 0; iter < globalConfig.niter; iter++) {
		Real dtlocal, dtglobal, maxK;
		maxK = 0.0;

		if (iproc == 0) fprintf(stdout, "Iter %d\n", iter);
		/* take a time step */
		swapFields(&T, &Told);
		
		if (globalConfig.dt > 0) {
			dtglobal = globalConfig.dt;
		} else {
			for (int i = 0; i < T.ny; i++)
				for (int j = 0; j < T.nx; j++)
					if (Kdiff.f[i*T.nx + j] > maxK) maxK = Kdiff.f[i*T.nx + j];
			dtlocal = 0.25 * pow(MIN(dx, dy), 2.0) / (4.0*maxK);
			MPI_Allreduce(&dtlocal, &dtglobal, 1, C_MPI_REAL, MPI_MIN, mpistate.comm2d);
		}

		for (int i = 1; i < T.ny-1; i++) {
			for (int j = 1; j < T.nx-1; j++) {
				T.f[i*T.nx + j] = dtglobal * (
						(
						 0.5 * (Kdiff.f[i*T.nx + j+1] + Kdiff.f[i*T.nx + j]) * (Told.f[i*T.nx + j+1] - Told.f[i*T.nx + j]) / dx + 
						 0.5 * (Kdiff.f[i*T.nx + j] + Kdiff.f[i*T.nx + j-1]) * (Told.f[i*T.nx + j] - Told.f[i*T.nx + j-1]) / dx
						) / dx +
						(
						 0.5 * (Kdiff.f[(i+1)*T.nx + j] + Kdiff.f[i*T.nx + j]) * (Told.f[(i+1)*T.nx + j] - Told.f[i*T.nx + j]) / dy + 
						 0.5 * (Kdiff.f[i*T.nx + j] + Kdiff.f[(i-1)*T.nx + j]) * (Told.f[i*T.nx + j] - Told.f[(i-1)*T.nx + j]) / dy
						) / dy
					) + Told.f[i*T.nx + j];
			}
		}

		if (mpistate.mpicoord[IX] == 0) {
			// i have (part of) the left bnd
			if (globalConfig.bctypes[2] == 1) {
				int j = 1;
				for (int i = 0; i < T.ny; i++) {
					T.f[i*T.nx + j] = globalConfig.bcvalues[2];
				}
			} else {
				fprintf(stderr, "Unrecognized BC type for boundary x0\n");
				MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_OPTION|ERR_CONFIG);
				exit(ERR_INVALID_OPTION|ERR_CONFIG);
			}
		} 
		if (mpistate.mpicoord[IX] == globalConfig.px-1) {
			// i have (part of) the right bnd bnd
			if (globalConfig.bctypes[3] == 1) {
				int j = T.nx-2;
				for (int i = 0; i < T.ny; i++) {
					T.f[i*T.nx + j] = globalConfig.bcvalues[3];
				}
			} else {
				fprintf(stderr, "Unrecognized BC type for boundary x1\n");
				MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_OPTION|ERR_CONFIG);
				exit(ERR_INVALID_OPTION|ERR_CONFIG);
			}
		}

		if (mpistate.mpicoord[IY] == 0) {
			// i have (part of) the upper bnd
			if (globalConfig.bctypes[0] == 1) {
				int i = 1;
				for (int j = 0; j < T.nx; j++) {
					T.f[i*T.nx + j] = globalConfig.bcvalues[0];
				}
			} else {
				fprintf(stderr, "Unrecognized BC type for boundary y0\n");
				MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_OPTION|ERR_CONFIG);
				exit(ERR_INVALID_OPTION|ERR_CONFIG);
			}
		}
		if (mpistate.mpicoord[IY] == globalConfig.py-1) {
			// i have (part of) the lower bnd bnd
			if (globalConfig.bctypes[1] == 1) {
				int i = T.ny-2;
				for (int j = 0; j < T.nx; j++) {
					T.f[i*T.nx + j] = globalConfig.bcvalues[1];
				}
			} else {
				fprintf(stderr, "Unrecognized BC type for boundary y1\n");
				MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_OPTION|ERR_CONFIG);
				exit(ERR_INVALID_OPTION|ERR_CONFIG);
			}
		}

		/* send halos around */
		communicateHalos(&mpistate, &T);
		communicateHalos(&mpistate, &Kdiff);

	}

	writeFields(&mstate, &mpistate, &globalConfig);

	MPI_Finalize();

	return 0;
}

