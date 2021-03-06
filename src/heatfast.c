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

	struct field surfaces[NSURFACE_MAX];
	int nsurfaces;

	Real dy, dx;

	double fulltick, fulltock;
	double tick, tock;


	/* inits */
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	MPI_Barrier(MPI_COMM_WORLD);
	fulltick = MPI_Wtime();

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

	fprintf(stderr, "[%d]: MPI coord: [%d,%d]\n", iproc, mpistate.mpicoord[0], mpistate.mpicoord[1]);
	
	
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


	/* init surfaces. all surfaces ar global */
	nsurfaces = globalConfig.nsurfaces;
	if (nsurfaces > NSURFACE_MAX) {
		fprintf(stderr, "ERROR: Max number of surfaces is %d\n", NSURFACE_MAX);
		MPI_Abort(MPI_COMM_WORLD, ERR_CONFIG);
		exit(ERR_CONFIG);
	}

	for (int i = 0; i < nsurfaces; i++) {
		mstate.surfaces[i] = &surfaces[i];
		surfaces[i].nx = gridx.nx;
		surfaces[i].ny = gridx.ny;
		surfaces[i].ox = gridx.ox;
		surfaces[i].oy = gridx.oy;
		surfaces[i].f = malloc(sizeof(Real) * surfaces[i].nx);
		for (int k = 0; k < surfaces[i].nx; k++) surfaces[i].f[k] = -1.0;
	}

	
	/* initialize fields */
	if (iproc == 0) fprintf(stdout, "Allocating memory for fields\n");
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

	/* set initial values */
	if (iproc == 0) fprintf(stdout, "Reading initial fields from HDF5\n");
	readHdf5(&T, &mpistate, globalConfig.initfile);
	readHdf5(&Kdiff, &mpistate, globalConfig.initfile);

	communicateHalos(&mpistate, &T);
	communicateHalos(&mpistate, &Kdiff);

	/* set initial surfaces */
	if (iproc == 0) fprintf(stdout, "Reading initial surfaces (n=%d) from HDF5\n", nsurfaces);
	for (int i = 0; i < nsurfaces; i++) {
		char tmpname[STR_MAXLEN];
		snprintf(tmpname, STR_MAXLEN, "surface_%d", i);
		nameField(&surfaces[i], tmpname);
		readHdf5(&surfaces[i], &mpistate, globalConfig.initfile);
		communicateHalos(&mpistate, &surfaces[i]);
	}


	tick = MPI_Wtime();

	for (int iter = 0; iter < globalConfig.niter; iter++) {
		Real dtlocal, dtglobal;

		if (iproc == 0) fprintf(stdout, "Iter %d\n", iter);

		/* take a time step */
		swapFields(&T, &Told);
		
		if (globalConfig.dt > 0) {
			dtglobal = globalConfig.dt;
		} else {
			Real maxK = 0.0;
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
			if (globalConfig.bctypes[BND_X0] & BC_TYPE_DIRICHLET) {
				int j = 1;
				for (int i = 0; i < T.ny; i++) {
					T.f[i*T.nx + j] = globalConfig.bcvalues[BND_X0];
				}
			} else if (globalConfig.bctypes[BND_X0] & BC_TYPE_NEUMANN) {
				int j = 1;
				for (int i = 0; i < T.ny; i++) {
					T.f[i*T.nx + j] = T.f[i*T.nx + j+1];
				}
			} else {
				fprintf(stderr, "Unrecognized BC type for boundary x0\n");
				MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_OPTION|ERR_CONFIG);
				exit(ERR_INVALID_OPTION|ERR_CONFIG);
			}
		} 
		if (mpistate.mpicoord[IX] == globalConfig.px-1) {
			// i have (part of) the right bnd bnd
			if (globalConfig.bctypes[BND_X1] & BC_TYPE_DIRICHLET) {
				int j = T.nx-2;
				for (int i = 0; i < T.ny; i++) {
					T.f[i*T.nx + j] = globalConfig.bcvalues[BND_X1];
				}
			} else if (globalConfig.bctypes[BND_X1] & BC_TYPE_NEUMANN) {
				int j = T.nx-2;
				for (int i = 0; i < T.ny; i++) {
					T.f[i*T.nx + j] = T.f[i*T.nx + j-1];
				}
			} else {
				fprintf(stderr, "Unrecognized BC type for boundary x1\n");
				MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_OPTION|ERR_CONFIG);
				exit(ERR_INVALID_OPTION|ERR_CONFIG);
			}
		}

		if (mpistate.mpicoord[IY] == 0) {
			// i have (part of) the upper bnd
			if (globalConfig.bctypes[BND_Y0] & BC_TYPE_DIRICHLET) {
				int i = 1;
				for (int j = 0; j < T.nx; j++) {
					T.f[i*T.nx + j] = globalConfig.bcvalues[BND_Y0];
				}
			} else if (globalConfig.bctypes[BND_Y0] & BC_TYPE_NEUMANN) {
				int i = 1;
				for (int j = 0; j < T.nx; j++) {
					T.f[i*T.nx + j] = T.f[(i+1)*T.nx + j];
				}
			} else {
				fprintf(stderr, "Unrecognized BC type for boundary y0\n");
				MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_OPTION|ERR_CONFIG);
				exit(ERR_INVALID_OPTION|ERR_CONFIG);
			}
		}
		if (mpistate.mpicoord[IY] == globalConfig.py-1) {
			// i have (part of) the lower bnd bnd
			if (globalConfig.bctypes[BND_Y1] & BC_TYPE_DIRICHLET) {
				int i = T.ny-2;
				for (int j = 0; j < T.nx; j++) {
					T.f[i*T.nx + j] = globalConfig.bcvalues[BND_Y1];
				}
			} else if (globalConfig.bctypes[BND_Y1] & BC_TYPE_NEUMANN) {
				int i = T.ny-2;
				for (int j = 0; j < T.nx; j++) {
					T.f[i*T.nx + j] = T.f[(i-1)*T.nx + j];
				}
			} else {
				fprintf(stderr, "Unrecognized BC type for boundary y1\n");
				MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_OPTION|ERR_CONFIG);
				exit(ERR_INVALID_OPTION|ERR_CONFIG);
			}
		}

		/* apply surfaces */
		/* TODO:
		 * 	- implement "surface types", atm only uppermost surface which forces T=0 above
		 * 	- implement surface process model
		 */
		for (int k = 0; k < 1; k++) {  // WIP: for now only considers the first surface
			for (int j = 1; j < gridx.nx-1; j++) {
				Real s_y = surfaces[k].f[j];
				for (int i = gridy.ny-1; i >= 1 && gridy.f[i] > s_y; i--) {
					T.f[i*T.nx + j] = 0.0;
				}
			}

			// here: Do surface processes 
			// then:
			communicateHalos(&mpistate, &surfaces[k]);
		}

		/* send halos around */
		communicateHalos(&mpistate, &T);
		communicateHalos(&mpistate, &Kdiff);
	}

	tock = MPI_Wtime();

	if (iproc == 0) fprintf(stdout, "*** %d iters took %g sec\n", globalConfig.niter, tock-tick);

	writeFields(&mstate, &mpistate, &globalConfig);

	MPI_Barrier(MPI_COMM_WORLD);
	fulltock = MPI_Wtime();
	if (iproc == 0) fprintf(stdout, "*** All in all, it took %g sec\n", fulltock-fulltick);

	MPI_Finalize();

	return 0;
}

