#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <hdf5.h>
#include <libconfig.h>

#define IY 0
#define IX 1

#define ERR_CONFIG 1
#define ERR_SIZE_MISMATCH 2
#define ERR_CMDLINE_ARGUMENT 4
#define ERR_FILEOPER 8
#define ERR_CFGFILE 16

typedef double Real;               // NB! libconfig currently support double, not float
#define C_MPI_REAL MPI_DOUBLE
#define C_H5_REAL H5T_NATIVE_DOUBLE

struct config {
	int nx, ny, px, py;   // num of grid points; num of procs
	Real Lx, Ly;        // physical dimensions
	Real Kdiff;         // heatdiffusivity
	Real dt;            // time step
	int niter;            // num of iters
};

struct field {
	Real *f;
	int nx, ny;
	int ox, oy;
};

struct mpidata {
	MPI_Comm comm2d;
	int mpicomm_ndims;
	int mpicomm_reorder;
	int mpicomm_periods[2];
	int mpicomm_dims[2];
	int mpicoord[2];
	int rank_neighb[4];
};

void createMPIDatatypeConfig(struct config *c, MPI_Datatype *newtype) {
	const int configStructLen = 9;
	const int configBlockLens[9] = {
		1, 1, 1, 1,
		1, 1, 
		1,
		1,
		1
	};
	const MPI_Datatype configDatatypes[9] = {
		MPI_INT, MPI_INT, MPI_INT, MPI_INT,
		C_MPI_REAL, C_MPI_REAL,
		C_MPI_REAL,
		C_MPI_REAL,
		MPI_INT
	};
	MPI_Aint configDisplacements[9];

	configDisplacements[0] = (long int)&c->nx;
	configDisplacements[1] = (long int)&c->ny    - configDisplacements[0];
	configDisplacements[2] = (long int)&c->px    - configDisplacements[0];
	configDisplacements[3] = (long int)&c->py    - configDisplacements[0];
	configDisplacements[4] = (long int)&c->Lx    - configDisplacements[0];
	configDisplacements[5] = (long int)&c->Ly    - configDisplacements[0];
	configDisplacements[6] = (long int)&c->Kdiff - configDisplacements[0];
	configDisplacements[7] = (long int)&c->dt    - configDisplacements[0];
	configDisplacements[8] = (long int)&c->niter - configDisplacements[0];
	configDisplacements[0] = 0;

	MPI_Type_create_struct(configStructLen, configBlockLens, configDisplacements, configDatatypes, newtype);
}

void createFromField(struct field const *const f1, struct field *const f2) {
	f2->nx = f1->nx; f2->ny = f1->ny;
	f2->ox = f1->ox; f2->oy = f1->oy;
	f2->f = malloc(sizeof(Real) * f2->nx * f2->ny);
}

Real gen_field_T(Real y, Real x) {
	return 1350.0;
	if (pow(0.5-x, 2.0) + pow(0.5-y, 2.0) < 0.4*0.4) {
		return 1.0;
	} else {
		return 0.0;
	}
}

void swapFields(struct field *const a, struct field *const b) {
	Real *tmp;
	int tmpox, tmpoy;

	if (a->nx != b->nx || a->ny != b->ny) {
		fprintf(stderr, "cannot swap fields of different sizes\n");
		MPI_Abort(MPI_COMM_WORLD, ERR_SIZE_MISMATCH);
		exit(ERR_SIZE_MISMATCH);
	}

	tmp = a->f;
	a->f = b->f;
	b->f = tmp;

	tmpox = a->ox;
	tmpoy = a->oy;
	a->ox = b->ox;
	a->oy = b->oy;
	b->ox = tmpox;
	b->oy = tmpoy;
}

void printDomains(struct mpidata const *const mpistate, struct field const *const fld) {
	int iproc, nproc;

	MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	for (int k = 0; k < nproc; k++) {
		if (iproc == k) {
			fprintf(stdout, "[%d] (%d,%d)\n", iproc, mpistate->mpicoord[IY], mpistate->mpicoord[IX]);
			for (int i = 0; i < fld->ny; i++) {
				for (int j = 0; j < fld->nx; j++) {
					fprintf(stdout, "%6.2g ", fld->f[i*fld->nx + j]);
				}
				fprintf(stdout, "\n");
			}
			fprintf(stdout, "\n--\n");
		}
		MPI_Barrier(mpistate->comm2d);
	}
}

int main(int argc, char **argv) {
	struct mpidata mpistate;
	mpistate.mpicomm_ndims = 2;
	mpistate.mpicomm_reorder = 1;
	mpistate.mpicomm_periods[0] = 0; 
	mpistate.mpicomm_periods[1] = 0;
	int iproc, nproc;

	//int mpicoord[2];

	struct config globalConfig;

	struct field T, Told;
	struct field globalGridx, globalGridy, gridx, gridy;

	Real dy, dx;
	MPI_Datatype columntype;

	double time2, time1;

	config_t inputcfg;
	FILE *cfgfp;

	MPI_Datatype globalConfigMPIType;

	/* inits */
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	//fprintf(stdout, "[%d/%d] Hello\n", iproc, nproc);

	/* read config */
	if (iproc == 0) {

		if (argc != 2) {
			fprintf(stderr, "ERROR: No config file\nUsage: %s configfile\n", argv[0]);
			MPI_Abort(MPI_COMM_WORLD, ERR_CMDLINE_ARGUMENT);
			exit(ERR_CMDLINE_ARGUMENT);
		}

		cfgfp = fopen(argv[1], "r");
		if (cfgfp == NULL) {
			fprintf(stderr, "ERROR: open file %s\n", argv[1]);
			MPI_Abort(MPI_COMM_WORLD, ERR_FILEOPER|ERR_CMDLINE_ARGUMENT);
			exit(ERR_FILEOPER|ERR_CMDLINE_ARGUMENT);
		}
		config_init(&inputcfg);
		config_read(&inputcfg, cfgfp);
		fclose(cfgfp);

		if (config_lookup_int(&inputcfg, "grid.nx", &globalConfig.nx) != CONFIG_TRUE) { fprintf(stderr, "config option grid.nx needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_int(&inputcfg, "grid.ny", &globalConfig.ny) != CONFIG_TRUE) { fprintf(stderr, "config option grid.ny needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_float(&inputcfg, "grid.Lx", &globalConfig.Lx) != CONFIG_TRUE) { fprintf(stderr, "config option grid.Lx needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_float(&inputcfg, "grid.Ly", &globalConfig.Ly) != CONFIG_TRUE) { fprintf(stderr, "config option grid.Ly needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_int(&inputcfg, "mpi.px", &globalConfig.px) != CONFIG_TRUE) { fprintf(stderr, "config option mpi.px needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_int(&inputcfg, "mpi.py", &globalConfig.py) != CONFIG_TRUE) { fprintf(stderr, "config option mpi.py needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_float(&inputcfg, "phys.kdiff", &globalConfig.Kdiff) != CONFIG_TRUE) { fprintf(stderr, "config option phys.kdiff needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_int(&inputcfg, "run.iter", &globalConfig.niter) != CONFIG_TRUE) { fprintf(stderr, "config option run.iter needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
	}
	createMPIDatatypeConfig(&globalConfig, &globalConfigMPIType);
	MPI_Type_commit(&globalConfigMPIType);
	if (iproc == 0) {
		for (int i = 1; i < nproc; i++) {
			MPI_Send(&globalConfig, 1, globalConfigMPIType, i, 0, MPI_COMM_WORLD);
		}
	} else {
		MPI_Recv(&globalConfig, 1, globalConfigMPIType, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

/*	globalConfig.nx = 2000;
	globalConfig.ny = 2000;
	globalConfig.Lx = 1.0;
	globalConfig.Ly = 1.0;
	globalConfig.px = 2;
	globalConfig.py = 2;
	globalConfig.Kdiff = 1.0;
	globalConfig.dt = 0; // dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
	globalConfig.niter = 100;*/


	/* find good proc configuration */
	if (globalConfig.nx % globalConfig.px != 0 || globalConfig.ny % globalConfig.py != 0) {
		fprintf(stderr, "globalConfig.nx %% globalConfig.px != 0 || globalConfig.ny %% globalConfig.py != 0\n");
		MPI_Abort(MPI_COMM_WORLD, ERR_CONFIG);
		exit(ERR_CONFIG);
	}

	if (nproc != globalConfig.px*globalConfig.py) {
		fprintf(stderr, "nproc (%d) != globalConfig.px*globalConfig.py (%d)\n", nproc, globalConfig.px*globalConfig.py);
		MPI_Abort(MPI_COMM_WORLD, ERR_CONFIG);
		exit(ERR_CONFIG);
	}

	mpistate.mpicomm_dims[IY] = globalConfig.py;
	mpistate.mpicomm_dims[IX] = globalConfig.px;

	MPI_Cart_create(MPI_COMM_WORLD, mpistate.mpicomm_ndims, mpistate.mpicomm_dims, mpistate.mpicomm_periods, mpistate.mpicomm_reorder, &mpistate.comm2d);
	MPI_Cart_get(mpistate.comm2d, mpistate.mpicomm_ndims, mpistate.mpicomm_dims, mpistate.mpicomm_periods, mpistate.mpicoord);


	/* init grids */

	dy = globalConfig.Ly / (globalConfig.ny-1);
	dx = globalConfig.Lx / (globalConfig.nx-1);
	globalConfig.dt = 0.5 * dx*dx * dy*dy / (2.0 * globalConfig.Kdiff * (dx*dx + dy*dy));
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

	/* initialize fields */
	T.nx = gridx.nx; T.ny = gridy.ny; 
	T.ox = gridx.ox; T.oy = gridy.oy;
	T.f = malloc(sizeof(Real) * T.nx*T.ny);

	createFromField(&T, &Told);



	/* setup initial values */
	for (int i = 0; i < gridy.ny; i++) {
		for (int j = 0; j < gridx.nx; j++) {
			T.f[i*gridx.nx + j] = gen_field_T(gridy.f[i], gridx.f[j]);
		}
	}

	time1 = MPI_Wtime();

	for (int iter = 0; iter < globalConfig.niter; iter++) {
		if (iproc == 0) fprintf(stdout, "Iter %d\n", iter);
		/* take a time step */
		swapFields(&T, &Told);
		
		for (int i = 1; i < T.ny-1; i++) {
			for (int j = 1; j < T.nx-1; j++) {
				// TODO: Variable dx/dy, variable k,rho,Cp
				T.f[i*T.nx + j] = globalConfig.dt * (
						globalConfig.Kdiff * (Told.f[i*T.nx + j+1] - 2.0*Told.f[i*T.nx + j] + Told.f[i*T.nx + j-1]) / (dx*dx) +
						globalConfig.Kdiff * (Told.f[(i+1)*T.nx + j] - 2.0*Told.f[i*T.nx + j] + Told.f[(i-1)*T.nx + j]) / (dy*dy)
					) + Told.f[i*T.nx + j];
			}
		}

		if (mpistate.mpicoord[IX] == 0) {
			// i have (part of) the left bnd
			int j = 1;
			for (int i = 0; i < T.ny; i++) {
				T.f[i*T.nx + j] = 0.0;
			}
		} 
		if (mpistate.mpicoord[IX] == globalConfig.px-1) {
			// i have (part of) the right bnd bnd
			int j = T.nx-2;
			for (int i = 0; i < T.ny; i++) {
				T.f[i*T.nx + j] = 0.0;
			}
		}

		if (mpistate.mpicoord[IY] == 0) {
			// i have (part of) the upper bnd
			int i = 1;
			for (int j = 0; j < T.nx; j++) {
				T.f[i*T.nx + j] = 0.0;
			}
		}
		if (mpistate.mpicoord[IY] == globalConfig.py-1) {
			// i have (part of) the lower bnd bnd
			int i = T.ny-2;
			for (int j = 0; j < T.nx; j++) {
				T.f[i*T.nx + j] = 0.0;
			}
		}

		/* send halos around */

		MPI_Cart_shift(mpistate.comm2d, IY, 1, &mpistate.rank_neighb[0], &mpistate.rank_neighb[1]);
		MPI_Cart_shift(mpistate.comm2d, IX, 1, &mpistate.rank_neighb[2], &mpistate.rank_neighb[3]);
		for (int i = 0; i < 4; i++) if (mpistate.rank_neighb[i] < 0) mpistate.rank_neighb[i] = MPI_PROC_NULL;

		// send lower and upper row
		MPI_Sendrecv(&T.f[T.nx*(T.ny-2)], T.nx, C_MPI_REAL, mpistate.rank_neighb[1], 0, 
				&T.f[0], T.nx, C_MPI_REAL, mpistate.rank_neighb[0], 0, mpistate.comm2d, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&T.f[T.nx*1], T.nx, C_MPI_REAL, mpistate.rank_neighb[0], 0, 
				&T.f[T.nx*(T.ny-1)], T.nx, C_MPI_REAL, mpistate.rank_neighb[1], 0, mpistate.comm2d, MPI_STATUS_IGNORE);

		// send right and left column
		MPI_Type_vector(T.ny, 1, T.nx, C_MPI_REAL, &columntype);
		MPI_Type_commit(&columntype);
		MPI_Sendrecv(&T.f[T.nx-2], 1, columntype, mpistate.rank_neighb[3], 0, 
				&T.f[0], 1, columntype, mpistate.rank_neighb[2], 0, mpistate.comm2d, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&T.f[1], 1, columntype, mpistate.rank_neighb[2], 0, 
				&T.f[T.nx-1], 1, columntype, mpistate.rank_neighb[3], 0, mpistate.comm2d, MPI_STATUS_IGNORE);
		MPI_Type_free(&columntype);

	}

	time2 = MPI_Wtime();
	if (iproc == 0) fprintf(stderr, "Iters took %g secs\n", time2-time1);

	//MPI_Barrier(MPI_COMM_WORLD);
	//printDomains(&mpistate, &T);
	//printDomains(&mpistate, &gridx);
	//printDomains(&mpistate, &gridy);
	//
	

	/* write data out
	 */
	if (iproc == 0) fprintf(stdout, "Output\n");
	time1 = MPI_Wtime();

	{
		herr_t status;
		hid_t plist_id, dset_id, filespace, memspace, file_id;
		hsize_t dims[2], counts[2], offsets[2];

		plist_id = H5Pcreate(H5P_FILE_ACCESS);
		H5Pset_fapl_mpio(plist_id, mpistate.comm2d, MPI_INFO_NULL);
		file_id = H5Fcreate("data.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
		H5Pclose(plist_id);


		plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

		/* Create the dataset */
		dims[0] = globalConfig.ny;
		dims[1] = globalConfig.nx;
		filespace = H5Screate_simple(2, dims, NULL);
		dset_id = H5Dcreate(file_id, "T", C_H5_REAL, filespace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		H5Sclose(filespace);

		/* Select a hyperslab of the file dataspace */
		offsets[0] = T.oy+1; offsets[1] = T.ox+1;
		counts[0] = T.ny-2; counts[1] = T.nx-2;
		filespace = H5Dget_space(dset_id);
		H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		counts[0] = T.ny; counts[1] = T.nx;
		memspace = H5Screate_simple(2, counts, NULL);
		counts[0] = T.ny-2; counts[1] = T.nx-2;
		offsets[0] = 1; offsets[1] = 1;
		H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		status = H5Dwrite(dset_id, C_H5_REAL, memspace, filespace, plist_id, T.f);

		H5Dclose(dset_id);
		H5Sclose(filespace);
		H5Sclose(memspace);


		/* datasets for coordinates, Y */
		dims[0] = globalGridy.ny;
		filespace = H5Screate_simple(1, dims, NULL);
		dset_id = H5Dcreate(file_id, "y", C_H5_REAL, filespace, 
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		H5Sclose(filespace);

		offsets[0] = T.oy+1; counts[0] = T.ny-2;
		filespace = H5Dget_space(dset_id);
		H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		counts[0] = T.ny;
		memspace = H5Screate_simple(1, counts, NULL);
		counts[0] = T.ny-2; offsets[0] = 1;
		H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

		status = H5Dwrite(dset_id, C_H5_REAL, memspace, filespace, plist_id, gridy.f);

		H5Dclose(dset_id);
		H5Sclose(filespace);
		H5Sclose(memspace);


		/* datasets for coordinates, X */
		dims[0] = globalGridx.nx;
		filespace = H5Screate_simple(1, dims, NULL);
		dset_id = H5Dcreate(file_id, "x", C_H5_REAL, filespace, 
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		H5Sclose(filespace);

		offsets[0] = T.ox+1; counts[0] = T.nx-2;
		filespace = H5Dget_space(dset_id);
		H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		counts[0] = T.nx;
		memspace = H5Screate_simple(1, counts, NULL);
		counts[0] = T.nx-2; offsets[0] = 1;
		H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

		status = H5Dwrite(dset_id, C_H5_REAL, memspace, filespace, plist_id, gridx.f);

		H5Dclose(dset_id);
		H5Sclose(filespace);
		H5Sclose(memspace);


		/* Close all opened HDF5 handles */
		H5Pclose(plist_id);
		H5Fclose(file_id);

	}

	time2 = MPI_Wtime();
	if (iproc == 0) fprintf(stderr, "Writing took %g secs\n", time2-time1);

	if (iproc == 0) {
		config_destroy(&inputcfg);
	}

	MPI_Finalize();

	return 0;
}

