#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <hdf5.h>
#include <libconfig.h>
#include "heatfast.h"



/* MPIDATA and related routines */



/* CONFIG and related routines */


void createMPIDatatypeConfig(struct config *c, MPI_Datatype *newtype) {
	MPI_Datatype strarray, intarray, realarray;
	MPI_Aint configDisplacements[12];

	MPI_Type_vector(1, STR_MAXLEN, 1, MPI_CHAR, &strarray);
	MPI_Type_vector(1, 4, 1, MPI_INT, &intarray);
	MPI_Type_vector(1, 4, 1, C_MPI_REAL, &realarray);

	MPI_Type_commit(&strarray);
	MPI_Type_commit(&intarray);
	MPI_Type_commit(&realarray);

	const int configStructLen = 12;
	const int configBlockLens[12] = {
		1, 1, 1, 1,
		1, 1, 
		1,
		1,
		1,
		1,
		1,
		1
	};
	const MPI_Datatype configDatatypes[12] = {
		MPI_INT, MPI_INT, MPI_INT, MPI_INT,
		C_MPI_REAL, C_MPI_REAL,
		C_MPI_REAL,
		C_MPI_REAL,
		MPI_INT,
		strarray,
		intarray,
		realarray
	};

	configDisplacements[0] = (long int)&c->nx;
	configDisplacements[1] = (long int)&c->ny       - configDisplacements[0];
	configDisplacements[2] = (long int)&c->px       - configDisplacements[0];
	configDisplacements[3] = (long int)&c->py       - configDisplacements[0];
	configDisplacements[4] = (long int)&c->Lx       - configDisplacements[0];
	configDisplacements[5] = (long int)&c->Ly       - configDisplacements[0];
	configDisplacements[6] = (long int)&c->Kdiff    - configDisplacements[0];
	configDisplacements[7] = (long int)&c->dt       - configDisplacements[0];
	configDisplacements[8] = (long int)&c->niter    - configDisplacements[0];
	configDisplacements[9] = (long int)&c->initfile - configDisplacements[0];
	configDisplacements[10]= (long int)&c->bctypes  - configDisplacements[0];
	configDisplacements[11]= (long int)&c->bcvalues - configDisplacements[0];
	configDisplacements[0] = 0;

	MPI_Type_create_struct(configStructLen, configBlockLens, configDisplacements, configDatatypes, newtype);

	MPI_Type_free(&strarray);
	MPI_Type_free(&intarray);
	MPI_Type_free(&realarray);
}


int domainDecomp(int nx, int ny, int np, int *px, int *py) {
	int rx, ry, sx, sy, sp;
	double b2log;

	b2log = log((double)nx) / log(2.0);
	rx = (int)b2log;
	if (b2log != rx) {
		fprintf(stderr, "nx (%d) is not an exponent of two\n", nx);
		return ERR_SIZE_MISMATCH|ERR_CFGFILE;
	}

	b2log = log((double)ny) / log(2.0);
	ry = (int)b2log;
	if (b2log != ry) {
		fprintf(stderr, "ny (%d) is not an exponent of two\n", ny);
		return ERR_SIZE_MISMATCH|ERR_CFGFILE;
	}

	b2log = log((double)np) / log(2.0);
	sp = (int)b2log;
	if (b2log != sp) {
		fprintf(stderr, "np (%d) is not an exponent of two\n", np);
		return ERR_SIZE_MISMATCH|ERR_CFGFILE;
	}

	sy = (ry - rx + sp) / 2;
	sx = sp - sy;

	if (sy + sx != sp) {
		fprintf(stderr, "domainDecomp(): sy + sx != sp\n");
		return ERR_INTERNAL;
	}

	*px = pow(2.0, sx);
	*py = pow(2.0, sy);

	if (*py * *px != np) {
		fprintf(stderr, "domainDecomp(): py + px != np\n");
		return ERR_INTERNAL;
	}

	return 0;
}


/* FIELD and related routines */


void createFromField(struct field const *const f1, struct field *const f2) {
	strncpy(f2->name, f1->name, STR_MAXLEN);
	f2->nx = f1->nx; f2->ny = f1->ny;
	f2->ox = f1->ox; f2->oy = f1->oy;
	f2->f = malloc(sizeof(Real) * f2->nx * f2->ny);
}

void communicateHalos(struct mpidata *mpistate, struct field *fld) {
	MPI_Datatype columntype;

	MPI_Cart_shift(mpistate->comm2d, IY, 1, &mpistate->rank_neighb[0], &mpistate->rank_neighb[1]);
	MPI_Cart_shift(mpistate->comm2d, IX, 1, &mpistate->rank_neighb[2], &mpistate->rank_neighb[3]);
	for (int i = 0; i < 4; i++) if (mpistate->rank_neighb[i] < 0) mpistate->rank_neighb[i] = MPI_PROC_NULL;

	// send lower and upper row
	MPI_Sendrecv(&fld->f[fld->nx*(fld->ny-2)], fld->nx, C_MPI_REAL, mpistate->rank_neighb[1], 0, 
			&fld->f[0], fld->nx, C_MPI_REAL, mpistate->rank_neighb[0], 0, mpistate->comm2d, MPI_STATUS_IGNORE);
	MPI_Sendrecv(&fld->f[fld->nx*1], fld->nx, C_MPI_REAL, mpistate->rank_neighb[0], 0, 
			&fld->f[fld->nx*(fld->ny-1)], fld->nx, C_MPI_REAL, mpistate->rank_neighb[1], 0, mpistate->comm2d, MPI_STATUS_IGNORE);

	// send right and left column
	MPI_Type_vector(fld->ny, 1, fld->nx, C_MPI_REAL, &columntype);
	MPI_Type_commit(&columntype);
	MPI_Sendrecv(&fld->f[fld->nx-2], 1, columntype, mpistate->rank_neighb[3], 0, 
			&fld->f[0], 1, columntype, mpistate->rank_neighb[2], 0, mpistate->comm2d, MPI_STATUS_IGNORE);
	MPI_Sendrecv(&fld->f[1], 1, columntype, mpistate->rank_neighb[2], 0, 
			&fld->f[fld->nx-1], 1, columntype, mpistate->rank_neighb[3], 0, mpistate->comm2d, MPI_STATUS_IGNORE);
	MPI_Type_free(&columntype);
}


void swapFields(struct field *const a, struct field *const b) {
	Real *tmp;
	int tmpox, tmpoy;
	char tmpname[STR_MAXLEN];

	if (a->nx != b->nx || a->ny != b->ny) {
		fprintf(stderr, "cannot swap fields of different sizes\n");
		MPI_Abort(MPI_COMM_WORLD, ERR_SIZE_MISMATCH);
		exit(ERR_SIZE_MISMATCH);
	}

	strncpy(tmpname, a->name, STR_MAXLEN);
	strncpy(a->name, b->name, STR_MAXLEN);
	strncpy(b->name, tmpname, STR_MAXLEN);

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

int nameField(struct field *f, char *name) {
	int hasNull = 0;


	for (int i = 0; i < STR_MAXLEN; i++) {
		if (name[i] == '\0') {
			hasNull = 1;
			break;
		}
	}

	if (hasNull) strncpy(f->name, name, STR_MAXLEN);
	else return ERR_INTERNAL;

	return 0;
}


void readHdf5(struct field *fld, struct mpidata *mpistate, char *filename) {

	hid_t plist_id, dset_id, filespace, memspace, file_id;
	hsize_t counts[2], offsets[2];
	char fieldpath[STR_MAXLEN];
	snprintf(fieldpath, STR_MAXLEN, "/%s", fld->name);

	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(plist_id, mpistate->comm2d, MPI_INFO_NULL);
	file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
	if (file_id < 0) {
		fprintf(stderr, "readHdf5(): Cannot open file %s\n", filename);
		MPI_Abort(MPI_COMM_WORLD, ERR_FILEOPER);
		exit(ERR_FILEOPER);
	}
	H5Pclose(plist_id);

	dset_id = H5Dopen(file_id, fieldpath, H5P_DEFAULT);

	offsets[0] = fld->oy+1; offsets[1] = fld->ox+1;
	counts[0] = fld->ny-2; counts[1] = fld->nx-2;
	filespace = H5Dget_space(dset_id);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

	counts[0] = fld->ny; counts[1] = fld->nx;
	memspace = H5Screate_simple(2, counts, NULL);
	counts[0] = fld->ny-2; counts[1] = fld->nx-2;
	offsets[0] = 1; offsets[1] = 1;
	H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);
	H5Dread(dset_id, C_H5_REAL, memspace, filespace, H5P_DEFAULT, fld->f);

	H5Dclose(dset_id);
	H5Fclose(file_id);
}



void writeFields(struct modeldata *mstate, struct mpidata *mpistate, struct config *globalConfig) {
	//herr_t status;
	hid_t plist_id, dset_id, filespace, memspace, file_id;
	hsize_t dims[2], counts[2], offsets[2];
	int ny, nx;

	/* create hdf5 file */

	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(plist_id, mpistate->comm2d, MPI_INFO_NULL);
	file_id = H5Fcreate("data.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
	H5Pclose(plist_id);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	ny = globalConfig->ny;
	nx = globalConfig->nx;


	/* create datasets */

	for (int ifield = 0; ifield < mstate->nfields; ifield++) {
		/* Create the dataset */
		dims[0] = ny;
		dims[1] = nx;
		filespace = H5Screate_simple(2, dims, NULL);
		dset_id = H5Dcreate(file_id, mstate->fields[ifield]->name, C_H5_REAL, filespace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		H5Sclose(filespace);

		/* Select a hyperslab of the file dataspace */
		offsets[0] = mstate->fields[ifield]->oy+1; offsets[1] = mstate->fields[ifield]->ox+1;
		counts[0] = mstate->fields[ifield]->ny-2; counts[1] = mstate->fields[ifield]->nx-2;
		filespace = H5Dget_space(dset_id);
		H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		counts[0] = mstate->fields[ifield]->ny; counts[1] = mstate->fields[ifield]->nx;
		memspace = H5Screate_simple(2, counts, NULL);
		counts[0] = mstate->fields[ifield]->ny-2; counts[1] = mstate->fields[ifield]->nx-2;
		offsets[0] = 1; offsets[1] = 1;
		H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		H5Dwrite(dset_id, C_H5_REAL, memspace, filespace, plist_id, mstate->fields[ifield]->f);  // status =

		H5Dclose(dset_id);
		H5Sclose(filespace);
		H5Sclose(memspace);
	}

	/* datasets for coordinates, Y */
	dims[0] = ny;
	filespace = H5Screate_simple(1, dims, NULL);
	dset_id = H5Dcreate(file_id, "_coord_y", C_H5_REAL, filespace, 
		H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Sclose(filespace);

	offsets[0] = mstate->grids[IY]->oy+1; counts[0] = mstate->grids[IY]->ny-2;
	filespace = H5Dget_space(dset_id);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

	counts[0] = mstate->grids[IY]->ny;
	memspace = H5Screate_simple(1, counts, NULL);
	counts[0] = mstate->grids[IY]->ny-2; offsets[0] = 1;
	H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	H5Dwrite(dset_id, C_H5_REAL, memspace, filespace, plist_id, mstate->grids[IY]->f);  // status =

	H5Dclose(dset_id);
	H5Sclose(filespace);
	H5Sclose(memspace);


	/* datasets for coordinates, X */
	dims[0] = nx;
	filespace = H5Screate_simple(1, dims, NULL);
	dset_id = H5Dcreate(file_id, "_coord_x", C_H5_REAL, filespace, 
		H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Sclose(filespace);

	offsets[0] = mstate->grids[IX]->ox+1; counts[0] = mstate->grids[IX]->nx-2;
	filespace = H5Dget_space(dset_id);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

	counts[0] = mstate->grids[IX]->nx;
	memspace = H5Screate_simple(1, counts, NULL);
	counts[0] = mstate->grids[IX]->nx-2; offsets[0] = 1;
	H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	H5Dwrite(dset_id, C_H5_REAL, memspace, filespace, plist_id, mstate->grids[IX]->f); // status = 

	H5Dclose(dset_id);
	H5Sclose(filespace);
	H5Sclose(memspace);


	/* Close all opened HDF5 handles */
	H5Pclose(plist_id);
	H5Fclose(file_id);

}



/* UTILITIES */

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

void readConfig(struct config *globalConfig, const char *configfile) {
	FILE *cfgfp;
	int iproc, nproc;
	const char *tmpstring;
	config_t inputcfg;
	config_setting_t *bctypes, *bcvalues;
	MPI_Datatype globalConfigMPIType;

	MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	/* read config and read initial fields */
	if (iproc == 0) {

		cfgfp = fopen(configfile, "r");
		if (cfgfp == NULL) {
			fprintf(stderr, "ERROR: open file %s\n", configfile);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Abort(MPI_COMM_WORLD, ERR_FILEOPER|ERR_CMDLINE_ARGUMENT);
			exit(ERR_FILEOPER|ERR_CMDLINE_ARGUMENT);
		}
		config_init(&inputcfg);
		config_read(&inputcfg, cfgfp);
		fclose(cfgfp);

		if (config_lookup_int(&inputcfg, "grid.nx", &globalConfig->nx) != CONFIG_TRUE) { fprintf(stderr, "config option grid.nx needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_int(&inputcfg, "grid.ny", &globalConfig->ny) != CONFIG_TRUE) { fprintf(stderr, "config option grid.ny needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_float(&inputcfg, "grid.Lx", &globalConfig->Lx) != CONFIG_TRUE) { fprintf(stderr, "config option grid.Lx needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_float(&inputcfg, "grid.Ly", &globalConfig->Ly) != CONFIG_TRUE) { fprintf(stderr, "config option grid.Ly needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_float(&inputcfg, "phys.kdiff", &globalConfig->Kdiff) != CONFIG_TRUE) { fprintf(stderr, "config option phys.kdiff needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_float(&inputcfg, "run.dt", &globalConfig->dt) != CONFIG_TRUE) { fprintf(stderr, "config option run.dt needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_string(&inputcfg, "initcond.init_file", &tmpstring) != CONFIG_TRUE) { fprintf(stderr, "config option initcond.init_file needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		strncpy(globalConfig->initfile, tmpstring, STR_MAXLEN);
		globalConfig->initfile[STR_MAXLEN-1] = '\0';
		if (config_lookup_int(&inputcfg, "run.iter", &globalConfig->niter) != CONFIG_TRUE) { fprintf(stderr, "config option run.iter needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }

		bctypes = config_lookup(&inputcfg, "bccond.types");
		if (bctypes == NULL || config_setting_is_array(bctypes) != CONFIG_TRUE) { fprintf(stderr, "config option bccond.types needed but not found or wrong type\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		bcvalues = config_lookup(&inputcfg, "bccond.values");
		if (bcvalues == NULL || config_setting_is_array(bctypes) != CONFIG_TRUE) { fprintf(stderr, "config option bccond.values needed but not found or wrong type\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		
		for (int i = 0; i < 4; i++) {
			globalConfig->bctypes[i] = config_setting_get_int_elem(bctypes, i);
			globalConfig->bcvalues[i] = config_setting_get_float_elem(bcvalues, i);
		}
	}
	createMPIDatatypeConfig(globalConfig, &globalConfigMPIType);
	MPI_Type_commit(&globalConfigMPIType);
	if (iproc == 0) {
		for (int i = 1; i < nproc; i++) {
			MPI_Send(globalConfig, 1, globalConfigMPIType, i, 0, MPI_COMM_WORLD);
		}
	} else {
		MPI_Recv(globalConfig, 1, globalConfigMPIType, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Type_free(&globalConfigMPIType);
	if (iproc == 0) {
		config_destroy(&inputcfg);
	}
}




/* MAIN PROGRAM */

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

