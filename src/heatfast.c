#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <hdf5.h>
#include <libconfig.h>

#define IY 0
#define IX 1

#define ERR_CONFIG 1
#define ERR_SIZE_MISMATCH 2
#define ERR_CMDLINE_ARGUMENT 4
#define ERR_FILEOPER 8
#define ERR_CFGFILE 16
#define ERR_INTERNAL 65536

#define STR_MAXLEN 255

typedef double Real;               // NB! libconfig currently support double, not float
#define C_MPI_REAL MPI_DOUBLE
#define C_H5_REAL H5T_NATIVE_DOUBLE



/* MPIDATA and related routines */

struct mpidata {
	MPI_Comm comm2d;
	int mpicomm_ndims;
	int mpicomm_reorder;
	int mpicomm_periods[2];
	int mpicomm_dims[2];
	int mpicoord[2];
	int rank_neighb[4];
};



/* CONFIG and related routines */

struct config {
	int nx, ny, px, py;   // num of grid points; num of procs
	Real Lx, Ly;          // physical dimensions
	Real Kdiff;           // heatdiffusivity
	Real dt;              // time step
	int niter;            // num of iters

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


/* FIELD and related routines */

struct field {
	Real *f;               // field values
	char name[STR_MAXLEN]; // name of the field
	int nx, ny;            // size of the field (grid points)
	int ox, oy;            // origin of the field relative to the global grid origin
};

void createFromField(struct field const *const f1, struct field *const f2) {
	strncpy(f2->name, f1->name, STR_MAXLEN);
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

int readNumsFile(char *filename, Real **data, size_t *datasize) {
	const size_t blocksize = 256;
	FILE *fp;
	size_t bufsize = 0;
	size_t nread, eread, cread;
	Real *buf = NULL;
	char elem[blocksize];

	if (*data != NULL) {
		fprintf(stderr, "readNumsFile needs non-allocated mem pointer!\n");
		return ERR_INTERNAL;
	}

	fp = fopen(filename, "r");
	if (fp == NULL) {
		fprintf(stderr, "Error opening file '%s'\n", filename);
		return ERR_FILEOPER;
	}

	eread = 0;
	nread = 0;
	while (1) {
		cread = fread(&elem[eread], 1, 1, fp);
		if (eread == 0 && cread == 1 && elem[0] <= 32) continue;  // pre-whitespace or something like that...
		if ((cread == 1 && elem[eread] <= 32) || (cread == 0 && eread > 0)) {  
			// post-whitespace, interpret!
			elem[eread] = '\0';
			if (nread+1 > bufsize) {
				bufsize += blocksize;
				buf = realloc(buf, bufsize);
			}
			buf[nread] = atof(elem);
			nread++;
			eread = 0;
		} else if (cread == 1) {
			eread++;
		} else {
			// eof
			break;
		}
	}

	fclose(fp);

	*data = buf;
	*datasize = nread;

	return 0;
}

int readFieldFile(struct field *fld, char *modelname) {
	char datafile[STR_MAXLEN];
	Real *indata = NULL;
	size_t ncoords;
	int ierr, ny, nx;
	nx = fld->nx;
	ny = fld->ny;

	snprintf(datafile, STR_MAXLEN, "%s.%s.txt", modelname, fld->name);

	ierr = readNumsFile(datafile, &indata, &ncoords);
	if (ierr != 0) return ierr;

	if (ncoords != ny*nx) {
		fprintf(stderr, "readFieldFile(): Reading from %s: ncoords (%ld) != ny*nx (%ld)\n", datafile, ncoords, ny*nx);
		return ERR_SIZE_MISMATCH;
	}

	for (int i = 0; i < ny; i++) {
		for (int j = 0; j < nx; j++) {
			fld->f[i*nx + j] = indata[i*nx + j];
		}
	}

	return 0;
}

void readHdf5(struct field *fld, struct mpidata *mpistate, struct config *globalConfig) {

	hid_t plist_id, dset_id, filespace, memspace, file_id;
	hsize_t counts[2], offsets[2];
	char fieldpath[STR_MAXLEN];
	snprintf(fieldpath, STR_MAXLEN, "/%s", fld->name);

	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(plist_id, mpistate->comm2d, MPI_INFO_NULL);
	file_id = H5Fopen("data.h5", H5F_ACC_RDONLY, plist_id);
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






void writeFields(int nfields, struct field **fields, 
		struct field *gridy, struct field *gridx, 
		struct mpidata *mpistate, struct config *globalConfig) {
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

	for (int ifield = 0; ifield < nfields; ifield++) {
		/* Create the dataset */
		dims[0] = ny;
		dims[1] = nx;
		filespace = H5Screate_simple(2, dims, NULL);
		dset_id = H5Dcreate(file_id, fields[ifield]->name, C_H5_REAL, filespace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		H5Sclose(filespace);

		/* Select a hyperslab of the file dataspace */
		offsets[0] = fields[ifield]->oy+1; offsets[1] = fields[ifield]->ox+1;
		counts[0] = fields[ifield]->ny-2; counts[1] = fields[ifield]->nx-2;
		filespace = H5Dget_space(dset_id);
		H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		counts[0] = fields[ifield]->ny; counts[1] = fields[ifield]->nx;
		memspace = H5Screate_simple(2, counts, NULL);
		counts[0] = fields[ifield]->ny-2; counts[1] = fields[ifield]->nx-2;
		offsets[0] = 1; offsets[1] = 1;
		H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

		H5Dwrite(dset_id, C_H5_REAL, memspace, filespace, plist_id, fields[ifield]->f);  // status =

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

	offsets[0] = gridy->oy+1; counts[0] = gridy->ny-2;
	filespace = H5Dget_space(dset_id);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

	counts[0] = gridy->ny;
	memspace = H5Screate_simple(1, counts, NULL);
	counts[0] = gridy->ny-2; offsets[0] = 1;
	H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	H5Dwrite(dset_id, C_H5_REAL, memspace, filespace, plist_id, gridy->f);  // status =

	H5Dclose(dset_id);
	H5Sclose(filespace);
	H5Sclose(memspace);


	/* datasets for coordinates, X */
	dims[0] = nx;
	filespace = H5Screate_simple(1, dims, NULL);
	dset_id = H5Dcreate(file_id, "_coord_x", C_H5_REAL, filespace, 
		H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Sclose(filespace);

	offsets[0] = gridx->ox+1; counts[0] = gridx->nx-2;
	filespace = H5Dget_space(dset_id);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

	counts[0] = gridx->nx;
	memspace = H5Screate_simple(1, counts, NULL);
	counts[0] = gridx->nx-2; offsets[0] = 1;
	H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	H5Dwrite(dset_id, C_H5_REAL, memspace, filespace, plist_id, gridx->f); // status = 

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
	config_t inputcfg;
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
		if (config_lookup_int(&inputcfg, "mpi.px", &globalConfig->px) != CONFIG_TRUE) { fprintf(stderr, "config option mpi.px needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_int(&inputcfg, "mpi.py", &globalConfig->py) != CONFIG_TRUE) { fprintf(stderr, "config option mpi.py needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_float(&inputcfg, "phys.kdiff", &globalConfig->Kdiff) != CONFIG_TRUE) { fprintf(stderr, "config option phys.kdiff needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
		if (config_lookup_int(&inputcfg, "run.iter", &globalConfig->niter) != CONFIG_TRUE) { fprintf(stderr, "config option run.iter needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }
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

	//int mpicoord[2];

	struct config globalConfig;

	struct field T, Told;
	struct field globalGridx, globalGridy, gridx, gridy;
	struct field **allFields;

	Real dy, dx;
	MPI_Datatype columntype;


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

	/*
	 * === if one wants to read the grid in, too:
	 * nameField(&globalGridx, "globalGridx");
	 * ierr = readFieldFile(&globalGridx, "test");
	 * if (ierr != 0) MPI_Abort(MPI_COMM_WORLD, ierr);
	 * printDomains(&mpistate, &gridx);
	 */

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
	T.f = calloc(sizeof(Real), T.nx*T.ny);

	createFromField(&T, &Told);

	nameField(&T, "T");
	nameField(&Told, "Told");

	allFields = malloc(sizeof(struct field *) * 2);
	allFields[0] = &T;
	allFields[1] = &Told;

	readHdf5(&T, &mpistate, &globalConfig);

		/* TODO TODO TODO : this to a subroutine (and the one within the time loop */
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

	printDomains(&mpistate, &T);
	return 0;

	/* setup initial values */
	for (int i = 0; i < gridy.ny; i++) {
		for (int j = 0; j < gridx.nx; j++) {
			T.f[i*gridx.nx + j] = gen_field_T(gridy.f[i], gridx.f[j]);
		}
	}

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

	writeFields(2, allFields, &gridy, &gridx, &mpistate, &globalConfig);

	MPI_Finalize();

	return 0;
}

