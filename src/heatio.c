#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <hdf5.h>
#include <libconfig.h>
#include "heatfast.h"


void readHdf5(struct field *fld, struct mpidata *mpistate, char *filename) {
	hid_t plist_id, dset_id, filespace, memspace, file_id;
	hsize_t counts[2], offsets[2];
	char fieldpath[STR_MAXLEN];
	snprintf(fieldpath, STR_MAXLEN, "/%s", fld->name);

	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	if (plist_id < -1) {
		fprintf(stderr, "readHdf5(): H5Pcreate() failed\n");
		MPI_Abort(MPI_COMM_WORLD, ERR_INTERNAL);
		exit(ERR_INTERNAL);
	}
	if (H5Pset_fapl_mpio(plist_id, mpistate->comm2d, MPI_INFO_NULL) < 0) {
		fprintf(stderr, "readHdf5(): H5Pset_fapl_mpio() failed\n");
		MPI_Abort(MPI_COMM_WORLD, ERR_INTERNAL);
		exit(ERR_INTERNAL);
	}
	file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id); 

	if (file_id < 0) {
		fprintf(stderr, "readHdf5(): Cannot open file %s\n", filename);
		MPI_Abort(MPI_COMM_WORLD, ERR_FILEOPER);
		exit(ERR_FILEOPER);
	}
	H5Pclose(plist_id);

	dset_id = H5Dopen(file_id, fieldpath, H5P_DEFAULT);

	/* If field is one-dimensional, all processes in shortest
	 * dimension will read in the values */
	if (fld->ny == 1) {
		offsets[0] = 0;
		counts[0] = 1;
	} else {
		offsets[0] = fld->oy+1; 
		counts[0] = fld->ny-2;
	}
	
	if (fld->nx == 1) {
		offsets[1] = 0;
		counts[1] = 1;
	} else {
		offsets[1] = fld->ox+1;
		counts[1] = fld->nx-2;
	}
	
	filespace = H5Dget_space(dset_id);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);
	counts[0] = fld->ny; counts[1] = fld->nx;
	memspace = H5Screate_simple(2, counts, NULL);

	if (fld->ny == 1) {
		offsets[0] = 0;
		counts[0] = 1;
	} else {
		offsets[0] = 1;
		counts[0] = fld->ny-2;
	}
	
	if (fld->nx == 1) {
		offsets[1] = 0;
		counts[1] = 1;
	} else {
		offsets[1] = 1;
		counts[1] = fld->nx-2;
	}
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
		if (config_lookup_int(&inputcfg, "surfaces.nsurfaces", &globalConfig->nsurfaces) != CONFIG_TRUE) { fprintf(stderr, "config option surfaces.nsurfaces needed but not found\n"); MPI_Abort(MPI_COMM_WORLD, ERR_CFGFILE); exit(ERR_CFGFILE); }

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


