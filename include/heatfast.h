#ifndef __HEATFAST_H
#define __HEATFAST_H

#define IY 0
#define IX 1

#define ERR_CONFIG 1
#define ERR_SIZE_MISMATCH 2
#define ERR_CMDLINE_ARGUMENT 4
#define ERR_FILEOPER 8
#define ERR_CFGFILE 16
#define ERR_INVALID_OPTION 32
#define ERR_INTERNAL 65536

#define STR_MAXLEN 255

typedef double Real;               // NB! libconfig currently support double, not float
#define C_MPI_REAL MPI_DOUBLE
#define C_H5_REAL H5T_NATIVE_DOUBLE

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))


struct mpidata {
	MPI_Comm comm2d;         // cartesian communicator
	int mpicomm_ndims;       // number of cartesian dimensions (2)
	int mpicomm_reorder;     // is reordering allowed
	int mpicomm_periods[2];  // are boundaries periodic (no)
	int mpicomm_dims[2];     // size of dimensions
	int mpicoord[2];         // cartesian mpi coordinates of current proc
	int rank_neighb[4];      // ranks of neighbouring processors
};


struct config {
	int nx, ny, px, py;   // num of grid points; num of procs
	Real Lx, Ly;          // physical dimensions
	Real Kdiff;           // heatdiffusivity
	Real dt;              // time step
	int niter;            // num of iters
	char initfile[STR_MAXLEN];      // where from to read initial conditions
	int bctypes[4];       // boundary condition types
	Real bcvalues[4];     // boundary condition values
};


struct field {
	Real *f;               // field values
	char name[STR_MAXLEN]; // name of the field
	int nx, ny;            // size of the field (grid points)
	int ox, oy;            // origin of the field relative to the global grid origin
};


struct modeldata {
	Real time;                      // model time in seconds
	int timestep;                   // current discrete timestep
	int nfields;                    // number of fields in **fields
	struct field **fields;          // pointers to variable fields
	struct field **grids;           // pointers to local grids 
	struct field **globalGrids;     // pointers to global grids
};


void createMPIDatatypeConfig(struct config *c, MPI_Datatype *newtype);
int domainDecomp(int nx, int ny, int np, int *px, int *py);

void createFromField(struct field const *const f1, struct field *const f2);
void communicateHalos(struct mpidata *mpistate, struct field *fld);
void swapFields(struct field *const a, struct field *const b);
int nameField(struct field *f, char *name);

void readHdf5(struct field *fld, struct mpidata *mpistate, char *filename);
void writeFields(struct modeldata *mstate, struct mpidata *mpistate, struct config *globalConfig);
void printDomains(struct mpidata const *const mpistate, struct field const *const fld);

void readConfig(struct config *globalConfig, const char *configfile);



#endif
