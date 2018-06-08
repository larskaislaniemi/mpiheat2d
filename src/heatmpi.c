#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "heatfast.h"


void createMPIDatatypeConfig(struct config *c, MPI_Datatype *newtype) {
	MPI_Datatype strarray, intarray, realarray;
	MPI_Aint configDisplacements[13];

	MPI_Type_vector(1, STR_MAXLEN, 1, MPI_CHAR, &strarray);
	MPI_Type_vector(1, 4, 1, MPI_INT, &intarray);
	MPI_Type_vector(1, 4, 1, C_MPI_REAL, &realarray);

	MPI_Type_commit(&strarray);
	MPI_Type_commit(&intarray);
	MPI_Type_commit(&realarray);

	const int configStructLen = 13;
	const int configBlockLens[13] = {
		1, 1, 1, 1,
		1, 1, 
		1,
		1,
		1,
		1,
		1,
		1,
		1
	};
	const MPI_Datatype configDatatypes[13] = {
		MPI_INT, MPI_INT, MPI_INT, MPI_INT,
		C_MPI_REAL, C_MPI_REAL,
		C_MPI_REAL,
		C_MPI_REAL,
		MPI_INT,
		strarray,
		intarray,
		realarray,
		MPI_INT
	};

	configDisplacements[0] = (long int)&c->nx;
	configDisplacements[1] = (long int)&c->ny        - configDisplacements[0];
	configDisplacements[2] = (long int)&c->px        - configDisplacements[0];
	configDisplacements[3] = (long int)&c->py        - configDisplacements[0];
	configDisplacements[4] = (long int)&c->Lx        - configDisplacements[0];
	configDisplacements[5] = (long int)&c->Ly        - configDisplacements[0];
	configDisplacements[6] = (long int)&c->Kdiff     - configDisplacements[0];
	configDisplacements[7] = (long int)&c->dt        - configDisplacements[0];
	configDisplacements[8] = (long int)&c->niter     - configDisplacements[0];
	configDisplacements[9] = (long int)&c->initfile  - configDisplacements[0];
	configDisplacements[10]= (long int)&c->bctypes   - configDisplacements[0];
	configDisplacements[11]= (long int)&c->bcvalues  - configDisplacements[0];
	configDisplacements[12]= (long int)&c->nsurfaces - configDisplacements[0];
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


