#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "heatfast.h"


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


