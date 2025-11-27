//
// Created by od641 on 27/11/2025.
//

#ifndef HIPC_ASSESSMENT_EXCHANGER_H
#define HIPC_ASSESSMENT_EXCHANGER_H

#include <mpi.h>

#include "types.h"

enum cell_flags;

struct exchanger
{
    MPI_Datatype row_t;
    MPI_Datatype col_t;
    MPI_Comm comm;

    const void * north_row;
    void * north_ghost;

    const void * south_row;
    void * south_ghost;

    const void * east_col;
    void * east_ghost;

    const void * west_col;
    void * west_ghost;
};

void exchanger_exchange(
    const struct exchanger *exchanger,
    const struct neighbours *neighbour_ranks);

struct exchanger exchanger_create_compute(
    compute_t * const * data,
    struct iterator h_bounds,
    struct iterator v_bounds,
    MPI_Comm comm,
    MPI_Datatype row_t,
    MPI_Datatype col_t);

struct exchanger exchanger_create_flags(
    enum cell_flags * const * data,
    struct iterator h_bounds,
    struct iterator v_bounds,
    MPI_Comm comm,
    MPI_Datatype row_t,
    MPI_Datatype col_t);

#endif // HIPC_ASSESSMENT_EXCHANGER_H
