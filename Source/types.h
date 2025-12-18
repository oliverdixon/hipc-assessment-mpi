//
// Created by od641 on 24/11/2025.
//

#ifndef HIPC_ASSESSMENT_TYPES_H
#define HIPC_ASSESSMENT_TYPES_H

#include <mpi.h>

typedef unsigned int indexer_t;
typedef double compute_t;

#define MPI_COMPUTE (MPI_DOUBLE)

struct iterator
{
    indexer_t begin;
    indexer_t end;
};

struct compute_dim2
{
    compute_t x;
    compute_t y;
};

struct dim2
{
    indexer_t x;
    indexer_t y;
};

struct neighbours
{
    int north;
    int east;
    int south;
    int west;
};

#endif // HIPC_ASSESSMENT_TYPES_H
