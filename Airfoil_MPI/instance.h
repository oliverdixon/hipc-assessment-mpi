//
// Created by od641 on 18/11/2025.
//

#ifndef HIPC_ASSESSMENT_INSTANCE_H
#define HIPC_ASSESSMENT_INSTANCE_H

#include <mpi.h>
#include <stdio.h>

#include "types.h"

struct region;

struct naca_specifier
{
    unsigned char maximum_camber;
    unsigned char edge_distance;
    unsigned char maximum_thickness;
};

struct instance
{
    const int rank;
    const int count;
    MPI_Comm cartesian_comm;

    const struct dim2 dim_extents;
    const struct dim2 cartesian_pos;
    const struct compute_dim2 problem_size;

    const struct naca_specifier naca_specifier;
    MPI_Datatype dim2_t;
};

struct instance instance_create();

void instance_destroy(struct instance *instance);

void instance_describe(const struct instance *instance, FILE *destination);

void instance_serialise_vtk(
    const struct instance *instance,
    const struct region *region,
    unsigned int max_subfile_digits,
    const char *subfile_prefix,
    const char *subfile_extension,
    FILE *destination);

struct dim2 instance_get_indentations(const struct instance *instance, struct dim2 own_size);

#endif // HIPC_ASSESSMENT_INSTANCE_H
