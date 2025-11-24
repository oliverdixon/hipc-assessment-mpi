//
// Created by od641 on 18/11/2025.
//

#ifndef HIPC_ASSESSMENT_INSTANCE_H
#define HIPC_ASSESSMENT_INSTANCE_H

#include <mpi.h>
#include <stdio.h>

#include "region.h"

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

    const int x_dim_extent;
    const int y_dim_extent;

    const int x_position;
    const int y_position;

    const float problem_width;
    const float problem_height;
    const struct naca_specifier naca_specifier;
};

struct instance instance_create();

void instance_describe(const struct instance *instance, FILE * destination);

struct dim2 instance_get_indentations(const struct instance *instance, struct dim2 own_size);

#endif // HIPC_ASSESSMENT_INSTANCE_H
