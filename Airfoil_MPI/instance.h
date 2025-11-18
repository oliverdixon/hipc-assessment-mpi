//
// Created by od641 on 18/11/2025.
//

#ifndef HIPC_ASSESSMENT_INSTANCE_H
#define HIPC_ASSESSMENT_INSTANCE_H

#include <mpi.h>
#include <stdio.h>

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

    const float global_problem_width;
    const float global_problem_height;
    const struct naca_specifier naca_specifier;
};

struct instance instance_create();

void instance_describe(const struct instance *instance, FILE * destination);

#endif // HIPC_ASSESSMENT_INSTANCE_H
