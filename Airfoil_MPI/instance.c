//
// Created by od641 on 18/11/2025.
//

#include "instance.h"

struct instance instance_create()
{
    int rank;
    int count;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &count);

    int dims[] = { 0, 0 };
    const int periods[] = { 0, 0 };
    MPI_Comm cartesian_comm;

    MPI_Dims_create(count, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartesian_comm);

    int coords[2];
    MPI_Cart_coords(cartesian_comm, rank, 2, coords);

    const struct instance instance = {
        .rank = rank,
        .count = count,
        .cartesian_comm = cartesian_comm,

        .x_dim_extent = dims[0],
        .y_dim_extent = dims[1],

        .x_position = coords[0],
        .y_position = coords[1],

        .global_problem_width = 4.0f,
        .global_problem_height = 1.0f,
        .naca_specifier = {
            .maximum_camber = 2,
            .edge_distance = 4,
            .maximum_thickness = 12
        }
    };

    return instance;
}

void instance_describe(const struct instance *instance, FILE *const destination)
{
    fprintf(destination, "Instance statistics:\n\t"
                         "Rank: %d / %d\n\t"
                         "Dimensions: (%d, %d)\n\t"
                         "Cartesian co-ordinates: (%d, %d)\n\t"
                         "Global problem size: (%lf, %lf)\n\t"
                         "NACA specifier: %2d%1d%1d\n",

                         instance->rank, instance->count - 1,
                         instance->x_dim_extent, instance->y_dim_extent,
                         instance->x_position, instance->y_position,
                         instance->global_problem_width, instance->global_problem_height,
                         instance->naca_specifier.maximum_camber, instance->naca_specifier.edge_distance,
                            instance->naca_specifier.maximum_thickness);
}
