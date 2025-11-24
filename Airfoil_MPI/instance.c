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

    int dims[] = {0, 0};
    const int periods[] = {0, 0};
    MPI_Comm cartesian_comm;

    MPI_Dims_create(count, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartesian_comm);

    int coords[2];
    MPI_Cart_coords(cartesian_comm, rank, 2, coords);

    const struct instance instance = {
            .rank = rank,
            .count = count,
            .cartesian_comm = cartesian_comm,

            .dim_extents.x = dims[0],
            .dim_extents.y = dims[1],

            .cartesian_pos.x = coords[0],
            .cartesian_pos.y = coords[1],

            .problem_size.x = 4.0,
            .problem_size.y = 1.0,
            .naca_specifier = {.maximum_camber = 2, .edge_distance = 4, .maximum_thickness = 12}};

    return instance;
}

void instance_describe(const struct instance *instance, FILE *const destination)
{
    fprintf(destination,
            "Instance statistics:\n\t"
            "Rank: %d / %d\n\t"
            "Dimensions: (%d, %d)\n\t"
            "Cartesian co-ordinates: (%d, %d)\n\t"
            "Global problem size: (%lf, %lf)\n\t"
            "NACA specifier: %2d%1d%1d\n",

            instance->rank, instance->count - 1, instance->dim_extents.x, instance->dim_extents.y,
            instance->cartesian_pos.x, instance->cartesian_pos.y, instance->problem_size.x, instance->problem_size.y,
            instance->naca_specifier.maximum_camber, instance->naca_specifier.edge_distance,
            instance->naca_specifier.maximum_thickness);
}

struct dim2 instance_get_indentations(const struct instance *instance, const struct dim2 own_size)
{
    struct dim2 indents;
    const struct dim2 size = {
            .x = instance->cartesian_pos.x == 0 ? 0 : own_size.x, .y = instance->cartesian_pos.y == 0 ? 0 : own_size.y};

    int fix_dimensions[] = {1, 0};
    MPI_Comm fixed_dim_comm;
    MPI_Cart_sub(instance->cartesian_comm, fix_dimensions, &fixed_dim_comm); // Fixed on X; row communicator.
    MPI_Scan(&size.x, &indents.x, 1, MPI_UNSIGNED, MPI_SUM, fixed_dim_comm);

    fix_dimensions[0] = 0;
    fix_dimensions[1] = 1;
    MPI_Cart_sub(instance->cartesian_comm, fix_dimensions, &fixed_dim_comm); // Fixed on Y; column communicator.
    MPI_Scan(&size.y, &indents.y, 1, MPI_UNSIGNED, MPI_SUM, fixed_dim_comm);

    return indents;
}
