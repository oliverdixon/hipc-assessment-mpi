//
// Created by od641 on 17/11/2025.
//

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "instance.h"
#include "region.h"

static void debug_print(
    compute_t * const * const array,
    const struct iterator h_bounds,
    const struct iterator v_bounds,
    const bool north_ghost,
    const bool south_ghost,
    const bool east_ghost,
    const bool west_ghost,
    FILE * const destination)
{
    const char h_delim[] = "           ";
    const char v_delim[] = "--------";

    if (north_ghost) {
        if (west_ghost)
            fputs(h_delim, destination);
        for (indexer_t h_idx = h_bounds.begin; h_idx < h_bounds.end - 1; ++h_idx)
            fprintf(destination, "%+10lf ", array[h_idx][0]);
        fprintf(destination, "%+10lf%s\n", array[h_bounds.end - 1][0], h_delim);
        if (west_ghost)
            fputs(h_delim, destination);

        for (indexer_t h_idx = h_bounds.begin; h_idx < h_bounds.end - 1; ++h_idx) {
            fputs(v_delim, destination);
            fputc(' ', destination);
        }

        fputs(v_delim, destination);
        if (east_ghost)
            fputs(h_delim, destination);
        fputc('\n', destination);
    }

    for (indexer_t v_idx = v_bounds.begin; v_idx < v_bounds.end; ++v_idx) {
        if (west_ghost)
            fprintf(destination, "%+10lf | ", array[0][v_idx]);
        for (indexer_t h_idx = h_bounds.begin; h_idx < h_bounds.end - 1; ++h_idx)
            fprintf(destination, "%+10lf ", array[h_idx][v_idx]);
        fprintf(destination, "%+10lf", array[h_bounds.end - 1][v_idx]);
        if (east_ghost)
            fprintf(destination, " | %+10lf", array[h_bounds.end][v_idx]);
        fputc('\n', destination);
    }

    if (south_ghost) {
        if (west_ghost)
            fputs(h_delim, destination);
        for (indexer_t h_idx = h_bounds.begin; h_idx < h_bounds.end - 1; ++h_idx) {
            fputs(v_delim, destination);
            fputc(' ', destination);
        }

        fputs(v_delim, destination);
        if (east_ghost)
            fputs(h_delim, destination);
        fputc('\n', destination);
        if (west_ghost)
            fputs(h_delim, destination);

        for (indexer_t h_idx = h_bounds.begin; h_idx < h_bounds.end - 1; ++h_idx)
            fprintf(destination, "%+10lf ", array[h_idx][v_bounds.end]);
        fprintf(destination, "%+10lf%s\n", array[h_bounds.end - 1][v_bounds.end], h_delim);

        if (east_ghost)
            fputs(h_delim, destination);
        fputc('\n', destination);
    }
}

static void serialise(const struct instance * const instance, const struct region * const region)
{
    static const unsigned int max_rank_digits = 2;
    assert(instance->count < (int) powf(10, max_rank_digits));

    static const char prefix[] = "flows";
    static const char suffix[] = "_00.vtr"; // TODO: time checkpoints indicated by suffix.

    chdir("./out/");

    FILE * const master_fp = instance->rank == 0 ? fopen("flows.pvtr", "w") : NULL;
    instance_serialise_vtk(instance, region, max_rank_digits, prefix, suffix, master_fp);

    if (master_fp != NULL)
        fclose(master_fp);

    static unsigned int prefix_length = sizeof(prefix) / sizeof(*prefix) - 1;
    static unsigned int suffix_length = sizeof(suffix) / sizeof(*suffix) - 1;

    char subfile_name[prefix_length + max_rank_digits + suffix_length + 1];
    sprintf(subfile_name, "%s%0*d%s", prefix, max_rank_digits, instance->rank, suffix);
    FILE * const subfile_fp = fopen(subfile_name, "w");
    region_serialise_vtk(region, instance, subfile_fp);
    fclose(subfile_fp);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    struct instance instance = instance_create();
    struct region region = region_create(&instance);

    instance_describe(&instance, stderr);

    region_describe(&region, stderr);
    region_initialise(&region, &instance);
    region_exchange(&region, MATRIX_FLAGS, &instance);

    static const compute_t max_simulation_runtime = 1.0;
    static const indexer_t sor_max_iterations = 100;
    static const compute_t sor_residual_epsilon = 0.001;
    static const indexer_t output_freq = 100;

    compute_t simulation_runtime = 0.0;
    indexer_t step_iteration = 0;

    // Collate the total fluid cell count, required when computing the L_2 residual norms.
    unsigned int fluid_cell_sum = 0;
    MPI_Allreduce(&region.fluid_cell_count, &fluid_cell_sum, 1, MPI_UNSIGNED, MPI_SUM, instance.cartesian_comm);

    while (simulation_runtime < max_simulation_runtime) {
        // \Delta_t timestep is fixed.
        region_apply_boundary_conditions(&region);
        // TODO: possibly need HX of velocities here? Since they can be changed following the boundary conditions.
        region_compute_tentative_velocities(&region, &instance);

        // Exchange tentative velocities for computation of Poisson term.
        region_exchange(&region, MATRIX_TENTATIVE_VELOCITY_X, &instance);
        region_exchange(&region, MATRIX_TENTATIVE_VELOCITY_Y, &instance);
        region_compute_poisson_source(&region, &instance);

        compute_t residual = INT_MAX;

        for (indexer_t sor_iteration = 0; sor_iteration < sor_max_iterations; ++sor_iteration) {
            // Perform an SOR cycle and halo-exchange the pressure matrix.
            region_sor_cycle(&region, &instance);

            // Compute the global residual L_2 norm given the cumulative residual and total fluid cell count.
            residual = region_compute_poisson_residual(&region);
            MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_COMPUTE, MPI_SUM, instance.cartesian_comm);
            residual = residual / fluid_cell_sum;

            if (fabs(residual) < sor_residual_epsilon * sor_residual_epsilon)
                break;
        }

        // Update and exchange the velocities.
        region_update_velocities(&region, &instance);
        region_exchange(&region, MATRIX_VELOCITY_X, &instance);
        region_exchange(&region, MATRIX_VELOCITY_Y, &instance);

        simulation_runtime += instance.timestep_duration;

        if (instance.rank == 0 && step_iteration % output_freq == 0)
            printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", step_iteration, simulation_runtime, residual);

        ++step_iteration;
    }

    serialise(&instance, &region);

    // TODO remove.
    char buf[16];
    sprintf(buf, "./v-%02d", instance.rank);
    FILE * fp = fopen(buf, "w");
    debug_print(region.velocity_x, region.h_exterior, region.v_exterior, region.region_flags & REGION_NORTH_GHOST,
        region.region_flags & REGION_SOUTH_GHOST, false, false, fp);
    fclose(fp);

    region_destroy(&region);
    instance_destroy(&instance);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
