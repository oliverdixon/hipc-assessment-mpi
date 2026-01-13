//
// Created by od641 on 17/11/2025.
//

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "instance.h"
#include "region.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    struct instance instance = instance_create();
    struct region region = region_create(&instance);

    instance_describe(&instance, stderr);

    region_describe(&region, stderr);
    region_initialise(&region, &instance);
    region_exchange(&region, MATRIX_FLAGS, &instance);

    static const compute_t max_simulation_runtime = 2.0;
    static const indexer_t sor_max_iterations = 100;
    static const compute_t sor_residual_epsilon = 0.001;
    static const indexer_t output_freq = 100;

    compute_t simulation_runtime = 0.0;
    indexer_t step_iteration = 0;

    // Collate the total fluid cell count, required when computing the L_2 residual norms.
    unsigned int fluid_cell_sum = 0;
    MPI_Allreduce(&region.fluid_cell_count, &fluid_cell_sum, 1, MPI_UNSIGNED, MPI_SUM, instance.cartesian_comm);

    while (simulation_runtime < max_simulation_runtime) {
        region_update_timestep_interval(&region, &instance);
        region_apply_boundary_conditions(&region);
        region_exchange(&region, MATRIX_VELOCITY_X, &instance);
        region_exchange(&region, MATRIX_VELOCITY_Y, &instance);
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

    region_destroy(&region);
    instance_destroy(&instance);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
