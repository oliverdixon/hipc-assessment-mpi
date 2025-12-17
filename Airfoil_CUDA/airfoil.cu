//
// Created by od641 on 17/11/2025.
//

#include "instance.h"

int main()
{
    instance * instance = instance_create();

    instance_compute_body_indices<<<1, instance->extents.x>>>(instance);

    static constexpr dim3 block_size = { 8, 8, 1 }; // TODO: dynamic block size with cudaOccupancyMaxPotentialBlockSize
    const dim3 grid_size = {
        static_cast<unsigned int>(std::ceil(instance->extents.x / block_size.x)),
        static_cast<unsigned int>(std::ceil(instance->extents.y / block_size.y)),
        1
    };

    instance_set_boundaries<<<grid_size, block_size>>>(instance);
    instance_set_neighbouring_flags<<<grid_size, block_size>>>(instance);

    static const compute_t max_simulation_runtime = 1.0;
    static const indexer_t sor_max_iterations = 100;
    static const compute_t sor_residual_epsilon = 0.001;
    static const indexer_t output_freq = 100;

    compute_t simulation_runtime = 0.0;
    indexer_t step_iteration = 0;

    while (simulation_runtime < max_simulation_runtime) {
        instance_apply_boundary_conditions<<<grid_size, block_size>>>(instance);
        instance_compute_tentative_velocities<<<grid_size, block_size>>>(instance);
        instance_compute_poisson_source<<<grid_size, block_size>>>(instance);

        compute_t residual = std::numeric_limits<compute_t>::max();

        for (indexer_t sor_iteration = 0; sor_iteration < sor_max_iterations; ++sor_iteration) {
            instance_perform_sor_cycle<<<grid_size, block_size>>>(instance);
            instance_compute_local_residual<<<grid_size, block_size>>>(instance);
            // TODO: reduction in shared memory for global norm of residual
            // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

            if (std::fabs(residual) < sor_residual_epsilon * sor_residual_epsilon)
                break;
        }

        instance_update_velocities<<<grid_size, block_size>>>(instance);
        simulation_runtime += instance->timestep_duration;

        break; // TODO
    }

    instance_device_to_host(instance);
    instance_serialise(instance);
    instance_destroy(instance);

    return 0;
}
