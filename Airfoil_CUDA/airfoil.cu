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

    safe_cuda(cudaDeviceSynchronize()); // Wait for airfoil body indices to be computed for each column.
    instance_set_extreme_boundaries<<<grid_size, block_size>>>(instance);

    safe_cuda(cudaDeviceSynchronize()); // Wait for finalised data beforing transferring back to the host.
    instance_device_to_host(instance);

    const cell_flags * const data = instance->host.flags;
    for (indexer_t v_idx = 0; v_idx < instance->extents.y; ++v_idx) {
        const std::size_t v_basis = instance->extents.x * v_idx;
        for (indexer_t h_idx = 0; h_idx < instance->extents.x; ++h_idx)
            printf("%02d ", data[v_basis + h_idx]);
        putchar('\n');
    }

    instance_destroy(instance);

    return 0;
}
