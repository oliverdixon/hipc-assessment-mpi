//
// Created by od641 on 10/12/2025.
//

#include "instance.h"

instance *instance_create()
{
    instance * instance;
    cudaMallocManaged(&instance, sizeof(struct instance));
    
    instance->resolution = 128;
    instance->problem_size.x = 4.0;
    instance->problem_size.y = 1.0;
    instance->initial_velocity_x = 1.0;
    instance->initial_velocity_y = 0.0;
    instance->initial_pressure = 0.0;
    instance->initial_flag = CELL_FLUID;
    instance->naca_specifier.maximum_camber = 2;
    instance->naca_specifier.edge_distance = 4;
    instance->naca_specifier.maximum_thickness = 12;

    data * const device = &instance->device;

    instance->extents.x = (indexer_t) ceil((compute_t) instance->resolution * instance->problem_size.x);
    instance->extents.y = (indexer_t) ceil((compute_t) instance->resolution * instance->problem_size.y);

    const std::size_t compute_byte_count = sizeof(compute_t) * instance->extents.x * instance->extents.y;

    safe_cuda(cudaMalloc(&device->velocity_x, compute_byte_count));
    safe_cuda(cudaMalloc(&device->velocity_y, compute_byte_count));
    safe_cuda(cudaMalloc(&device->tentative_velocity_x, compute_byte_count));
    safe_cuda(cudaMalloc(&device->tentative_velocity_y, compute_byte_count));
    safe_cuda(cudaMalloc(&device->pressure, compute_byte_count));
    safe_cuda(cudaMalloc(&device->poisson_source, compute_byte_count));
    safe_cuda(cudaMalloc(&device->flags, sizeof(cell_flags) * instance->extents.x * instance->extents.y));
    safe_cuda(cudaMalloc(&instance->v_body_bounds, sizeof(iterator) * instance->extents.x));

    data * const host = &instance->host;

    host->velocity_x = new compute_t[instance->extents.x * instance->extents.y];
    host->velocity_y = new compute_t[instance->extents.x * instance->extents.y];
    host->tentative_velocity_x = new compute_t[instance->extents.x * instance->extents.y];
    host->tentative_velocity_y = new compute_t[instance->extents.x * instance->extents.y];
    host->pressure = new compute_t[instance->extents.x * instance->extents.y];
    host->poisson_source = new compute_t[instance->extents.x * instance->extents.y];
    host->flags = new cell_flags[instance->extents.x * instance->extents.y];

    return instance;
}

void instance_destroy(instance * const instance)
{
    const data * const device = &instance->device;

    safe_cuda(cudaFree(instance->v_body_bounds));
    safe_cuda(cudaFree(device->flags));
    safe_cuda(cudaFree(device->poisson_source));
    safe_cuda(cudaFree(device->pressure));
    safe_cuda(cudaFree(device->tentative_velocity_y));
    safe_cuda(cudaFree(device->tentative_velocity_x));
    safe_cuda(cudaFree(device->velocity_y));
    safe_cuda(cudaFree(device->velocity_x));

    const data * const host = &instance->host;

    delete[] host->flags;
    delete[] host->poisson_source;
    delete[] host->pressure;
    delete[] host->tentative_velocity_y;
    delete[] host->tentative_velocity_x;
    delete[] host->velocity_y;
    delete[] host->velocity_x;

    cudaFree(instance);
}

__global__ void instance_compute_body_indices(const instance *const instance)
{
    iterator bounds = {
        .begin = 0,
        .end = 0
    };

    const std::size_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_idx >= instance->extents.x)
        return;

    const compute_t x = static_cast<compute_t>(x_idx) / instance->resolution - 0.5;

    if (x >= 0.0 || x <= 1.0) {
        const compute_t maximum_camber = instance->naca_specifier.maximum_camber / 100.0;
        const compute_t edge_distance = instance->naca_specifier.edge_distance / 10.0;
        const compute_t thickness = instance->naca_specifier.maximum_thickness / 100.0;

        const compute_t x_sq = x * x;

        const compute_t mean_camber_line_y = x <= edge_distance
                ? maximum_camber / (edge_distance * edge_distance) * (2.0 * edge_distance * x - x_sq)
                : // 0 <= x <= p
                maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * // p < x <= 1
                        (1.0 - 2.0 * edge_distance + 2.0 * edge_distance * x - x_sq);

        const compute_t norm = x <= edge_distance
                ? 2.0 * maximum_camber / (edge_distance * edge_distance) * (edge_distance - x)
                : 2.0 * maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * (edge_distance - x);

        const compute_t midline_distance = 5.0 * thickness * cos(atan(norm)) *
            (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x_sq + 0.2843 * x * x_sq - 0.1015 * x_sq * x_sq);

        bounds.begin = floor((mean_camber_line_y - midline_distance + instance->problem_size.y / 2.0) *
            instance->resolution);

        bounds.end = ceil((mean_camber_line_y + midline_distance + instance->problem_size.y / 2.0) *
            instance->resolution);
    }

    instance->v_body_bounds[x_idx] = bounds;
}

__global__ void instance_set_extreme_boundaries(const instance *const instance)
{
    const dim2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x < instance->extents.x && idx.y < instance->extents.y) {
        const data * const data = &instance->device;
        const indexer_t array_idx = idx.x + instance->extents.x * idx.y;

        data->velocity_x[array_idx] = instance->initial_velocity_x;
        data->velocity_y[array_idx] = instance->initial_velocity_y;
        data->pressure[array_idx] = instance->initial_pressure;

        const iterator body_bounds = instance->v_body_bounds[idx.x];

        data->flags[array_idx] =
            idx.x == 0 || idx.x == instance->extents.x - 1 || idx.y == 0 || idx.y == instance->extents.y - 1 ||
                (idx.y >= body_bounds.begin && idx.y < body_bounds.end) ? CELL_BOUNDARY : CELL_FLUID;
    }
}

void instance_device_to_host(const instance *instance)
{
    const data * src = &instance->device;
    const data * dst = &instance->host;
    const std::size_t compute_byte_count = instance->extents.x * instance->extents.y * sizeof(compute_t);
    const std::size_t flags_byte_count = instance->extents.x * instance->extents.y * sizeof(cell_flags);

    static constexpr cudaMemcpyKind mode = cudaMemcpyDeviceToHost;

    safe_cuda(cudaMemcpy(dst->velocity_x, src->velocity_x, compute_byte_count, mode));
    safe_cuda(cudaMemcpy(dst->velocity_y, src->velocity_y, compute_byte_count, mode));
    safe_cuda(cudaMemcpy(dst->tentative_velocity_x, src->tentative_velocity_x, compute_byte_count, mode));
    safe_cuda(cudaMemcpy(dst->tentative_velocity_y, src->tentative_velocity_y, compute_byte_count, mode));
    safe_cuda(cudaMemcpy(dst->pressure, src->pressure, compute_byte_count, mode));
    safe_cuda(cudaMemcpy(dst->poisson_source, src->poisson_source, compute_byte_count, mode));
    safe_cuda(cudaMemcpy(dst->flags, src->flags, flags_byte_count, mode));
}
