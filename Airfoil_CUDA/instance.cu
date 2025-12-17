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
    instance->timestep_duration = 0.003;

    data * const device = &instance->device;

    instance->extents.x = (indexer_t) ceil((compute_t) instance->resolution * instance->problem_size.x);
    instance->extents.y = (indexer_t) ceil((compute_t) instance->resolution * instance->problem_size.y);

    const std::size_t allocation_extent = (instance->extents.x + 1) * (instance->extents.y + 1);
    const std::size_t allocation_extent_bytes = sizeof(compute_t) * allocation_extent;

    safe_cuda(cudaMalloc(&device->velocity_x, allocation_extent_bytes));
    safe_cuda(cudaMalloc(&device->velocity_y, allocation_extent_bytes));
    safe_cuda(cudaMalloc(&device->tentative_velocity_x, allocation_extent_bytes));
    safe_cuda(cudaMalloc(&device->tentative_velocity_y, allocation_extent_bytes));
    safe_cuda(cudaMalloc(&device->pressure, allocation_extent_bytes));
    safe_cuda(cudaMalloc(&device->poisson_source, allocation_extent_bytes));
    safe_cuda(cudaMalloc(&device->flags, sizeof(cell_flags) * allocation_extent));
    safe_cuda(cudaMalloc(&instance->v_body_bounds, sizeof(iterator) * instance->extents.x));

    data * const host = &instance->host;

    host->velocity_x = new compute_t[allocation_extent];
    host->velocity_y = new compute_t[allocation_extent];
    host->tentative_velocity_x = new compute_t[allocation_extent];
    host->tentative_velocity_y = new compute_t[allocation_extent];
    host->pressure = new compute_t[allocation_extent];
    host->poisson_source = new compute_t[allocation_extent];
    host->flags = new cell_flags[allocation_extent];

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

__global__ void instance_apply_boundary_conditions(const instance *const instance)
{
    const dim2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x >= instance->extents.x || idx.y >= instance->extents.y)
        return;

    const indexer_t v_basis = instance->extents.x * idx.y;
    const data * const data = &instance->device;

    if (idx.x == 0) {
        // Fluid freely flows in from the west
        data->velocity_x[v_basis] = instance->device.velocity_x[v_basis + 1];
        data->velocity_y[v_basis] = instance->device.velocity_y[v_basis + 1];
    } else if (idx.x == instance->extents.x - 1) {
        // Fluid freely flows out to the east
        data->velocity_x[v_basis + idx.x - 1] = data->velocity_x[v_basis + idx.x - 2];
        data->velocity_y[v_basis + idx.x] = data->velocity_x[v_basis + idx.x - 1];
    }
    
    /*
     * At the north and south boundaries, the vertical velocity approaches zero and fluid flows freely on the
     * horizontal.
     */
    if (idx.y == 0) {
        const indexer_t north_idx_basis = idx.x;
        data->velocity_x[north_idx_basis] = data->velocity_x[north_idx_basis + instance->extents.x];
        data->velocity_y[north_idx_basis] = 0.0;
    } else if (idx.y == instance->extents.y - 1) {
        const indexer_t south_idx_basis = idx.x + instance->extents.x * (instance->extents.y - 1);
        data->velocity_x[south_idx_basis] = data->velocity_x[south_idx_basis - instance->extents.x];
        data->velocity_y[south_idx_basis - instance->extents.x] = 0.0;
    }
    
    if (data->flags[v_basis + idx.x] & CELL_FLUID_ALL) {
        const indexer_t idx_central = idx.x + instance->extents.x * idx.y;

        const indexer_t idx_north = idx.x + instance->extents.x * (idx.y - 1);
        const indexer_t idx_south = idx.x + instance->extents.x * (idx.y + 1);
        const indexer_t idx_west = idx.x - 1 + instance->extents.x * idx.y;
        const indexer_t idx_east = idx.x + 1 + instance->extents.x * idx.y;

        const indexer_t idx_northeast = idx.x + 1 + instance->extents.x * (idx.y - 1);
        const indexer_t idx_southwest = idx.x - 1 + instance->extents.x * (idx.y + 1);
        const indexer_t idx_northwest = idx.x - 1 + instance->extents.x * (idx.y - 1);

        switch (data->flags[v_basis + idx.x]) {
        case CELL_FLUID_NORTH:
            data->velocity_y[idx_central] = 0.0;
            data->velocity_x[idx_central] = -data->velocity_x[idx_south];
            data->velocity_x[idx_west] = -data->velocity_x[idx_southwest];
            break;
        case CELL_FLUID_EAST:
            data->velocity_x[idx_central] = 0.0;
            data->velocity_y[idx_central] = -data->velocity_y[idx_east];
            data->velocity_y[idx_north] = -data->velocity_y[idx_northeast];
            break;
        case CELL_FLUID_SOUTH:
            data->velocity_y[idx_north] = 0.0;
            data->velocity_x[idx_central] = -data->velocity_x[idx_north];
            data->velocity_x[idx_west] = -data->velocity_x[idx_northwest];
            break;
        case CELL_FLUID_WEST:
            data->velocity_x[idx_west] = 0.0;
            data->velocity_y[idx_central] = -data->velocity_y[idx_west];
            data->velocity_y[idx_north] = -data->velocity_y[idx_northwest];
            break;
        case CELL_FLUID_NORTHEAST:
            data->velocity_y[idx_central] = 0.0;
            data->velocity_x[idx_central] = 0.0;
            data->velocity_y[idx_north] = -data->velocity_y[idx_northeast];
            data->velocity_x[idx_west] = -data->velocity_x[idx_southwest];
            break;
        case CELL_FLUID_SOUTHEAST:
            data->velocity_y[idx_north] = 0.0;
            data->velocity_x[idx_central] = 0.0;
            data->velocity_y[idx_central] = -data->velocity_y[idx_east];
            data->velocity_x[idx_west] = -data->velocity_x[idx_northwest];
            break;
        case CELL_FLUID_SOUTHWEST:
            data->velocity_y[idx_north] = 0.0;
            data->velocity_x[idx_west] = 0.0;
            data->velocity_y[idx_central] = -data->velocity_y[idx_west];
            data->velocity_x[idx_central] = -data->velocity_x[idx_north];
            break;
        case CELL_FLUID_NORTHWEST:
            data->velocity_y[idx_central] = 0.0;
            data->velocity_x[idx_west] = 0.0;
            data->velocity_y[idx_north] = -data->velocity_y[idx_northwest];
            data->velocity_x[idx_central] = -data->velocity_x[idx_south];
            break;
        default:;
        }
    }

    if (idx.x == 0) {
        /*
         * If we're on a western boundary, fix the western-edge velocities such that there is a continual flow of fluid
         * into the simulation space.
         */

        const indexer_t west_anchored_idx = instance->extents.x * idx.y;
        data->velocity_x[west_anchored_idx] = instance->initial_velocity_x;
        data->velocity_y[west_anchored_idx] = 2 * instance->initial_velocity_y -
            data->velocity_y[west_anchored_idx + 1];
    }
}

__global__ void instance_set_neighbouring_flags(const instance *const instance)
{
    const dim2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x >= instance->extents.x || idx.y >= instance->extents.y)
        return;

    const indexer_t v_basis = instance->extents.x * idx.y;
    const indexer_t flat_idx = idx.x + v_basis;
    cell_flags * const flags = instance->device.flags;

    if (!(flags[flat_idx] & CELL_FLUID)) {
        if (idx.x > 0 && flags[flat_idx - 1] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_WEST);
        if (idx.x < instance->extents.x - 1 && flags[flat_idx + 1] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_EAST);
        if (idx.y > 0 && flags[flat_idx - instance->extents.x] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_SOUTH);
        if (idx.y < instance->extents.y - 1 && flags[flat_idx + instance->extents.x] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_NORTH);
    }
}

__global__ void instance_compute_tentative_velocities(const instance *instance)
{
    const dim2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y
    };

    // TODO maybe this condition is too restrictive.
    if (idx.x < 1 || idx.x >= instance->extents.x - 1 || idx.y < 1 || idx.y >= instance->extents.y - 1)
        return;
    
    static constexpr compute_t reynolds = 500.0;
    static constexpr double gamma = 0.9; // Upwind differencing factor in PDE discretisation
    
    const data * const data = &instance->device;
    const compute_t * const velocity_x = data->velocity_x;
    const compute_t * const velocity_y = data->velocity_y;
    const cell_flags * const flags = data->flags;
    
    const indexer_t idx_central = idx.x + instance->extents.x * idx.y;

    const indexer_t idx_north = idx.x + instance->extents.x * (idx.y - 1);
    const indexer_t idx_south = idx.x + instance->extents.x * (idx.y + 1);
    const indexer_t idx_west = idx.x - 1 + instance->extents.x * idx.y;
    const indexer_t idx_east = idx.x + 1 + instance->extents.x * idx.y;

    const indexer_t idx_northeast = idx.x + 1 + instance->extents.x * (idx.y - 1);
    const indexer_t idx_southwest = idx.x - 1 + instance->extents.x * (idx.y + 1);

    const compute_t quarter_resolution = instance->resolution / 4.0;
    const compute_t sq_resolution = instance->resolution * instance->resolution;

    // TODO: check this. Why only checking east when else comment indicates "adjacent cells"?
    if (flags[idx_central] & CELL_FLUID && flags[idx_east] & CELL_FLUID) {
        const double self_advection_x =
            (
                (velocity_x[idx_central] + velocity_x[idx_east]) *
                (velocity_x[idx_central] + velocity_x[idx_east]) +
                gamma * fabs(velocity_x[idx_central] + velocity_x[idx_east]) *
                (velocity_x[idx_central] - velocity_x[idx_east]) -
                (velocity_x[idx_west] + velocity_x[idx_central]) *
                (velocity_x[idx_west] + velocity_x[idx_central]) -
                gamma * fabs(velocity_x[idx_west] + velocity_x[idx_central]) *
                (velocity_x[idx_west] - velocity_x[idx_central])
            ) * quarter_resolution;

        const double cross_advection_y =
            (
                (velocity_y[idx_central] + velocity_y[idx_east]) *
                (velocity_x[idx_central] + velocity_x[idx_south]) +
                gamma * fabs(velocity_y[idx_central] + velocity_y[idx_east]) *
                (velocity_x[idx_central] - velocity_x[idx_south]) -
                (velocity_y[idx_north] + velocity_y[idx_northeast]) *
                (velocity_x[idx_north] + velocity_x[idx_central]) -
                gamma * fabs(velocity_y[idx_north] + velocity_y[idx_northeast]) *
                (velocity_x[idx_north] - velocity_x[idx_central])
            ) * quarter_resolution;

        const double diffusion =
            (
                velocity_x[idx_east] -
                2.0 * velocity_x[idx_central] +
                velocity_x[idx_west] +
                velocity_x[idx_south] -
                2.0 * velocity_x[idx_central] +
                velocity_x[idx_north]
            ) * sq_resolution;

        data->tentative_velocity_x[idx_central] = velocity_x[idx_central] + instance->timestep_duration *
            (diffusion / reynolds - self_advection_x - cross_advection_y);
    } else
        // If both adjacent cells are not fluids, the velocity is unchanged.
        data->tentative_velocity_x[idx_central] = velocity_x[idx_central];

    if (flags[idx_central] & CELL_FLUID && flags[idx_south] & CELL_FLUID) {
        const double cross_advection_x =
            (
                (velocity_x[idx_central] + velocity_x[idx_south]) *
                (velocity_y[idx_central] + velocity_y[idx_east]) +
                gamma * fabs(velocity_x[idx_central] + velocity_x[idx_south]) *
                (velocity_y[idx_central] - velocity_y[idx_east]) -
                (velocity_x[idx_west] + velocity_x[idx_southwest]) *
                (velocity_y[idx_west] + velocity_y[idx_central]) -
                gamma * fabs(velocity_x[idx_west] + velocity_x[idx_southwest]) *
                (velocity_y[idx_west] - velocity_y[idx_central])
            ) * quarter_resolution;

        const double self_advection_y =
            (
                (velocity_y[idx_central] + velocity_y[idx_south]) *
                (velocity_y[idx_central] + velocity_y[idx_south]) +
                gamma * fabs(velocity_y[idx_central] + velocity_y[idx_south]) *
                (velocity_y[idx_central] - velocity_y[idx_south]) -
                (velocity_y[idx_north] + velocity_y[idx_central]) *
                (velocity_y[idx_north] + velocity_y[idx_central]) -
                gamma * fabs(velocity_y[idx_north] + velocity_y[idx_central]) *
                (velocity_y[idx_north] - velocity_y[idx_central])
            ) * quarter_resolution;

        const double diffusion =
            (
                velocity_y[idx_east] -
                2.0 * velocity_y[idx_central] +
                velocity_y[idx_west] +
                velocity_y[idx_south] -
                2.0 * velocity_y[idx_central] +
                velocity_y[idx_north]
            ) * sq_resolution;

        data->tentative_velocity_y[idx_central] = velocity_y[idx_central] + instance->timestep_duration *
            (diffusion / reynolds - cross_advection_x - self_advection_y);

    } else
        // If both adjacent cells are not fluids, the velocity is unchanged.
        data->tentative_velocity_y[idx_central] = velocity_y[idx_central];
}

__global__ void instance_compute_poisson_source(const instance *const instance)
{
    const dim2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x > instance->extents.x - 1 || idx.y > instance->extents.y - 1)
        return;

    compute_t * const velocity_x = instance->device.velocity_x;
    compute_t * const velocity_y = instance->device.velocity_y;
    compute_t * const tentative_velocity_x = instance->device.tentative_velocity_x;
    compute_t * const tentative_velocity_y = instance->device.tentative_velocity_y;
    compute_t * const pressure = instance->device.pressure;

    if (idx.x == 0) {
        const indexer_t west_anchored_idx = instance->extents.x * idx.y;
        tentative_velocity_x[west_anchored_idx] = velocity_x[west_anchored_idx];
        pressure[west_anchored_idx] = pressure[west_anchored_idx + 1];
    }

    else if (idx.x == instance->extents.x - 1) {
        const indexer_t east_anchored_idx = instance->extents.x * idx.y + instance->extents.x;
        tentative_velocity_x[east_anchored_idx] = velocity_x[east_anchored_idx];
        pressure[east_anchored_idx] = pressure[east_anchored_idx - 1];
    }

    if (idx.y == 0) {
        const indexer_t north_anchored_idx = idx.x;
        tentative_velocity_y[north_anchored_idx] = velocity_y[north_anchored_idx];
        pressure[north_anchored_idx] = pressure[north_anchored_idx + instance->extents.x];
    }

    else if (idx.y == instance->extents.y - 1) {
        const indexer_t south_anchored_idx = instance->extents.x * idx.y;
        tentative_velocity_y[south_anchored_idx] = velocity_y[south_anchored_idx];
        pressure[south_anchored_idx] = pressure[south_anchored_idx - instance->extents.x];
    }
}

__global__ void instance_perform_sor_cycle(const instance *const instance)
{
    const dim2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x > instance->extents.x - 1 || idx.y > instance->extents.y - 1)
        return;

    const data * const data = &instance->device;
    const indexer_t idx_central = idx.x + instance->extents.x * idx.y;

    static constexpr compute_t omega = 1.7; // TODO move

    const compute_t r_step_sq = 1.0 / (instance->resolution * instance->resolution);
    compute_t weight;
    compute_t x_spatial;
    compute_t y_spatial;

    if (data->flags[idx_central] & CELL_FLUID_ALL) {

        weight = omega / (4 * r_step_sq);
        x_spatial = (data->pressure[idx_central + 1] + data->pressure[idx_central - 1]) * r_step_sq;
        y_spatial = (data->pressure[idx_central + instance->extents.x] +
            data->pressure[idx_central - instance->extents.x]) * r_step_sq;

    } else if (data->flags[idx_central] & CELL_FLUID) {

        const compute_t epsilon_east = !!(data->flags[idx_central + 1] & CELL_FLUID);
        const compute_t epsilon_west = !!(data->flags[idx_central - 1] & CELL_FLUID);
        const compute_t epsilon_north = !!(data->flags[idx_central - instance->extents.x] & CELL_FLUID);
        const compute_t epsilon_south = !!(data->flags[idx_central + instance->extents.x] & CELL_FLUID);

        weight = omega / ((epsilon_east + epsilon_west) * r_step_sq + (epsilon_north + epsilon_south) *
            r_step_sq);

        x_spatial = (
            data->pressure[idx_central + 1] * epsilon_west +
            data->pressure[idx_central - 1] * epsilon_east) * r_step_sq;

        y_spatial = (
            data->pressure[idx_central + instance->extents.x] * epsilon_south +
            data->pressure[idx_central - instance->extents.x] * epsilon_north) * r_step_sq;
    } else
        // Nothing to do for non-fluid cells.
        return;

    data->pressure[idx_central] = (1.0 - omega) * data->pressure[idx_central] + weight *
        (x_spatial + y_spatial - data->poisson_source[idx_central]);
}

__global__ void instance_compute_local_residual(const instance *instance)
{
    const dim2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x > instance->extents.x - 1 || idx.y > instance->extents.y - 1)
        return;

    // TODO...
}

__global__ void instance_update_velocities(const instance *instance)
{
    const dim2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x > instance->extents.x - 1 || idx.y > instance->extents.y - 1)
        return;

    const compute_t x_pressure_diff_factor = instance->timestep_duration * instance->resolution;
    const compute_t y_pressure_diff_factor = instance->timestep_duration * instance->resolution;

    const data * const data = &instance->device;
    const indexer_t idx_central = idx.x + instance->extents.x * idx.y;

    if (data->flags[idx_central] & CELL_FLUID && data->flags[idx_central + 1] & CELL_FLUID) {

        // TODO: might need to check the bounds here?

        data->velocity_x[idx_central] = data->tentative_velocity_x[idx_central] -
            (data->pressure[idx_central + 1] - data->pressure[idx_central]) * x_pressure_diff_factor;
        data->velocity_y[idx_central] = data->tentative_velocity_y[idx_central] -
            (data->pressure[idx_central + instance->extents.x] - data->pressure[idx_central]) * y_pressure_diff_factor;
    }
}

__global__ void instance_set_boundaries(const instance *const instance)
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

void instance_serialise(const instance *instance)
{
    FILE * const destination = fopen("out/flows.vtr", "w");

    const indexer_t h_pixel_count = static_cast<unsigned int>(instance->problem_size.x) * instance->resolution;
    const indexer_t v_pixel_count = static_cast<unsigned int>(instance->problem_size.y) * instance->resolution;

    assert(h_pixel_count == instance->extents.x);
    assert(v_pixel_count == instance->extents.y);

    fprintf(destination,
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "\t<RectilinearGrid WholeExtent=\"0 %u 0 %u 0 0\" GhostLevel=\"0\">\n"
        "\t\t<Piece Extent=\"0 %u 0 %u 0 0\">\n"
        "\t\t\t<Coordinates>\n"
        "\t\t\t\t<DataArray type=\"Float64\" name=\"X\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"%lf\">\n",

        h_pixel_count, v_pixel_count, h_pixel_count, v_pixel_count, instance->problem_size.x);

    // Write out physical positions of X co-ordinates.
    for (indexer_t h_idx = 0; h_idx <= h_pixel_count; ++h_idx)
        fprintf(destination, "%lf ", static_cast<compute_t>(h_idx) / instance->resolution);

    fprintf(destination,
        "\n\t\t\t\t</DataArray>\n"
        "\t\t\t\t<DataArray type=\"Float64\" name=\"Y\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"%lf\">\n",
        instance->problem_size.y);

    // Write out physical positions of Y co-ordinates.
    for (indexer_t v_idx = 0; v_idx <= v_pixel_count; ++v_idx)
        fprintf(destination, "%lf ", static_cast<compute_t>(v_idx) / instance->resolution);

    // Write out velocity vectors.
    fprintf(destination,
        "\n\t\t\t\t</DataArray>\n"
        "\t\t\t\t<DataArray type=\"Float64\" name=\"Y\" format=\"ascii\">\n"
        "0.0\n"
        "\t\t\t\t</DataArray>\n"
        "\t\t\t</Coordinates>\n"
        "\t\t\t<PointData Vectors=\"uv\">\n"
        "\t\t\t\t<DataArray type=\"Float64\" Name=\"uv\" NumberOfComponents=\"3\" format=\"ascii\">\n");

    for (indexer_t v_idx = 0; v_idx <= v_pixel_count; ++v_idx) {
        const indexer_t v_basis = v_idx * instance->extents.x;
        for (indexer_t h_idx = 0; h_idx <= h_pixel_count; ++h_idx)
            fprintf(destination, "%lf %lf 0\n",
                instance->host.velocity_x[v_basis + h_idx],
                instance->host.velocity_y[v_basis + h_idx]);
    }

    // Write out pressure scalars.
    fputs(
        "\t\t\t\t</DataArray>\n"
        "\t\t\t</PointData>\n"
        "\t\t\t<CellData Scalars=\"p\">\n"
        "\t\t\t\t<DataArray type=\"Float64\" format=\"ascii\" Name=\"p\">\n",
        destination);

    for (indexer_t v_idx = 0; v_idx < v_pixel_count; ++v_idx) {
        const indexer_t v_basis = v_idx * instance->extents.x;
        for (indexer_t h_idx = 0; h_idx < h_pixel_count; ++h_idx)
            fprintf(destination, "%lf ", instance->host.pressure[v_basis + h_idx]);
        fputc('\n', destination);
    }

    fputs(
        "\t\t\t\t</DataArray>\n"
        "\t\t\t</CellData>\n"
        "\t\t</Piece>\n"
        "\t</RectilinearGrid>\n"
        "</VTKFile>\n",
        destination);

    fclose(destination);
}
