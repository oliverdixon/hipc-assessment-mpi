//
// Created by od641 on 10/12/2025.
//

#ifndef HIPC_ASSESSMENT_INSTANCE_H
#define HIPC_ASSESSMENT_INSTANCE_H

#include "types.h"

enum cell_flags
{
    CELL_BOUNDARY = 0, /**< Boundary cell */

    CELL_FLUID_NORTH = 1, /**< Boundary cell with fluid to the north */
    CELL_FLUID_SOUTH = 1 << 1, /**< Boundary cell with fluid to the south */
    CELL_FLUID_WEST = 1 << 2, /**< Boundary cell with fluid to the west */
    CELL_FLUID_EAST = 1 << 3, /**< Boundary cell with fluid to the east */

    CELL_FLUID_NORTHWEST = CELL_FLUID_NORTH | CELL_FLUID_WEST,
    CELL_FLUID_SOUTHWEST = CELL_FLUID_SOUTH | CELL_FLUID_WEST,
    CELL_FLUID_NORTHEAST = CELL_FLUID_NORTH | CELL_FLUID_EAST,
    CELL_FLUID_SOUTHEAST = CELL_FLUID_SOUTH | CELL_FLUID_EAST,
    CELL_FLUID_ALL = CELL_FLUID_NORTH | CELL_FLUID_SOUTH | CELL_FLUID_EAST | CELL_FLUID_WEST,

    CELL_FLUID = 1 << 4, /**< Fluid cell */
};

struct naca_specifier
{
    unsigned char maximum_camber;
    unsigned char edge_distance;
    unsigned char maximum_thickness;
};

struct data
{
    compute_t * velocity_x;
    compute_t * velocity_y;
    compute_t * tentative_velocity_x;
    compute_t * tentative_velocity_y;
    compute_t * pressure;
    compute_t * poisson_source;
    cell_flags * flags;
};

struct instance
{
    dim2 extents;
    unsigned int resolution;

    naca_specifier naca_specifier;
    compute_dim2 problem_size;
    compute_t timestep_duration;

    compute_t initial_velocity_x;
    compute_t initial_velocity_y;
    compute_t initial_pressure;
    cell_flags initial_flag;

    data device;
    data host;

    iterator * v_body_bounds;
};

instance *instance_create();

void instance_destroy(instance *instance);

__global__ void instance_set_boundaries(const instance *instance);

__global__ void instance_compute_body_indices(const instance * instance);

__global__ void instance_apply_boundary_conditions(const instance * instance);

__global__ void instance_set_neighbouring_flags(const instance * instance);

__global__ void instance_compute_tentative_velocities(const instance * instance);

__global__ void instance_compute_poisson_source(const instance * instance);

__global__ void instance_perform_sor_cycle(const instance * instance);

__global__ void instance_compute_local_residual(const instance * instance);

__global__ void instance_update_velocities(const instance * instance);

void instance_device_to_host(const instance *instance);

void instance_serialise(const instance* instance);

#endif // HIPC_ASSESSMENT_INSTANCE_H
