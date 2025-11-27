//
// Created by od641 on 18/11/2025.
//

#ifndef HIPC_ASSESSMENT_REGION_H
#define HIPC_ASSESSMENT_REGION_H

#include <mpi.h>
#include <stdio.h>

#include "exchanger.h"

struct instance;

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

enum region_flags
{
    REGION_UNREMARKABLE = 0,

    REGION_BOUNDARY_START_POSITION = 0,

    REGION_NORTH_BOUNDARY = 1,
    REGION_SOUTH_BOUNDARY = 1 << 1,
    REGION_WEST_BOUNDARY = 1 << 2,
    REGION_EAST_BOUNDARY = 1 << 3,

    REGION_GHOST_START_POSITION = 4,

    REGION_NORTH_GHOST = 1 << 4,
    REGION_SOUTH_GHOST = 1 << 5,
    REGION_WEST_GHOST = 1 << 6,
    REGION_EAST_GHOST = 1 << 7,
};

struct region
{
    compute_t *const *const velocity_x;
    compute_t *const *const velocity_y;
    compute_t *const *const tentative_velocity_x;
    compute_t *const *const tentative_velocity_y;
    compute_t *const *const pressure;
    compute_t *const *const poisson_source;
    enum cell_flags *const *const flags;

    const enum region_flags region_flags;
    unsigned int fluid_cell_count;

    const struct iterator h_interior;
    const struct iterator v_interior;
    const struct iterator h_exterior;
    const struct iterator v_exterior;
    const unsigned int resolution;

    const struct dim2 indents;

    const compute_t initial_velocity_x;
    const compute_t initial_velocity_y;
    const compute_t initial_pressure;
    const enum cell_flags initial_flag;

    MPI_Datatype compute_col_t;
    MPI_Datatype compute_row_t;
    MPI_Datatype flags_col_t;
    MPI_Datatype flags_row_t;

    struct exchanger velocity_x_exchanger;
    struct exchanger velocity_y_exchanger;
    struct exchanger pressure_exchanger;
    struct exchanger flags_exchanger;
};

struct region region_create(const struct instance *instance);

void region_destroy(struct region *region);

void region_describe(const struct region *region, FILE *destination);

void region_apply_boundary_conditions(const struct region *region);

void region_initialise(struct region *region, const struct instance *instance);

void region_serialise_vtk(const struct region *region, const struct instance *instance, FILE *destination);

void region_compute_halo_exchange(const struct region *region);

void step(const struct region * region);

#endif // HIPC_ASSESSMENT_REGION_H
