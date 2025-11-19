//
// Created by od641 on 18/11/2025.
//

#ifndef HIPC_ASSESSMENT_REGION_H
#define HIPC_ASSESSMENT_REGION_H

#include "instance.h"

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
    REGION_NORTH_BOUNDARY = 1,
    REGION_SOUTH_BOUNDARY = 1 << 1,
    REGION_WEST_BOUNDARY = 1 << 2,
    REGION_EAST_BOUNDARY = 1 << 3,
    REGION_NORTH_GHOST = 1 << 4,
    REGION_SOUTH_GHOST = 1 << 5,
    REGION_WEST_GHOST = 1 << 6,
    REGION_EAST_GHOST = 1 << 7
};

struct iterator
{
    unsigned int begin;
    unsigned int end;
};

struct region
{
    double * const * const velocity_x;
    double * const * const velocity_y;
    double * const * const tentative_velocity_x;
    double * const * const tentative_velocity_y;
    double * const * const pressure;
    double * const * const poisson_source;
    enum cell_flags * const * const flags;

    const enum region_flags region_flags;

    const struct iterator h_interior;
    const struct iterator v_interior;
    const struct iterator h_exterior;
    const struct iterator v_exterior;
    const unsigned int resolution;

    const unsigned int x_indent;
    const unsigned int y_indent;

    const double initial_velocity_x;
    const double initial_velocity_y;
    const double initial_pressure;
    const enum cell_flags initial_flag;
};

struct region region_create(const struct instance * instance);

void region_destroy(const struct region *region);

void region_describe(const struct region * region, FILE * destination);

void region_print(const struct region *region, FILE *destination);

void region_initialise(const struct region * region, const struct instance * instance);

#endif // HIPC_ASSESSMENT_REGION_H
