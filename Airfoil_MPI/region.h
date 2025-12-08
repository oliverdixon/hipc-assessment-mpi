//
// Created by od641 on 18/11/2025.
//

#ifndef HIPC_ASSESSMENT_REGION_H
#define HIPC_ASSESSMENT_REGION_H

#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>

struct instance;

/**
 * Bitwise flags indicating the nature of a cell in the simulation.
 */
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

/**
 * Bitwise flags indicating the characteristics of an entire region.
 */
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

/**
 * Identifiers for the data matrices in a region.
 */
enum matrix_identifier
{
    MATRIX_VELOCITY_X,
    MATRIX_VELOCITY_Y,
    MATRIX_TENTATIVE_VELOCITY_X,
    MATRIX_TENTATIVE_VELOCITY_Y,
    MATRIX_POISSON,
    MATRIX_PRESSURE,
    MATRIX_FLAGS,

    MATRIX_TYPES_COUNT = 7
};

/**
 * Metadata for performing halo exchanges on the data matrices, commonly used as a cache.
 */
struct exchange_cache
{
    bool initialised;

    const void * north_row;
    void * north_ghost;

    const void * south_row;
    void * south_ghost;

    const void * east_col;
    void * east_ghost;

    const void * west_col;
    void * west_ghost;
};

/**
 * Derived region parameters used heavily by the iterative solvers.
 */
struct cached_parameters
{
    const compute_t resolution_sq;
};

/**
 * A region describes a spatial object within a simulation containing multiple mutable vector and scalar fields, and
 * their associated metadata.
 */
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

    struct exchange_cache exchange_cache[MATRIX_TYPES_COUNT];
    struct cached_parameters derived_params;
};

/**
 * Create a new region to be managed by the given instance. The new region has initialised metadata and allocated data
 * matrices. To initialise the data to something meaningful, follow this call with @ref region_initialise.
 *
 * @param instance The constructed instance to manage the region.
 * @return The created region
 */
struct region region_create(const struct instance *instance);

/**
 * Reverse MPI state changes and heap allocations performed by @ref region_create.
 *
 * @param region The region to destroy.
 */
void region_destroy(struct region *region);

/**
 * Produce a short, human-readable summary of the given region to the given file.
 *
 * @param region The region whose metadata to describe.
 * @param destination The destination stream for the metadata summary text.
 */
void region_describe(const struct region *region, FILE *destination);

/**
 * Apply horizontal fluid flow (in from the west; out to the east), and no-slip boundary conditions on the velocity
 * matrices i.a.w. Eqns. 3.21, 3.22, and 3.33 of Griebel.
 *
 * @param region The region containing the velocity fields to be subject to the boundary conditions.
 */
void region_apply_boundary_conditions(const struct region *region);

/**
 * Initialise the given region's data matrices, trace the airfoil shape specified by the instance into boundary cells,
 * and set the rectangular border.
 *
 * @param region The region to initialise.
 * @param instance The instance by which the region is managed.
 */
void region_initialise(struct region *region, const struct instance *instance);

void region_serialise_vtk(const struct region *region, const struct instance *instance, FILE *destination);

/**
 * Compute the tentative velocities based on self- and cross-advection, and diffusion, from previous velocity values.
 * These are computed i.a.w. Eqns. 3.36 and 3.37 of Griebel.
 *
 * @param region The region containing the previous velocities and current tentative velocities.
 * @param instance The instance managing the region.
 */
void region_compute_tentative_velocities(const struct region *region, const struct instance *instance);

/**
 * Compute the Poisson source/forcing term for the pressure computation i.a.w. Eqn. 3.38 of Griebel. This spans across
 * the entire interior.
 *
 * @param region The region containing the tentative velocities and Poisson targets.
 * @param instance The instance managing the region.
 */
void region_compute_poisson_source(const struct region *region, const struct instance *instance);

/**
 * Perform a single cycle of Successive Overrelaxation i.a.w. Eqn. 3.44 of Griebel to produce revised a revised pressure
 * scalar field.
 *
 * @param region The region containing the pressure matrix on which SOR should be performed.
 * @param instance The instance managing the region.
 */
void region_sor_cycle(const struct region *region, const struct instance *instance);

/**
 * Perform a halo exchange on region data for its neighbours on the 2D cartesian grid: north, south, east, and west.
 *
 * @param region The region containing the data to exchange.
 * @param matrix The identifier of the data segment to exchange.
 * @param instance The instance managing the target region.
 * @details For the selected data segment, assumed to be a 2D matrix, the most extreme rows or columns on the dimension
 *  will be exchanged with the local ghost arrays. For example, the southernmost row of the northern neighbour's matrix
 *  will be loaded into the local north ghost row. If there is no neighbour in a particular direction, nothing is done.
 *  This function allows the exchanges to occur in any order, but will block the thread until they are all complete.
 */
void region_exchange(struct region *region, enum matrix_identifier matrix, const struct instance *instance);

/**
 * Compute the discrete residual of the Poisson equation i.a.w. Eqns. 3.45 and 3.46 of Griebel.
 *
 * @param region The region containing the populated Poisson scalar field.
 * @return The partial Poisson residual. To compute the L^2 norm, take the square root.
 */
compute_t region_compute_poisson_residual(const struct region * region);

/**
 * Update the X and Y velocities by the tentative velocities and pressures, according to Eqns. 3.34 and 3.35 of Griebel.
 *
 * @param region The region containing the velocities.
 * @param instance The instance managing the target region.
 */
void region_update_velocities(const struct region *region, const struct instance *instance);

#endif // HIPC_ASSESSMENT_REGION_H
