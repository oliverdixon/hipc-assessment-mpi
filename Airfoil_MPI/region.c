//
// Created by od641 on 18/11/2025.
//

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <strings.h>
#include <omp.h>

#include "instance.h"
#include "region.h"

enum exchanger_tags
{
    TAGS_NORTH,
    TAGS_SOUTH,
    TAGS_EAST,
    TAGS_WEST,
    TAGS_SELF,
};

enum sor_phase
{
    SOR_RED = 0,
    SOR_BLACK = 1
};

static MPI_Datatype create_row_t(
        const MPI_Aint column_count,
        const MPI_Aint row_count,
        const MPI_Datatype element_type, // NOLINT(*-misplaced-const)
        const MPI_Aint element_size)
{
    assert(column_count <= INT_MAX);
    assert(row_count <= INT_MAX);

    MPI_Datatype raw_row_t;
    MPI_Type_vector((int) column_count, 1, (int) row_count, element_type, &raw_row_t);
    MPI_Type_commit(&raw_row_t);

    MPI_Datatype row_t;
    MPI_Type_create_resized(raw_row_t, 0, element_size, &row_t);
    MPI_Type_commit(&row_t);
    MPI_Type_free(&raw_row_t);

    return row_t;
}

static MPI_Datatype create_column_t(
    const indexer_t row_count,
    const MPI_Datatype element_type) // NOLINT(*-misplaced-const)
{
    assert(row_count <= INT_MAX);

    MPI_Datatype column_t;
    MPI_Type_contiguous((int) row_count, element_type, &column_t);
    MPI_Type_commit(&column_t);

    return column_t;
}

static compute_t **alloc_2d_compute_array(const struct dim2 size)
{
    compute_t **array = malloc(size.x * sizeof(compute_t *));
    array[0] = calloc(size.x * size.y, sizeof(compute_t));

    for (size_t column_idx = 1; column_idx < size.x; ++column_idx)
        array[column_idx] = &array[0][column_idx * size.y];

    return array;
}

static enum cell_flags **alloc_2d_flags_array(const struct dim2 size)
{
    enum cell_flags **array = malloc(size.x * sizeof(enum cell_flags *));
    array[0] = calloc(size.x * size.y, sizeof(enum cell_flags));

    for (size_t column_idx = 1; column_idx < size.x; ++column_idx)
        array[column_idx] = &array[0][column_idx * size.y];

    return array;
}

static void free_2d_array(void **array)
{
    free(array[0]);
    free(array);
}

static struct iterator get_initial_v_idx_boundaries(
        const struct region *const region, const compute_t problem_height, const float maximum_camber,
        const float edge_distance, const float thickness, const indexer_t h_idx)
{
    struct iterator boundaries = {
        .begin = 0,
        .end = 0
    };

    /*
     * Position along chord, normalised to [0, 1]. From here, 'x' is translated into the co-ordinate space of the
     * global problem, and not the region.
     */
    const compute_t x = (compute_t) (h_idx + region->indents.x) / (compute_t) region->resolution - 0.5f;

    if (x < 0.0 || x > 1.0)
        return boundaries;

    /*
     * The midline distance is the half-thickness from the fixed 'x' to the horizontal central line of the airfoil.
     * It is the Euclidean distance from the 'x' co-ordinate to the midline. This is NACA standard formulae.
     */
    const compute_t x_sq = x * x;
    compute_t midline_distance = 5.0 * thickness *
            (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x_sq + 0.2843 * x * x_sq - 0.1015 * x_sq * x_sq);

    /*
     * Compute the 'y' co-ordinate of the mean camber line, given the fixed 'x' position. This is NACA standard
     * formulae, represented as a piecewise map over 'x' in intervals [0, p] and (p, 1], where 'p' is the edge
     * distance.
     */
    const compute_t mean_camber_line_y = x <= edge_distance
            ? maximum_camber / (edge_distance * edge_distance) * (2.0 * edge_distance * x - x_sq)
            : // 0 <= x <= p
            maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * // p < x <= 1
                    (1.0 - 2.0 * edge_distance + 2.0 * edge_distance * x - x_sq);

    /*
     * Use standard calculus formulae to find the numerical derivative of the mean camber line 'y' co-ordinate.
     * Thickness is applied perpendicular to the mean camber line. Use standard geometric formulae to compute the 'y'
     * co-ordinates for the upper and lower camber surface lines.
     */
    const compute_t norm = x <= edge_distance
            ? 2.0 * maximum_camber / (edge_distance * edge_distance) * (edge_distance - x)
            : 2.0 * maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * (edge_distance - x);

    midline_distance *= cos(atan(norm));

    const compute_t upper_camber_y = mean_camber_line_y + midline_distance;
    const compute_t lower_camber_y = mean_camber_line_y - midline_distance;

    indexer_t v_idx_boundary_start = floor((lower_camber_y + problem_height / 2.0) * region->resolution);
    indexer_t v_idx_boundary_end = ceil((upper_camber_y + problem_height / 2.0) * region->resolution);

    const indexer_t south_boundary_idx = region->indents.y + (region->v_exterior.end - region->v_exterior.begin);

    if (v_idx_boundary_start < region->indents.y)
        // If the start is a pixel in a region to the north of us, bound it at our northernmost point.
        v_idx_boundary_start = region->v_exterior.begin;
    else if (v_idx_boundary_start > south_boundary_idx)
        // If the start is a pixel in a region to the south of us, this region doesn't contain any of the points.
        return boundaries;
    else
        // Otherwise, the start-point is somewhere within our region. Translate the index to zero-based from the north.
        v_idx_boundary_start -= region->indents.y;

    if (v_idx_boundary_end >= south_boundary_idx)
        // If the end is a pixel in a region to the south of us, bound it at our southernmost point.
        v_idx_boundary_end = region->v_exterior.end;
    else if (v_idx_boundary_end < region->indents.y)
        // If the end is a pixel in a region to the north of us, this region doesn't contain any of the points.
        return boundaries;
    else
        // Otherwise, the end-point is somewhere within our region. Translate the index to zero-based from the north.
        v_idx_boundary_end -= region->indents.y;

    boundaries.begin = v_idx_boundary_start;
    boundaries.end = v_idx_boundary_end;
    return boundaries;
}

static void write_initial_extreme_boundaries(const struct region *const region)
{
    enum cell_flags *const *const flags = region->flags;

    if (region->region_flags & REGION_NORTH_BOUNDARY)
        for (indexer_t h_idx = region->h_exterior.begin; h_idx < region->h_exterior.end; ++h_idx)
            flags[h_idx][region->v_exterior.begin] = CELL_BOUNDARY;

    if (region->region_flags & REGION_SOUTH_BOUNDARY)
        for (indexer_t h_idx = region->h_exterior.begin; h_idx < region->h_exterior.end; ++h_idx)
            flags[h_idx][region->v_exterior.end - 1] = CELL_BOUNDARY;

    if (region->region_flags & REGION_WEST_BOUNDARY)
        for (indexer_t v_idx = region->v_exterior.begin; v_idx < region->v_exterior.end; ++v_idx)
            flags[region->h_exterior.begin][v_idx] = CELL_BOUNDARY;

    if (region->region_flags & REGION_EAST_BOUNDARY)
        for (indexer_t v_idx = region->v_exterior.begin; v_idx < region->v_exterior.end; ++v_idx)
            flags[region->h_exterior.end - 1][v_idx] = CELL_BOUNDARY;
}

static enum region_flags compute_region_flags(
        const struct instance *const instance,
        const struct dim2 *const global_cell_counts,
        struct dim2 *const local_cell_counts,
        struct dim2 *const allocations)
{
    enum region_flags region_flags = REGION_UNREMARKABLE;
    struct dim2 ghost_counts = {
        .x = 0,
        .y = 0
    };

    bool x_ghosts_finalised = false;
    bool y_ghosts_finalised = false;

    if (instance->dim_extents.x > 1) {
        if (instance->cartesian_pos.x == 0) {
            /*
             * Westernmost region with at least one region to the east; marks a west boundary, definitely requires an
             * east ghost.
             */
            region_flags |= REGION_WEST_BOUNDARY | REGION_EAST_GHOST;
            ++ghost_counts.x;
            x_ghosts_finalised = true;
        }

        if (instance->cartesian_pos.x == instance->dim_extents.x - 1) {
            /*
             * Easternmost region with at least one region to the west; marks an east boundary, definitely requires a
             * west ghost. As the last region, also pick up any slack on the X axis.
             */
            region_flags |= REGION_EAST_BOUNDARY | REGION_WEST_GHOST;
            local_cell_counts->x += global_cell_counts->x % instance->dim_extents.x;
            ++ghost_counts.x;
            x_ghosts_finalised = true;
        }
    } else {
        // Single region on the X. Marks west and east boundaries; thus doesn't require east or west ghosts.
        region_flags |= REGION_WEST_BOUNDARY | REGION_EAST_BOUNDARY;
        x_ghosts_finalised = true;
    }

    if (instance->dim_extents.y > 1) {
        if (instance->cartesian_pos.y == 0) {
            /*
             * Northernmost region with at least one region to the south; marks a north boundary, definitely requires a
             * south ghost.
             */
            region_flags |= REGION_NORTH_BOUNDARY | REGION_SOUTH_GHOST;
            ++ghost_counts.y;
            y_ghosts_finalised = true;
        }

        if (instance->cartesian_pos.y == instance->dim_extents.y - 1) {
            /*
             * Southernmost region with at least one region to the north; marks a south boundary, definitely requires a
             * north ghost.
             */
            region_flags |= REGION_SOUTH_BOUNDARY | REGION_NORTH_GHOST;
            local_cell_counts->y += global_cell_counts->y % instance->dim_extents.y;
            ++ghost_counts.y;
            y_ghosts_finalised = true;
        }
    } else {
        // Single region on the Y. Marks north and south boundaries; thus doesn't require north or south ghosts.
        region_flags |= REGION_NORTH_BOUNDARY | REGION_SOUTH_BOUNDARY;
        y_ghosts_finalised = true;
    }

    if (!x_ghosts_finalised) {
        // Multiple regions on the X, and we're in the middle.
        region_flags |= REGION_WEST_GHOST | REGION_EAST_GHOST;
        ghost_counts.x += 2;
    }

    if (!y_ghosts_finalised) {
        // Multiple regions on the Y, and we're in the middle.
        region_flags |= REGION_NORTH_GHOST | REGION_SOUTH_GHOST;
        ghost_counts.y += 2;
    }

    // Verify that pole-wise boundaries and ghosts are mutually exclusive.
    static const enum region_flags mask = ~(~0U << (REGION_GHOST_START_POSITION - REGION_BOUNDARY_START_POSITION));
    assert(!(((region_flags >> REGION_BOUNDARY_START_POSITION) & (region_flags >> REGION_GHOST_START_POSITION))
        & mask));

    // Update allocations with final values.
    allocations->x = local_cell_counts->x + ghost_counts.x;
    allocations->y = local_cell_counts->y + ghost_counts.y;

    return region_flags;
}

static const struct exchange_cache * initialise_exchanger_cache_compute(struct region * const region,
    const enum matrix_identifier matrix)
{
    static compute_t dummy;
    struct exchange_cache * const cache = &region->exchange_cache[matrix];

    if (cache->initialised)
        return cache; // Already initialised; nothing to do.

    compute_t * const * data = NULL;

    switch (matrix) {
    case MATRIX_VELOCITY_X:
        data = region->velocity_x;
        break;
    case MATRIX_VELOCITY_Y:
        data = region->velocity_y;
        break;
    case MATRIX_TENTATIVE_VELOCITY_X:
        data = region->tentative_velocity_x;
        break;
    case MATRIX_TENTATIVE_VELOCITY_Y:
        data = region->tentative_velocity_y;
        break;
    case MATRIX_POISSON:
        data = region->poisson_source;
        break;
    case MATRIX_PRESSURE:
        data = region->pressure;
        break;
    default:
        // Unsupported matrix type.
        assert(0);
    }

    // ReSharper disable once CppDFANullDereference - null branch protected by assertion.
    cache->north_row = &data[0][region->v_exterior.begin];
    cache->north_ghost = region->region_flags & REGION_NORTH_GHOST ? &data[0][region->v_exterior.begin - 1] : &dummy;

    cache->south_row = &data[0][region->v_exterior.end - 1];
    cache->south_ghost = region->region_flags & REGION_SOUTH_GHOST ? &data[0][region->v_exterior.end] : &dummy;

    cache->west_col = data[region->h_exterior.begin];
    cache->west_ghost = region->region_flags & REGION_WEST_GHOST ? data[region->h_exterior.begin - 1] : &dummy;

    cache->east_col = data[region->h_exterior.end - 1];
    cache->east_ghost = region->region_flags & REGION_EAST_GHOST ? data[region->h_exterior.end] : &dummy;

    cache->initialised = true;
    return cache;
}

static const struct exchange_cache * initialise_exchanger_cache_flags(struct region * const region,
    const enum matrix_identifier matrix)
{
    static enum cell_flags dummy;

    assert(matrix == MATRIX_FLAGS); // Ensure supported matrix type.
    struct exchange_cache * const cache = &region->exchange_cache[matrix];

    if (cache->initialised)
        return cache; // Already initialised; nothing to do.

    enum cell_flags * const * const data = region->flags;

    cache->north_row = &data[0][region->v_exterior.begin];
    cache->north_ghost = region->region_flags & REGION_NORTH_GHOST ? &data[0][region->v_exterior.begin - 1] : &dummy;

    cache->south_row = &data[0][region->v_exterior.end - 1];
    cache->south_ghost = region->region_flags & REGION_SOUTH_GHOST ? &data[0][region->v_exterior.end] : &dummy;

    cache->west_col = data[region->h_exterior.begin];
    cache->west_ghost = region->region_flags & REGION_WEST_GHOST ? data[region->h_exterior.begin - 1] : &dummy;

    cache->east_col = data[region->h_exterior.end - 1];
    cache->east_ghost = region->region_flags & REGION_EAST_GHOST ? data[region->h_exterior.end] : &dummy;

    cache->initialised = true;
    return cache;
}

/**
 * Set arbitrary values for the non-covered areas of the tentative velocities and pressure matrices, typically following
 * a computation of tentatives and preceding a computation of the pressure Poisson term. See Eqns. 3.41 and 3.42 of
 * Griebel.
 *
 * @param region The region containing the velocity and pressure matrices.
 */
static void fix_tentative_boundaries(const struct region * const region)
{
    compute_t * const * const velocity_x = region->velocity_x;
    compute_t * const * const velocity_y = region->velocity_y;
    compute_t * const * const tentative_velocity_x = region->tentative_velocity_x;
    compute_t * const * const tentative_velocity_y = region->tentative_velocity_y;
    compute_t * const * const pressure = region->pressure;

    // F_{0, j} and F_{i_{max}, j}
    // p_{0, j} and p_{i_{max}+1, j}
    // for j = 1, ..., j_{max}
    for (indexer_t v_idx = region->v_interior.begin; v_idx < region->v_interior.end; ++v_idx) {
        tentative_velocity_x[region->h_exterior.begin][v_idx] = velocity_x[region->h_exterior.begin][v_idx];
        tentative_velocity_x[region->h_interior.end - 1][v_idx] = velocity_x[region->h_interior.end - 1][v_idx];
        pressure[region->h_exterior.begin][v_idx] = pressure[region->h_exterior.begin + 1][v_idx];
        pressure[region->h_exterior.end - 1][v_idx] = pressure[region->h_interior.end - 1][v_idx];
    }

    // G_{i, 0} and G_{i, j_{max}}
    // p_{0, i} and p_{i, j_{max}+1}
    // for i = 1, ..., i_{max}
    for (indexer_t h_idx = region->h_interior.begin; h_idx < region->h_interior.end; ++h_idx) {
        tentative_velocity_y[h_idx][region->v_exterior.begin] = velocity_y[h_idx][region->v_exterior.begin];
        tentative_velocity_y[h_idx][region->v_interior.end - 1] = velocity_y[h_idx][region->v_interior.end - 1];
        pressure[h_idx][region->v_exterior.begin] = pressure[h_idx][region->v_exterior.begin + 1];
        pressure[h_idx][region->v_exterior.end - 1] = pressure[h_idx][region->v_interior.end - 1];
    }
}

/**
 * Perform a single-phased (red or black) cycle of SOR to update the pressure scalar field. Run for multiple passes over
 * different phases for a fully populated field.
 *
 * @param region The region containing the pressure matrix on which SOR should be performed.
 * @param omega The relaxation parameter; see Chapter 8.3 of Stoer, J. & Bulirsch, R. (1980).
 *  Introduction to Numerical Analysis.
 * @param phase The phase of the checkerboard pattern to populate in the pressure matrix.
 */
static void sor_cycle_phase(const struct region *const region, const compute_t omega, const enum sor_phase phase)
{
    const compute_t step_sq = region->derived_params.resolution_sq;

    for (indexer_t h_idx = region->h_interior.begin; h_idx < region->h_interior.end; ++h_idx) {
        // Align to correct parity for RB indexing. See Figure 2 of https://arxiv.org/abs/1401.0763.
        indexer_t v_start = region->v_interior.begin;
        v_start += h_idx + v_start & 1 ^ (indexer_t) phase;

        for (indexer_t v_idx = v_start; v_idx < region->v_interior.end; v_idx += 2) {
            compute_t weight;

            // Epsilon parameters indicate whether fluid lies in the cell in the corresponding direction.
            compute_t epsilon[4] = {
                (region->flags[h_idx][v_idx + 1] & CELL_FLUID) >> 4, // North
                (region->flags[h_idx][v_idx - 1] & CELL_FLUID) >> 4, // South
                (region->flags[h_idx + 1][v_idx] & CELL_FLUID) >> 4, // East
                (region->flags[h_idx - 1][v_idx] & CELL_FLUID) >> 4, // West
            };

            const compute_t poisson[5] = {
                region->poisson_source[h_idx][v_idx + 1], // North
                region->poisson_source[h_idx][v_idx - 1], // South
                region->poisson_source[h_idx + 1][v_idx], // East
                region->poisson_source[h_idx - 1][v_idx], // West
                region->poisson_source[h_idx][v_idx],     // Self
            };

            const compute_t pressure[5] = {
                region->pressure[h_idx][v_idx + 1], // North
                region->pressure[h_idx][v_idx - 1], // South
                region->pressure[h_idx + 1][v_idx], // East
                region->pressure[h_idx - 1][v_idx], // West
                region->pressure[h_idx][v_idx],     // Self
            };

            if (region->flags[h_idx][v_idx] & CELL_FLUID)

                weight = omega / ((epsilon[TAGS_EAST] + epsilon[TAGS_WEST] + epsilon[TAGS_NORTH] + epsilon[TAGS_SOUTH])
                    * step_sq);

            else {

                epsilon[TAGS_EAST] = 0.0;
                epsilon[TAGS_WEST] = pressure[TAGS_WEST] == 0.0 ?
                    0.0 : omega * pressure[TAGS_SELF] / pressure[TAGS_WEST] / step_sq;
                epsilon[TAGS_SOUTH] = 0.0;
                epsilon[TAGS_NORTH] = pressure[TAGS_NORTH] == 0.0 ?
                    0.0 : poisson[TAGS_SELF] / pressure[TAGS_NORTH] / step_sq;

                weight = 1.0;

            }

            const compute_t x_spatial = epsilon[TAGS_EAST] * pressure[TAGS_EAST] +
                    epsilon[TAGS_WEST] * pressure[TAGS_WEST];

            const compute_t y_spatial = epsilon[TAGS_NORTH] * pressure[TAGS_NORTH] +
                epsilon[TAGS_SOUTH] * pressure[TAGS_SOUTH];

            region->pressure[h_idx][v_idx] = (1 - omega) * pressure[TAGS_SELF] + weight *
                (step_sq * (x_spatial + y_spatial) - poisson[TAGS_SELF]);
        }
    }
}

struct region region_create(const struct instance *const instance)
{
    // Number of cells per unit-distance.
    static const unsigned int resolution = 128;

    // Scaled spatial dimensions by the resolution, to map problem size to problem cell counts.
    const struct dim2 global_cell_counts = {
        .x = (indexer_t) ceil((compute_t) resolution * instance->problem_size.x),
        .y = (indexer_t) ceil((compute_t) resolution * instance->problem_size.y)
    };

    // Provisional region cell counts based on uniform distribution, plus a single unit to map cells to VTK points.
    struct dim2 local_cell_counts = {
        .x = global_cell_counts.x / instance->dim_extents.x + 1,
        .y = global_cell_counts.y / instance->dim_extents.y + 1
    };

    // Provisional sizes to allocate on the heap. This encapsulates everything, including ghosts and boundaries.
    struct dim2 allocations = {
        .x = local_cell_counts.x,
        .y = local_cell_counts.y
    };

    /*
     * Determine properties of the individual region according to its position within the problem space (based on
     * location within the Cartesian virtual topology assigned by MPI). This updates the local cell counts and
     * allocation requirements to their final values.
     */
    const enum region_flags region_flags = compute_region_flags(instance, &global_cell_counts, &local_cell_counts,
        &allocations);

    /*
     * Compute exterior index iterator boundaries, shifting rightwards or downwards to accommodate west and north
     * ghosts, respectively.
     *
     * N.B. East and south ghosts are represented at the end of their respective arrays, and are
     * accounted for by the updated allocation counts. This provides the guarantee that solvers iterating over the
     * interior of a region may safely index one place beyond the position of any interior cell in any direction.
     */
    const struct iterator h_exterior = {
        .begin = !!(region_flags & REGION_WEST_GHOST),
        .end = local_cell_counts.x + !!(region_flags & REGION_WEST_GHOST)
    };

    const struct iterator v_exterior = {
        .begin = !!(region_flags & REGION_NORTH_GHOST),
        .end = local_cell_counts.y + !!(region_flags & REGION_NORTH_GHOST)
    };

    struct region region = {
        .velocity_x = alloc_2d_compute_array(allocations),
        .velocity_y = alloc_2d_compute_array(allocations),
        .tentative_velocity_x = alloc_2d_compute_array(allocations),
        .tentative_velocity_y = alloc_2d_compute_array(allocations),
        .pressure = alloc_2d_compute_array(allocations),
        .poisson_source = alloc_2d_compute_array(allocations),
        .flags = alloc_2d_flags_array(allocations),

        .region_flags = region_flags,

        /*
         * Compute interior index iterator boundaries, defined to be the respective exteriors minus any artificial
         * spatial boundaries created by the problem specification.
         */
        .h_interior = {
            .begin = h_exterior.begin + !!(region_flags & REGION_WEST_BOUNDARY),
            .end = h_exterior.end - !!(region_flags & REGION_EAST_BOUNDARY)
        },

        .v_interior = {
            .begin = v_exterior.begin + !!(region_flags & REGION_NORTH_BOUNDARY),
            .end = v_exterior.end - !!(region_flags & REGION_SOUTH_BOUNDARY)
        },

        .h_exterior = h_exterior,
        .v_exterior = v_exterior,
        .resolution = resolution,

        // Compute the region's absolute positioning within the problem space, required for the RB-SOL PDE solver.
        .indents = instance_get_indentations(instance, local_cell_counts),

        .initial_velocity_x = 1.0,
        .initial_velocity_y = 0.0,
        .initial_pressure = 0.0,
        .initial_flag = CELL_FLUID,

        // Create MPI types for reliable data transport.
        .compute_row_t = create_row_t(allocations.x, allocations.y, MPI_COMPUTE, sizeof(compute_t)),
        .compute_col_t = create_column_t(allocations.y, MPI_COMPUTE),
        .flags_row_t = create_row_t(allocations.x, allocations.y, MPI_INT, sizeof(enum cell_flags)),
        .flags_col_t = create_column_t(allocations.y, MPI_INT),

        // Store values commonly used in solver loops, dependent on the region parameters.
        .derived_params = {
            .resolution_sq = resolution * resolution
        }
    };

    // Indicate that no exchange caches have been initialised.
    assert(sizeof(region.exchange_cache) / sizeof(region.exchange_cache[0]) == MATRIX_TYPES_COUNT);
    for (indexer_t cache_idx = 0; cache_idx < MATRIX_TYPES_COUNT; ++cache_idx)
        region.exchange_cache[cache_idx].initialised = false;

    // All cells are initially fluid. This count can be decremented throughout the simulation.
    region.fluid_cell_count = (region.h_interior.end - region.h_interior.begin) *
        (region.v_interior.end - region.v_interior.begin);

    assert(h_exterior.end <= allocations.x);
    assert(v_exterior.end <= allocations.y);

    return region;
}

void region_destroy(struct region *const region)
{
    MPI_Type_free(&region->compute_row_t);
    MPI_Type_free(&region->compute_col_t);
    MPI_Type_free(&region->flags_row_t);
    MPI_Type_free(&region->flags_col_t);

    free_2d_array((void **) region->velocity_x);
    free_2d_array((void **) region->velocity_y);
    free_2d_array((void **) region->tentative_velocity_x);
    free_2d_array((void **) region->tentative_velocity_y);
    free_2d_array((void **) region->pressure);
    free_2d_array((void **) region->flags);
}

void region_describe(const struct region *const region, FILE *const destination)
{
    fprintf(destination,
            "Region statistics:\n\t"
            "North boundary? %s\n\t"
            "South boundary? %s\n\t"
            "West boundary? %s\n\t"
            "East boundary? %s\n\t"
            "North ghost? %s\n\t"
            "South ghost? %s\n\t"
            "West ghost? %s\n\t"
            "East ghost? %s\n\t"
            "Horizontal interior: [%d, %d]\n\t"
            "Vertical interior: [%d, %d]\n\t"
            "Horizontal exterior: [%d, %d]\n\t"
            "Vertical exterior: [%d, %d]\n\t"
            "Horizontal indent: %d cells\n\t"
            "Vertical indent: %d cells\n",

            region->region_flags & REGION_NORTH_BOUNDARY ? "Yes" : "No",
            region->region_flags & REGION_SOUTH_BOUNDARY ? "Yes" : "No",
            region->region_flags & REGION_WEST_BOUNDARY ? "Yes" : "No",
            region->region_flags & REGION_EAST_BOUNDARY ? "Yes" : "No",
            region->region_flags & REGION_NORTH_GHOST ? "Yes" : "No",
            region->region_flags & REGION_SOUTH_GHOST ? "Yes" : "No",
            region->region_flags & REGION_WEST_GHOST ? "Yes" : "No",
            region->region_flags & REGION_EAST_GHOST ? "Yes" : "No",
            region->h_interior.begin, region->h_interior.end - 1,
            region->v_interior.begin, region->v_interior.end - 1,
            region->h_exterior.begin, region->h_exterior.end - 1,
            region->v_exterior.begin, region->v_exterior.end - 1,
            region->indents.x,
            region->indents.y);
}

void region_apply_boundary_conditions(const struct region *const region)
{
    compute_t *const *const velocity_x = region->velocity_x;
    compute_t *const *const velocity_y = region->velocity_y;
    enum cell_flags *const *const flags = region->flags;

    for (indexer_t v_idx = region->v_exterior.begin; v_idx < region->v_exterior.end; ++v_idx) {
        // Fluid freely flows in from the west
        velocity_x[region->h_exterior.begin][v_idx] = velocity_x[region->h_exterior.begin + 1][v_idx];
        velocity_y[region->h_exterior.begin][v_idx] = velocity_y[region->h_exterior.begin + 1][v_idx];

        // Fluid freely flows out to the east
        velocity_x[region->h_exterior.end - 2][v_idx] = velocity_x[region->h_exterior.end - 3][v_idx];
        velocity_y[region->h_exterior.end - 1][v_idx] = velocity_y[region->h_exterior.end - 2][v_idx];
    }

    for (indexer_t h_idx = region->h_exterior.begin; h_idx < region->h_exterior.end; ++h_idx) {
        /*
         * The vertical velocity approaches zero at the north and south boundaries, but fluid flows freely in the
         * horizontal direction. */
        velocity_y[h_idx][region->v_exterior.end - 2] = 0.0;
        velocity_x[h_idx][region->v_exterior.end - 1] = velocity_x[h_idx][region->v_exterior.end - 2];

        velocity_y[h_idx][region->v_exterior.begin] = 0.0;
        velocity_x[h_idx][region->v_exterior.begin] = velocity_x[h_idx][region->v_exterior.begin + 1];
    }

    /*
     * Apply no-slip boundary conditions to cells that are adjacent to internal obstacle cells. This forces the
     * velocities to tend towards zero in these cells.
     */
    for (indexer_t h_idx = region->h_interior.begin; h_idx < region->h_interior.end; ++h_idx)
        for (indexer_t v_idx = region->v_interior.begin; v_idx < region->v_interior.end; ++v_idx)
            if (flags[h_idx][v_idx] & CELL_FLUID_ALL)
                switch (flags[h_idx][v_idx]) {
                case CELL_FLUID_NORTH:
                    velocity_y[h_idx][v_idx] = 0.0;
                    velocity_x[h_idx][v_idx] = -velocity_x[h_idx][v_idx + 1];
                    velocity_x[h_idx - 1][v_idx] = -velocity_x[h_idx - 1][v_idx + 1];
                    break;
                case CELL_FLUID_EAST:
                    velocity_x[h_idx][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx] = -velocity_y[h_idx + 1][v_idx];
                    velocity_y[h_idx][v_idx - 1] = -velocity_y[h_idx + 1][v_idx - 1];
                    break;
                case CELL_FLUID_SOUTH:
                    velocity_y[h_idx][v_idx - 1] = 0.0;
                    velocity_x[h_idx][v_idx] = -velocity_x[h_idx][v_idx - 1];
                    velocity_x[h_idx - 1][v_idx] = -velocity_x[h_idx - 1][v_idx - 1];
                    break;
                case CELL_FLUID_WEST:
                    velocity_x[h_idx - 1][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx] = -velocity_y[h_idx - 1][v_idx];
                    velocity_y[h_idx][v_idx - 1] = -velocity_y[h_idx - 1][v_idx - 1];
                    break;
                case CELL_FLUID_NORTHEAST:
                    velocity_y[h_idx][v_idx] = 0.0;
                    velocity_x[h_idx][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx - 1] = -velocity_y[h_idx + 1][v_idx - 1];
                    velocity_x[h_idx - 1][v_idx] = -velocity_x[h_idx - 1][v_idx + 1];
                    break;
                case CELL_FLUID_SOUTHEAST:
                    velocity_y[h_idx][v_idx - 1] = 0.0;
                    velocity_x[h_idx][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx] = -velocity_y[h_idx + 1][v_idx];
                    velocity_x[h_idx - 1][v_idx] = -velocity_x[h_idx - 1][v_idx - 1];
                    break;
                case CELL_FLUID_SOUTHWEST:
                    velocity_y[h_idx][v_idx - 1] = 0.0;
                    velocity_x[h_idx - 1][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx] = -velocity_y[h_idx - 1][v_idx];
                    velocity_x[h_idx][v_idx] = -velocity_x[h_idx][v_idx - 1];
                    break;
                case CELL_FLUID_NORTHWEST:
                    velocity_y[h_idx][v_idx] = 0.0;
                    velocity_x[h_idx - 1][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx - 1] = -velocity_y[h_idx - 1][v_idx - 1];
                    velocity_x[h_idx][v_idx] = -velocity_x[h_idx][v_idx + 1];
                    break;
                default:;
                }

    if (region->region_flags & REGION_WEST_BOUNDARY) {
        /*
         * If we're on a western boundary, fix the western-edge velocities such that there is a continual flow of fluid
         * into the simulation space.
         */
        // TODO: is this first assignment needed?
        velocity_y[region->h_exterior.begin][region->v_exterior.begin] =
                2 * region->initial_velocity_y - velocity_y[region->h_exterior.begin + 1][region->v_exterior.begin];

        for (indexer_t v_idx = region->v_interior.begin; v_idx < region->v_interior.end; ++v_idx) {
            velocity_x[region->h_exterior.begin][v_idx] = region->initial_velocity_x;
            velocity_y[region->h_exterior.begin][v_idx] = 2 * region->initial_velocity_y -
                velocity_y[region->h_exterior.begin + 1][v_idx];
        }
    }
}

void region_update_velocities(const struct region *const region, const struct instance *instance)
{
    /*
     * The pressure differential factors are the constants implied by the discretisation of the momentum equation. They
     * represent fixed-axis grid spacings, warped by the timestep duration, to numerically approximate the next velocity
     * values in terms of the computed pressures.
     */
    const compute_t x_pressure_diff_factor = instance->timestep_duration * region->resolution;
    const compute_t y_pressure_diff_factor = instance->timestep_duration * region->resolution;

    const enum region_flags flags = region->region_flags;

    /*
     * N.B. If the region represents an eastern or southern boundary, we want to write to up to one before the interior
     * edge. Otherwise, we need to cover the entire space (so adjacent regions can access correct data via HX).
     */

    // X velocities
    indexer_t h_bound = flags & REGION_EAST_BOUNDARY ? region->h_interior.end - 1 : region->h_interior.end;
    indexer_t v_bound = region->v_interior.end;

    for (indexer_t h_idx = region->h_interior.begin; h_idx < h_bound; ++h_idx)
        for (indexer_t v_idx = region->v_interior.begin; v_idx < v_bound; ++v_idx)
            if (region->flags[h_idx][v_idx] & CELL_FLUID && region->flags[h_idx + 1][v_idx] & CELL_FLUID)
                region->velocity_x[h_idx][v_idx] = region->tentative_velocity_x[h_idx][v_idx] -
                    (region->pressure[h_idx + 1][v_idx] - region->pressure[h_idx][v_idx]) * x_pressure_diff_factor;

    // Y velocities
    h_bound = region->h_interior.end;
    v_bound = flags & REGION_SOUTH_BOUNDARY ? region->v_interior.end - 1 : region->v_interior.end;

    for (indexer_t h_idx = region->h_interior.begin; h_idx < h_bound; ++h_idx)
        for (indexer_t v_idx = region->v_interior.begin; v_idx < v_bound; ++v_idx)
            if (region->flags[h_idx][v_idx] & CELL_FLUID && region->flags[h_idx][v_idx + 1] & CELL_FLUID)
                region->velocity_y[h_idx][v_idx] = region->tentative_velocity_y[h_idx][v_idx] -
                    (region->pressure[h_idx][v_idx + 1] - region->pressure[h_idx][v_idx]) * y_pressure_diff_factor;
}

void region_compute_tentative_velocities(const struct region *const region, const struct instance *instance)
{
    static const compute_t reynolds = 500.0;
    static const double gamma = 0.9; // Upwind differencing factor in PDE discretisation

    // Get local copies of pointers to avoid excessive dereferencing in loop.
    compute_t * const * const velocity_x = region->velocity_x;
    compute_t * const * const velocity_y = region->velocity_y;
    compute_t * const * const tentative_velocity_x = region->tentative_velocity_x;
    compute_t * const * const tentative_velocity_y = region->tentative_velocity_y;
    enum cell_flags * const * const flags = region->flags;

    const compute_t quarter_resolution = region->resolution / 4.0;
    const compute_t sq_resolution = region->resolution * region->resolution;

    for (indexer_t h_idx = region->h_interior.begin; h_idx < region->h_interior.end - 1; ++h_idx)
        for (indexer_t v_idx = region->v_interior.begin; v_idx < region->v_interior.end; ++v_idx)
            if (flags[h_idx][v_idx] & CELL_FLUID && flags[h_idx + 1][v_idx] & CELL_FLUID) {

                const double self_advection_x =
                    (
                        (velocity_x[h_idx][v_idx] + velocity_x[h_idx + 1][v_idx]) *
                        (velocity_x[h_idx][v_idx] + velocity_x[h_idx + 1][v_idx]) +
                        gamma * fabs(velocity_x[h_idx][v_idx] + velocity_x[h_idx + 1][v_idx]) *
                        (velocity_x[h_idx][v_idx] - velocity_x[h_idx + 1][v_idx]) -
                        (velocity_x[h_idx - 1][v_idx] + velocity_x[h_idx][v_idx]) *
                        (velocity_x[h_idx - 1][v_idx] + velocity_x[h_idx][v_idx]) -
                        gamma * fabs(velocity_x[h_idx - 1][v_idx] + velocity_x[h_idx][v_idx]) *
                        (velocity_x[h_idx - 1][v_idx] - velocity_x[h_idx][v_idx])
                    ) * quarter_resolution;

                const double cross_advection_y =
                    (
                        (velocity_y[h_idx][v_idx] + velocity_y[h_idx + 1][v_idx]) *
                        (velocity_x[h_idx][v_idx] + velocity_x[h_idx][v_idx + 1]) +
                        gamma * fabs(velocity_y[h_idx][v_idx] + velocity_y[h_idx + 1][v_idx]) *
                        (velocity_x[h_idx][v_idx] - velocity_x[h_idx][v_idx + 1]) -
                        (velocity_y[h_idx][v_idx - 1] + velocity_y[h_idx + 1][v_idx - 1]) *
                        (velocity_x[h_idx][v_idx - 1] + velocity_x[h_idx][v_idx]) -
                        gamma * fabs(velocity_y[h_idx][v_idx - 1] +
                            velocity_y[h_idx + 1][v_idx - 1]) *
                        (velocity_x[h_idx][v_idx - 1] - velocity_x[h_idx][v_idx])
                    ) * quarter_resolution;

                const double diffusion =
                    (
                        velocity_x[h_idx + 1][v_idx] -
                        2.0 * velocity_x[h_idx][v_idx] +
                        velocity_x[h_idx - 1][v_idx] +
                        velocity_x[h_idx][v_idx + 1] -
                        2.0 * velocity_x[h_idx][v_idx] +
                        velocity_x[h_idx][v_idx - 1]
                    ) * sq_resolution;

                tentative_velocity_x[h_idx][v_idx] = velocity_x[h_idx][v_idx] + instance->timestep_duration *
                    (diffusion / reynolds - self_advection_x - cross_advection_y);

            } else
                // If both adjacent cells are not fluids, the velocity is unchanged.
                tentative_velocity_x[h_idx][v_idx] = velocity_x[h_idx][v_idx];

    for (indexer_t h_idx = region->h_interior.begin; h_idx < region->h_interior.end; ++h_idx)
        for (indexer_t v_idx = region->v_interior.begin; v_idx < region->v_interior.end - 1; ++v_idx)
            if (flags[h_idx][v_idx] & CELL_FLUID && flags[h_idx][v_idx + 1] & CELL_FLUID) {

                const double cross_advection_x =
                    (
                        (velocity_x[h_idx][v_idx] + velocity_x[h_idx][v_idx + 1]) *
                        (velocity_y[h_idx][v_idx] + velocity_y[h_idx + 1][v_idx]) +
                        gamma * fabs(velocity_x[h_idx][v_idx] + velocity_x[h_idx][v_idx + 1]) *
                        (velocity_y[h_idx][v_idx] - velocity_y[h_idx + 1][v_idx]) -
                        (velocity_x[h_idx - 1][v_idx] + velocity_x[h_idx - 1][v_idx + 1]) *
                        (velocity_y[h_idx - 1][v_idx] + velocity_y[h_idx][v_idx]) -
                        gamma * fabs(velocity_x[h_idx - 1][v_idx] +
                            velocity_x[h_idx - 1][v_idx + 1]) *
                        (velocity_y[h_idx - 1][v_idx] - velocity_y[h_idx][v_idx])
                    ) * quarter_resolution;

                const double self_advection_y =
                    (
                        (velocity_y[h_idx][v_idx] + velocity_y[h_idx][v_idx + 1]) *
                        (velocity_y[h_idx][v_idx] + velocity_y[h_idx][v_idx + 1]) +
                        gamma * fabs(velocity_y[h_idx][v_idx] + velocity_y[h_idx][v_idx + 1]) *
                        (velocity_y[h_idx][v_idx] - velocity_y[h_idx][v_idx + 1]) -
                        (velocity_y[h_idx][v_idx - 1] + velocity_y[h_idx][v_idx]) *
                        (velocity_y[h_idx][v_idx - 1] + velocity_y[h_idx][v_idx]) -
                        gamma * fabs(velocity_y[h_idx][v_idx - 1] + velocity_y[h_idx][v_idx]) *
                        (velocity_y[h_idx][v_idx - 1] - velocity_y[h_idx][v_idx])
                    ) * quarter_resolution;

                const double diffusion =
                    (
                        velocity_y[h_idx + 1][v_idx] -
                        2.0 * velocity_y[h_idx][v_idx] +
                        velocity_y[h_idx - 1][v_idx] +
                        velocity_y[h_idx][v_idx + 1] -
                        2.0 * velocity_y[h_idx][v_idx] +
                        velocity_y[h_idx][v_idx - 1]
                    ) * sq_resolution;

                tentative_velocity_y[h_idx][v_idx] = velocity_y[h_idx][v_idx] + instance->timestep_duration *
                    (diffusion / reynolds - cross_advection_x - self_advection_y);

            } else
                // If both adjacent cells are not fluids, the velocity is unchanged.
                tentative_velocity_y[h_idx][v_idx] = velocity_y[h_idx][v_idx];
}

void region_compute_poisson_source(const struct region *const region, const struct instance *instance)
{
    fix_tentative_boundaries(region);

    compute_t * const * const tentative_velocity_x = region->tentative_velocity_x;
    compute_t * const * const tentative_velocity_y = region->tentative_velocity_y;
    compute_t * const * const poisson_source = region->poisson_source;
    enum cell_flags * const * const flags = region->flags;
    
    for (indexer_t h_idx = region->h_interior.begin; h_idx < region->h_interior.end; ++h_idx)
        for (indexer_t v_idx = region->v_interior.begin; v_idx < region->v_interior.end; ++v_idx)
            if (flags[h_idx][v_idx] & CELL_FLUID) {
                const compute_t x_tent_vel_diff = (tentative_velocity_x[h_idx][v_idx] -
                    tentative_velocity_x[h_idx - 1][v_idx]) * region->resolution;

                const compute_t y_tent_vel_diff = (tentative_velocity_y[h_idx][v_idx] -
                    tentative_velocity_y[h_idx][v_idx - 1]) * region->resolution;

                poisson_source[h_idx][v_idx] = (x_tent_vel_diff + y_tent_vel_diff) / instance->timestep_duration;
            }
}

void region_sor_cycle(struct region *const region, const struct instance *instance)
{
    /*
     * Perform red-black SOR to reduce data dependencies. If tracing the computation over the interior pressure grid,
     * this produces a checkerboard. See https://arxiv.org/abs/1401.0763 for discussion. We also unroll the outermost
     * red-phase black-phase loop to avoid the branch instruction and excessive BP failures.
     */

    sor_cycle_phase(region, instance->sor_omega, SOR_RED);
    region_exchange(region, MATRIX_PRESSURE, instance);
    sor_cycle_phase(region, instance->sor_omega, SOR_BLACK);
    region_exchange(region, MATRIX_PRESSURE, instance);
}

void region_exchange(
    struct region *const region,
    const enum matrix_identifier matrix,
    const struct instance * const instance)
{
    const struct exchange_cache * exchanger = NULL;
    MPI_Datatype row_t = 0;
    MPI_Datatype col_t = 0;

    switch (matrix) {
    case MATRIX_VELOCITY_X:
    case MATRIX_VELOCITY_Y:
    case MATRIX_TENTATIVE_VELOCITY_X:
    case MATRIX_TENTATIVE_VELOCITY_Y:
    case MATRIX_POISSON:
    case MATRIX_PRESSURE:
        exchanger = initialise_exchanger_cache_compute(region, matrix);
        row_t = region->compute_row_t;
        col_t = region->compute_col_t;
        break;
    case MATRIX_FLAGS:
        exchanger = initialise_exchanger_cache_flags(region, matrix);
        row_t = region->flags_row_t;
        col_t = region->flags_col_t;
        break;
    default: ;
    }

    // Ensure that we have selected a valid matrix.
    assert(exchanger != NULL && row_t != NULL && col_t != NULL);

    MPI_Request requests[8];
    int request_idx = 0;

    // North
    MPI_Isend(exchanger->north_row, 1, row_t, instance->neighbours.north, TAGS_NORTH, instance->cartesian_comm,
        &requests[request_idx++]);
    MPI_Irecv(exchanger->north_ghost, 1, row_t, instance->neighbours.north, TAGS_SOUTH, instance->cartesian_comm,
        &requests[request_idx++]);

    // South
    MPI_Isend(exchanger->south_row, 1, row_t, instance->neighbours.south, TAGS_SOUTH, instance->cartesian_comm,
        &requests[request_idx++]);
    MPI_Irecv(exchanger->south_ghost, 1, row_t, instance->neighbours.south, TAGS_NORTH, instance->cartesian_comm,
        &requests[request_idx++]);

    // East
    MPI_Isend(exchanger->east_col, 1, col_t, instance->neighbours.east, TAGS_EAST, instance->cartesian_comm,
        &requests[request_idx++]);
    MPI_Irecv(exchanger->east_ghost, 1, col_t, instance->neighbours.east, TAGS_WEST, instance->cartesian_comm,
        &requests[request_idx++]);

    // West
    MPI_Isend(exchanger->west_col, 1, col_t, instance->neighbours.west, TAGS_WEST, instance->cartesian_comm,
        &requests[request_idx++]);
    MPI_Irecv(exchanger->west_ghost, 1, col_t, instance->neighbours.west, TAGS_EAST, instance->cartesian_comm,
        &requests[request_idx++]);

    MPI_Waitall(request_idx, requests, MPI_STATUSES_IGNORE);
}

compute_t region_compute_poisson_residual(const struct region *const region)
{
    const compute_t step_sq = region->derived_params.resolution_sq;
    compute_t residual = 0.0;

    for (indexer_t h_idx = region->h_interior.begin; h_idx < region->h_interior.end; ++h_idx)
        for (indexer_t v_idx = region->v_interior.begin; v_idx < region->v_interior.end; ++v_idx)
            if (region->flags[h_idx][v_idx] & CELL_FLUID) {
                const double epsilon_east = !!(region->flags[h_idx + 1][v_idx] & CELL_FLUID);
                const double epsilon_west = !!(region->flags[h_idx - 1][v_idx] & CELL_FLUID);
                const double epsilon_north = !!(region->flags[h_idx][v_idx + 1] & CELL_FLUID);
                const double epsilon_south = !!(region->flags[h_idx][v_idx - 1] & CELL_FLUID);

                const double x_residual = (
                    epsilon_east * (region->pressure[h_idx + 1][v_idx] - region->pressure[h_idx][v_idx]) -
                    epsilon_west * (region->pressure[h_idx][v_idx] - region->pressure[h_idx - 1][v_idx])
                ) * step_sq;

                const double y_residual = (
                    epsilon_north * (region->pressure[h_idx][v_idx + 1] - region->pressure[h_idx][v_idx]) -
                    epsilon_south * (region->pressure[h_idx][v_idx] - region->pressure[h_idx][v_idx - 1])
                ) * step_sq;

                const double add = x_residual + y_residual - region->poisson_source[h_idx][v_idx];
                residual += add * add;
            }

    return residual;
}

void region_initialise(struct region *const region, const struct instance *const instance)
{
    // Transform the NACA digits into the scale expected by the initial boundary calculi.
    const float maximum_camber = (float) instance->naca_specifier.maximum_camber / 100.0f;
    const float edge_distance = (float) instance->naca_specifier.edge_distance / 10.0f;
    const float thickness = (float) instance->naca_specifier.maximum_thickness / 100.0f;

    for (indexer_t h_idx = region->h_exterior.begin; h_idx < region->h_exterior.end; ++h_idx) {
        // Populate all cells' information matrices with fixed initial values.
        for (indexer_t v_idx = region->v_exterior.begin; v_idx < region->v_exterior.end; ++v_idx) {
            region->velocity_x[h_idx][v_idx] = region->initial_velocity_x;
            region->velocity_y[h_idx][v_idx] = region->initial_velocity_y;
            region->pressure[h_idx][v_idx] = region->initial_pressure;
            region->flags[h_idx][v_idx] = region->initial_flag;
        }

        // Compute the vertical index boundaries of the airfoil body at the fixed horizontal index.
        const struct iterator v_idx_boundaries = get_initial_v_idx_boundaries(
                region, instance->problem_size.y, maximum_camber, edge_distance, thickness, h_idx);

        // Populate the airfoil body with boundary markers.
        for (indexer_t v_idx = v_idx_boundaries.begin; v_idx < v_idx_boundaries.end; ++v_idx)
            region->flags[h_idx][v_idx] = CELL_BOUNDARY;
    }

    write_initial_extreme_boundaries(region);

    // TODO: probably need a barrier (likely HX) here, as we're reading into other ranks' spaces after they write boundaries.
    // Mask in additional directional indicator flags for non-fluid cells, describing presence of nearby fluid cells.
    enum cell_flags * const * const flags = region->flags;

    for (indexer_t h_idx = region->h_interior.begin; h_idx < region->h_interior.end; ++h_idx)
        for (indexer_t v_idx = region->v_interior.begin; v_idx < region->v_interior.end; ++v_idx)
            if (!(flags[h_idx][v_idx] & CELL_FLUID)) {
                --region->fluid_cell_count;

                if (flags[h_idx - 1][v_idx] & CELL_FLUID)
                    flags[h_idx][v_idx] |= CELL_FLUID_WEST;
                if (flags[h_idx + 1][v_idx] & CELL_FLUID)
                    flags[h_idx][v_idx] |= CELL_FLUID_EAST;
                if (flags[h_idx][v_idx - 1] & CELL_FLUID)
                    flags[h_idx][v_idx] |= CELL_FLUID_SOUTH;
                if (flags[h_idx][v_idx + 1] & CELL_FLUID)
                    flags[h_idx][v_idx] |= CELL_FLUID_NORTH;
            }
}

void region_serialise_vtk(
    const struct region *const region,
    const struct instance *const instance,
    FILE *const destination)
{
    const struct dim2 shifted_indents = instance_translate_to_cells(instance, &region->indents);
    const struct dim2 size = {
        .x = region->h_exterior.end - region->h_exterior.begin - 1,
        .y = region->v_exterior.end - region->v_exterior.begin - 1
    };

    fprintf(destination,
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "\t<RectilinearGrid WholeExtent=\"%u %u %u %u 0 0\" GhostLevel=\"0\">\n"
        "\t\t<Piece Extent=\"%u %u %u %u 0 0\">\n"
        "\t\t\t<Coordinates>\n",

        shifted_indents.x, shifted_indents.x + size.x,
        shifted_indents.y, shifted_indents.y + size.y,
        shifted_indents.x, shifted_indents.x + size.x,
        shifted_indents.y, shifted_indents.y + size.y);

    fprintf(destination, "\t\t\t\t<DataArray type=\"Float64\" format=\"ascii\" Name=\"X\" RangeMin=\"%lf\" "
                         "RangeMax=\"%lf\">\n",
        (compute_t) shifted_indents.x / region->resolution,
        (compute_t) (shifted_indents.x + size.x) / region->resolution);

    // Write out physical positions of X co-ordinates.
    for (indexer_t h_idx = 0; h_idx <= size.x; ++h_idx)
        fprintf(destination, "%lf ", (compute_t) (h_idx + shifted_indents.x) / region->resolution);

    fprintf(destination, "\n\t\t\t\t</DataArray>\n"
                         "\t\t\t\t<DataArray type=\"Float64\" format=\"ascii\" Name=\"Y\" RangeMin=\"%lf\" "
                         "RangeMax=\"%lf\">\n",
        (compute_t) shifted_indents.y / region->resolution,
        (compute_t) (shifted_indents.y + size.y) / region->resolution);

    // Write out physical positions of Y co-ordinates.
    for (indexer_t v_idx = 0; v_idx <= size.y; ++v_idx)
        fprintf(destination, "%lf ", (compute_t) (v_idx + shifted_indents.y) / region->resolution);

    fputs(
        "\n\t\t\t\t</DataArray>\n"
        "\t\t\t\t<DataArray type=\"Float64\" format=\"ascii\" Name=\"Z\">\n"
        "0.0\n"
        "\t\t\t\t</DataArray>\n"
        "\t\t\t</Coordinates>\n"
        "\t\t\t<PointData Vectors=\"uv\">\n"
        "\t\t\t\t<DataArray type=\"Float64\" format=\"ascii\" Name=\"uv\" NumberOfComponents=\"3\">\n",

        destination);

    // Write out velocity vectors.
    for (indexer_t v_idx = region->v_exterior.begin; v_idx < region->v_exterior.end; ++v_idx)
        for (indexer_t h_idx = region->h_exterior.begin; h_idx < region->h_exterior.end; ++h_idx)
            fprintf(destination, "%lf %lf 0\n", region->velocity_x[h_idx][v_idx], region->velocity_y[h_idx][v_idx]);

    fputs(
        "\t\t\t\t</DataArray>\n"
        "\t\t\t</PointData>\n"
        "\t\t\t<CellData Scalars=\"p\">\n"
        "\t\t\t\t<DataArray type=\"Float64\" format=\"ascii\" Name=\"p\">\n",

        destination);

    // Write out pressure scalars.
    for (indexer_t v_idx = region->v_exterior.begin; v_idx < region->v_exterior.end - 1; ++v_idx) {
        for (indexer_t h_idx = region->h_exterior.begin; h_idx < region->h_exterior.end - 1; ++h_idx)
            fprintf(destination, "%lf  ", region->pressure[h_idx][v_idx]);
        fputc('\n', destination);
    }

    fputs(
        "\n\t\t\t\t</DataArray>\n"
        "\t\t\t</CellData>\n"
        "\t\t</Piece>\n"
        "\t</RectilinearGrid>\n"
        "</VTKFile>\n",

        destination);
}
