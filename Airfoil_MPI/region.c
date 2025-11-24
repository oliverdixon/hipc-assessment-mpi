//
// Created by od641 on 18/11/2025.
//

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "instance.h"
#include "region.h"

static double ** alloc_2d_double_array(const unsigned int column_count, const unsigned int row_count)
{
    double ** array = malloc(column_count * sizeof(double *));
    array[0] = calloc(column_count * row_count, sizeof(double));

    for (size_t column_idx = 1; column_idx < column_count; ++column_idx)
        array[column_idx] = &array[0][column_idx * row_count];

    return array;
}

static enum cell_flags ** alloc_2d_flags_array(const unsigned int column_count, const unsigned int row_count)
{
    enum cell_flags ** array = malloc(column_count * sizeof(enum cell_flags *));
    array[0] = calloc(column_count * row_count, sizeof(enum cell_flags));

    for (size_t column_idx = 1; column_idx < column_count; ++column_idx)
        array[column_idx] = &array[0][column_idx * row_count];

    return array;
}

static void free_2d_array(void **array)
{
    free(array[0]);
    free(array);
}

static MPI_Datatype create_row_t(const unsigned int column_count, const unsigned int row_count)
{
    assert(column_count <= INT_MAX);
    assert(row_count <= INT_MAX);
    
    MPI_Datatype raw_row_t;
    MPI_Type_vector((int) column_count, 1, (int) row_count, MPI_DOUBLE, &raw_row_t);
    MPI_Type_commit(&raw_row_t);

    MPI_Datatype row_t;
    MPI_Type_create_resized(raw_row_t, 0, sizeof(double), &row_t);
    MPI_Type_commit(&row_t);
    MPI_Type_free(&raw_row_t);

    return row_t;
}

static MPI_Datatype create_column_t(const unsigned int row_count)
{
    assert(row_count <= INT_MAX);
    
    MPI_Datatype column_t;
    MPI_Type_contiguous((int) row_count, MPI_DOUBLE, &column_t);
    MPI_Type_commit(&column_t);

    return column_t;
}

static struct iterator get_initial_v_idx_boundaries(const struct region * const region, const double problem_height,
    const float maximum_camber, const float edge_distance, const float thickness, const unsigned h_cell_idx)
{
    struct iterator boundaries = {
        .begin = 0,
        .end = 0
    };

    /*
     * Position along chord, normalised to [0, 1]. From here, 'x' is translated into the co-ordinate space of the
     * global problem, and not the region.
     */
    const double x = (float) (h_cell_idx + region->indents.x) / (float) region->resolution - 0.5f;

    if (x < 0.0 || x > 1.0)
        return boundaries;

    /*
     * The midline distance is the half-thickness from the fixed 'x' to the horizontal central line of the airfoil.
     * It is the Euclidean distance from the 'x' co-ordinate to the midline. This is NACA standard formulae.
     */
    const double x_sq = x * x;
    double midline_distance = 5.0 * thickness *
        (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x_sq + 0.2843 * x * x_sq - 0.1015 * x_sq * x_sq);

    /*
     * Compute the 'y' co-ordinate of the mean camber line, given the fixed 'x' position. This is NACA standard
     * formulae, represented as a piecewise map over 'x' in intervals [0, p] and (p, 1], where 'p' is the edge
     * distance.
     */
    const double mean_camber_line_y = x <= edge_distance ?
        maximum_camber / (edge_distance * edge_distance) * (2.0 * edge_distance * x - x_sq) :  // 0 <= x <= p
        maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) *                     // p < x <= 1
            (1.0 - 2.0 * edge_distance + 2.0 * edge_distance * x - x_sq);

    /*
     * Use standard calculus formulae to find the numerical derivative of the mean camber line 'y' co-ordinate.
     * Thickness is applied perpendicular to the mean camber line. Use standard geometric formulae to compute the 'y'
     * co-ordinates for the upper and lower camber surface lines.
     */
    const double norm = x <= edge_distance
        ? 2.0 * maximum_camber / (edge_distance * edge_distance) * (edge_distance - x)
        : 2.0 * maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * (edge_distance - x);

    midline_distance *= cos(atan(norm));

    const double upper_camber_y = mean_camber_line_y + midline_distance;
    const double lower_camber_y = mean_camber_line_y - midline_distance;

    unsigned int v_idx_boundary_start = floor((lower_camber_y + problem_height / 2.0) * region->resolution);
    unsigned int v_idx_boundary_end = ceil((upper_camber_y + problem_height / 2.0) * region->resolution);

    const unsigned int south_boundary_idx = region->indents.y + (region->v_exterior.end - region->v_exterior.begin);

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

static void write_initial_extreme_boundaries(const struct region * const region)
{
    enum cell_flags * const * const flags = region->flags;

    if (region->region_flags & REGION_NORTH_BOUNDARY)
        for (unsigned int h_cell_idx = region->h_exterior.begin; h_cell_idx < region->h_exterior.end; ++h_cell_idx)
            flags[h_cell_idx][region->v_exterior.begin] = CELL_BOUNDARY;

    if (region->region_flags & REGION_SOUTH_BOUNDARY)
        for (unsigned int h_cell_idx = region->h_exterior.begin; h_cell_idx < region->h_exterior.end; ++h_cell_idx)
            flags[h_cell_idx][region->v_exterior.end - 1] = CELL_BOUNDARY;

    if (region->region_flags & REGION_WEST_BOUNDARY)
        for (unsigned int v_cell_idx = region->v_exterior.begin; v_cell_idx < region->v_exterior.end; ++v_cell_idx)
            flags[region->h_exterior.begin][v_cell_idx] = CELL_BOUNDARY;

    if (region->region_flags & REGION_EAST_BOUNDARY)
        for (unsigned int v_cell_idx = region->v_exterior.begin; v_cell_idx < region->v_exterior.end; ++v_cell_idx)
            flags[region->h_exterior.end - 1][v_cell_idx] = CELL_BOUNDARY;
}

static char get_flag_identifier(const enum cell_flags flag)
{
    if (flag & CELL_FLUID) return ' ';
    if (flag & CELL_BOUNDARY) return 'B';

    return '?';
}

static void print_velocity_x(const struct region * const region, FILE * const destination)
{
    const unsigned int h_cell_count = region->h_exterior.end - region->h_exterior.begin - 1;

    for (unsigned int h_idx = 0; h_idx < h_cell_count; ++h_idx)
        fputs("-----------", destination);
    fputs("----------\n", destination);

    if (region->region_flags & REGION_NORTH_GHOST) {
        for (unsigned int h_idx = region->h_exterior.begin; h_idx < region->h_exterior.end; ++h_idx)
            fprintf(destination, "%10lf ", region->velocity_x[h_idx][region->v_exterior.begin - 1]);
        fputc('\n', destination);

        for (unsigned int h_idx = 0; h_idx < h_cell_count; ++h_idx)
            fputs("-----------", destination);
        fputs("----------\n", destination);
    }

    for (unsigned int v_idx = region->v_exterior.begin; v_idx < region->v_exterior.end; ++v_idx) {
        if (region->region_flags & REGION_WEST_GHOST)
            fprintf(destination, "%10lf | ", region->velocity_x[region->h_exterior.begin - 1][v_idx]);
        for (unsigned int h_idx = region->h_exterior.begin; h_idx < region->h_exterior.end; ++h_idx)
            fprintf(destination, "%10lf ", region->velocity_x[h_idx][v_idx]);
        if (region->region_flags & REGION_EAST_GHOST)
            fprintf(destination, " | %10lf", region->velocity_x[region->h_exterior.end][v_idx]);
        fputc('\n', destination);
    }

    for (unsigned int h_idx = 0; h_idx < h_cell_count; ++h_idx)
        fputs("-----------", destination);
    fputs("----------", destination);

    if (region->region_flags & REGION_SOUTH_GHOST) {
        fputc('\n', destination);
        for (unsigned int h_idx = region->h_exterior.begin; h_idx < region->h_exterior.end; ++h_idx)
            fprintf(destination, "%10lf ", region->velocity_x[h_idx][region->v_exterior.end]);
        fputc('\n', destination);
    }
}

static void halo_exchange(const struct region * const region, const struct instance * const instance)
{
    int north_id, south_id, east_id, west_id;
    MPI_Cart_shift(instance->cartesian_comm, 0, 1, &west_id, &east_id);
    MPI_Cart_shift(instance->cartesian_comm, 1, 1, &north_id, &south_id);

    MPI_Request requests[4];
    int request_idx = 0;
    double dummy_ghost;

    // North
    const double * const north_row = &region->velocity_x[0][region->v_exterior.begin];
    double * const north_ghost = north_id == MPI_PROC_NULL ? &dummy_ghost :
        &region->velocity_x[0][region->v_exterior.begin - 1];

    MPI_Isendrecv(
        north_row, 1, region->row_t, north_id, 2,
        north_ghost, 1, region->row_t, north_id, 3,
        instance->cartesian_comm, &requests[request_idx++]);

    // South
    const double * const south_row = &region->velocity_x[0][region->v_exterior.end - 1];
    double * const south_ghost = south_id == MPI_PROC_NULL ? &dummy_ghost :
        &region->velocity_x[0][region->v_exterior.end];

    MPI_Isendrecv(
        south_row, 1, region->row_t, south_id, 3,
        south_ghost, 1, region->row_t, south_id, 2,
        instance->cartesian_comm, &requests[request_idx++]);

    // East
    const double * const east_column = region->velocity_x[region->h_exterior.end - 1];
    double * const east_ghost = east_id == MPI_PROC_NULL ? &dummy_ghost :
        region->velocity_x[region->h_exterior.end];

    MPI_Isendrecv(
        east_column, 1, region->col_t, east_id, 0,
        east_ghost, 1, region->col_t, east_id, 1,
        instance->cartesian_comm, &requests[request_idx++]
    );

    // West
    const double * const west_column = region->velocity_x[region->h_exterior.begin];
    double * const west_ghost = west_id == MPI_PROC_NULL ? &dummy_ghost :
        region->velocity_x[region->h_exterior.begin - 1];

    MPI_Isendrecv(
        west_column, 1, region->col_t, west_id, 1,
        west_ghost, 1, region->col_t, west_id, 0,
        instance->cartesian_comm, &requests[request_idx++]
    );

    MPI_Waitall(request_idx, requests, MPI_STATUSES_IGNORE);
}

struct region region_create(const struct instance * const instance)
{
    static const unsigned int resolution = 128; // Number of cells per unit-distance.

    const struct dim2 global_cell_counts = {
        .x = (unsigned int) ceilf((float) resolution * instance->problem_width),
        .y = (unsigned int) ceilf((float) resolution * instance->problem_height)
    };

    struct dim2 cell_counts = {
        .x = global_cell_counts.x / instance->x_dim_extent,
        .y = global_cell_counts.y / instance->y_dim_extent
    };

    struct dim2 allocations = {
        .x = cell_counts.x,
        .y = cell_counts.y
    };

    enum region_flags region_flags = REGION_UNREMARKABLE;
    bool x_ghosts_finalised = false;
    bool y_ghosts_finalised = false;

    if (instance->x_dim_extent > 1) {
        if (instance->x_position == 0) {
            /*
             * Westernmost region with at least one region to the east; marks a west boundary, definitely requires an
             * east ghost.
             */
            region_flags |= REGION_WEST_BOUNDARY | REGION_EAST_GHOST;
            ++allocations.x;
            x_ghosts_finalised = true;
        }

        if (instance->x_position == instance->x_dim_extent - 1) {
            /*
             * Easternmost region with at least one region to the west; marks an east boundary, definitely requires a
             * west ghost. As the last region, also pick up any slack on the X axis.
             */
            region_flags |= REGION_EAST_BOUNDARY | REGION_WEST_GHOST;
            cell_counts.x += global_cell_counts.x % instance->x_dim_extent;
            ++allocations.x;
            x_ghosts_finalised = true;
        }
    } else {
        // Single region on the X. Marks west and east boundaries; thus doesn't require east or west ghosts.
        region_flags |= REGION_WEST_BOUNDARY | REGION_EAST_BOUNDARY;
        x_ghosts_finalised = true;
    }

    if (instance->y_dim_extent > 1) {
        if (instance->y_position == 0) {
            /*
             * Northernmost region with at least one region to the south; marks a north boundary, definitely requires a
             * south ghost.
             */
            region_flags |= REGION_NORTH_BOUNDARY | REGION_SOUTH_GHOST;
            ++allocations.y;
            y_ghosts_finalised = true;
        }

        if (instance->y_position == instance->y_dim_extent - 1) {
            /*
             * Southernmost region with at least one region to the north; marks a south boundary, definitely requires a
             * north ghost.
             */
            region_flags |= REGION_SOUTH_BOUNDARY | REGION_NORTH_GHOST;
            cell_counts.y += global_cell_counts.y % instance->y_dim_extent;
            ++allocations.y;
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
        allocations.x += 2;
    }

    if (!y_ghosts_finalised) {
        // Multiple regions on the Y, and we're in the middle.
        region_flags |= REGION_NORTH_GHOST | REGION_SOUTH_GHOST;
        allocations.y += 2;
    }

    const struct iterator h_exterior = {
        .begin = !!(region_flags & REGION_WEST_GHOST),
        .end = cell_counts.x + !!(region_flags & REGION_WEST_GHOST)
    };

    const struct iterator v_exterior = {
        .begin = !!(region_flags & REGION_NORTH_GHOST),
        .end = cell_counts.y + !!(region_flags & REGION_NORTH_GHOST)
    };

    const struct region region = {
        .velocity_x = alloc_2d_double_array(allocations.x, allocations.y),
        .velocity_y = alloc_2d_double_array(allocations.x, allocations.y),
        .tentative_velocity_x = alloc_2d_double_array(allocations.x, allocations.y),
        .tentative_velocity_y = alloc_2d_double_array(allocations.x, allocations.y),
        .pressure = alloc_2d_double_array(allocations.x, allocations.y),
        .flags = alloc_2d_flags_array(allocations.x, allocations.y),

        .region_flags = region_flags,

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
        .indents = instance_get_indentations(instance, cell_counts),

        .initial_velocity_x = 1.0,
        .initial_velocity_y = 0.0,
        .initial_pressure = 0.0,
        .initial_flag = CELL_FLUID,

        .row_t = create_row_t(allocations.x, allocations.y),
        .col_t = create_column_t(allocations.y),
    };
    
    return region;
}

void region_destroy(struct region *const region)
{
    MPI_Type_free(&region->col_t);
    MPI_Type_free(&region->row_t);

    free_2d_array((void **) region->velocity_x);
    free_2d_array((void **) region->velocity_y);
    free_2d_array((void **) region->tentative_velocity_x);
    free_2d_array((void **) region->tentative_velocity_y);
    free_2d_array((void **) region->pressure);
    free_2d_array((void **) region->flags);
}

void region_describe(const struct region * const region, FILE * const destination)
{
    fprintf(destination, "Region statistics:\n\t"
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

void region_print_flags(const struct region *const region, FILE *const destination)
{
    for (unsigned int v_cell_idx = region->v_exterior.begin; v_cell_idx < region->v_exterior.end; ++v_cell_idx) {
        for (unsigned int h_cell_idx = region->h_exterior.begin; h_cell_idx < region->h_exterior.end; ++h_cell_idx) {
            fputc(get_flag_identifier(region->flags[h_cell_idx][v_cell_idx]), destination);
            fputc(' ', destination);
        }

        fputc('\n', destination);
    }
}

void region_initialise(const struct region * const region, const struct instance * const instance)
{
    // Transform the NACA digits into the scale expected by the initial boundary calculi.
    const float maximum_camber = (float) instance->naca_specifier.maximum_camber / 100.0f;
    const float edge_distance = (float) instance->naca_specifier.edge_distance / 10.0f;
    const float thickness = (float) instance->naca_specifier.maximum_thickness / 100.0f;

    for (unsigned int h_cell_idx = region->h_interior.begin; h_cell_idx < region->h_interior.end; ++h_cell_idx) {
        // Populate all cells' information matrices with fixed initial values.
        for (unsigned int v_cell_idx = region->v_interior.begin; v_cell_idx < region->v_interior.end; ++v_cell_idx) {
            // region->velocity_x[h_cell_idx][v_cell_idx] = region->initial_velocity_x; // TODO testing
            region->velocity_x[h_cell_idx][v_cell_idx] = h_cell_idx + 1 + v_cell_idx + 1;
            region->velocity_y[h_cell_idx][v_cell_idx] = region->initial_velocity_y;
            region->pressure[h_cell_idx][v_cell_idx] = region->initial_pressure;
            region->flags[h_cell_idx][v_cell_idx] = region->initial_flag;
        }

        // Compute the vertical index boundaries of the airfoil body at the fixed horizontal index.
        const struct iterator v_idx_boundaries = get_initial_v_idx_boundaries(region, instance->problem_height,
            maximum_camber, edge_distance, thickness, h_cell_idx);

        // Populate the airfoil body with boundary markers.
        for (unsigned int v_cell_idx = v_idx_boundaries.begin; v_cell_idx < v_idx_boundaries.end; ++v_cell_idx)
            region->flags[h_cell_idx][v_cell_idx] = CELL_BOUNDARY;
    }

    write_initial_extreme_boundaries(region);
}
