//
// Created by od641 on 18/11/2025.
//

#include <math.h>
#include <stdlib.h>

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
    const double x = (float) (h_cell_idx + region->x_indent) / (float) region->resolution - 0.5f;

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

    const unsigned int south_boundary_idx = region->y_indent + (region->v_exterior.end - region->v_exterior.begin);

    if (v_idx_boundary_start < region->y_indent)
        // If the start is a pixel in a region to the north of us, bound it at our northernmost point.
        v_idx_boundary_start = region->v_exterior.begin;
    else if (v_idx_boundary_start > south_boundary_idx)
        // If the start is a pixel in a region to the south of us, this region doesn't contain any of the points.
        return boundaries;
    else
        // Otherwise, the start-point is somewhere within our region. Translate the index to zero-based from the north.
        v_idx_boundary_start -= region->y_indent;

    if (v_idx_boundary_end >= south_boundary_idx)
        // If the end is a pixel in a region to the south of us, bound it at our southernmost point.
        v_idx_boundary_end = region->v_exterior.end;
    else if (v_idx_boundary_end < region->y_indent)
        // If the end is a pixel in a region to the north of us, this region doesn't contain any of the points.
        return boundaries;
    else
        // Otherwise, the end-point is somewhere within our region. Translate the index to zero-based from the north.
        v_idx_boundary_end -= region->y_indent;

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

struct region region_create(const struct instance * const instance)
{
    static const unsigned int resolution = 128; // Number of cells per unit-distance.

    const unsigned int global_h_cell_count = (unsigned int) ceilf((float) resolution * instance->problem_width);
    const unsigned int global_v_cell_count = (unsigned int) ceilf((float) resolution * instance->problem_height);

    unsigned int h_cell_count = global_h_cell_count / instance->x_dim_extent;
    unsigned int v_cell_count = global_v_cell_count / instance->y_dim_extent;

    enum region_flags region_flags = REGION_UNREMARKABLE;

    if (instance->x_position == instance->x_dim_extent - 1) {
        region_flags |= REGION_EAST_BOUNDARY;
        h_cell_count += global_h_cell_count % instance->x_dim_extent;
    }

    if (instance->y_position == instance->y_dim_extent - 1) {
        region_flags |= REGION_SOUTH_BOUNDARY;
        v_cell_count += global_v_cell_count % instance->y_dim_extent;
    }

    if (instance->x_position == 0)
        region_flags |= REGION_WEST_BOUNDARY;

    if (instance->y_position == 0)
        region_flags |= REGION_NORTH_BOUNDARY;

    const struct iterator h_exterior = {
        .begin = 0,
        .end = h_cell_count
    };

    const struct iterator v_exterior = {
        .begin = 0,
        .end = v_cell_count
    };

    /*
     * Regions need to know their positions relative to the global problem space, such that co-ordinate transforms can
     * be done before and after running the Navier-Stokes algorithms. An indent value (number of cells from (0, 0) in
     * the global problem space) is sufficient, and can be collected by scanning over the horizontal and vertical axes
     * for the 'X' and 'Y' indent values, respectively.
     *
     * Axes can be fixed by subsetting communications on their respective fixed dimensions, and calling MPI_Scan on the
     * fixed-dimensional communicators. Note that the sub-Cartesian communicator produced by MPI_Cart_sub orders its
     * constituent ranks, such that MPI_Scan will visit regions in the expected order.
     */
    unsigned int x_indent;
    unsigned int y_indent;
    const unsigned int local_x_indent = instance->x_position == 0 ? 0 : h_cell_count;
    const unsigned int local_y_indent = instance->y_position == 0 ? 0 : v_cell_count;

    int fix_dimensions[] = { 1, 0 };
    MPI_Comm fixed_dim_comm;
    MPI_Cart_sub(instance->cartesian_comm, fix_dimensions, &fixed_dim_comm); // Fixed on X; row communicator.
    MPI_Scan(&local_x_indent, &x_indent, 1, MPI_UNSIGNED, MPI_SUM, fixed_dim_comm);

    fix_dimensions[0] = 0;
    fix_dimensions[1] = 1;
    MPI_Cart_sub(instance->cartesian_comm, fix_dimensions, &fixed_dim_comm); // Fixed on Y; column communicator.
    MPI_Scan(&local_y_indent, &y_indent, 1, MPI_UNSIGNED, MPI_SUM, fixed_dim_comm);

    const struct region region = {
        .velocity_x = alloc_2d_double_array(h_cell_count, v_cell_count),
        .velocity_y = alloc_2d_double_array(h_cell_count, v_cell_count),
        .tentative_velocity_x = alloc_2d_double_array(h_cell_count, v_cell_count),
        .tentative_velocity_y = alloc_2d_double_array(h_cell_count, v_cell_count),
        .pressure = alloc_2d_double_array(h_cell_count, v_cell_count),
        .flags = alloc_2d_flags_array(h_cell_count, v_cell_count),

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
        .x_indent = x_indent,
        .y_indent = y_indent,

        .initial_velocity_x = 1.0,
        .initial_velocity_y = 0.0,
        .initial_pressure = 0.0,
        .initial_flag = CELL_FLUID
    };

    return region;
}

void region_destroy(const struct region *const region)
{
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
                         region->h_interior.begin, region->h_interior.end - 1,
                         region->v_interior.begin, region->v_interior.end - 1,
                         region->h_exterior.begin, region->h_exterior.end - 1,
                         region->v_exterior.begin, region->v_exterior.end - 1,
                         region->x_indent,
                         region->y_indent);
}

void region_print(const struct region *const region, FILE *const destination)
{
    for (unsigned int v_cell_idx = region->v_exterior.begin; v_cell_idx < region->v_exterior.end; ++v_cell_idx) {
        for (unsigned int h_cell_idx = region->h_exterior.begin; h_cell_idx < region->h_exterior.end; ++h_cell_idx) {
            char identifier;

            switch (region->flags[h_cell_idx][v_cell_idx]) {
            case CELL_FLUID:
                identifier = ' ';
                break;
            case CELL_BOUNDARY:
                identifier = 'B';
                break;
            default:
                identifier = '?';
            }

            fprintf(destination, "%c ", identifier);
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
            region->velocity_x[h_cell_idx][v_cell_idx] = region->initial_velocity_x;
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
