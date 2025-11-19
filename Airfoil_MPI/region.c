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

struct region region_create(const struct instance * const instance)
{
    static const unsigned int resolution = 128; // Number of cells per unit-distance.

    const unsigned int global_h_cell_count = (unsigned int) ceilf((float) resolution * instance->global_problem_width);
    const unsigned int global_v_cell_count = (unsigned int) ceilf((float) resolution * instance->global_problem_height);

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

    unsigned int x_indent;
    unsigned int y_indent;
    const unsigned int local_x_indent = instance->x_position == 0 ? 0 : h_cell_count;
    const unsigned int local_y_indent = instance->y_position == 0 ? 0 : v_cell_count;

    int fix_dimensions[] = { 1, 0 };
    MPI_Comm fixed_dim_comm;
    MPI_Cart_sub(instance->cartesian_comm, fix_dimensions, &fixed_dim_comm);
    MPI_Scan(&local_x_indent, &x_indent, 1, MPI_UNSIGNED, MPI_SUM, fixed_dim_comm);

    fix_dimensions[0] = 0;
    fix_dimensions[1] = 1;
    MPI_Cart_sub(instance->cartesian_comm, fix_dimensions, &fixed_dim_comm);
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
    double * const * velocity_x = region->velocity_x;
    double * const * velocity_y = region->velocity_y;
    double * const * pressure = region->pressure;
    enum cell_flags * const * flags = region->flags;

    const float resolution_reciprocal = 1.0f / (float) region->resolution;

    const float maximum_camber = (float) instance->naca_specifier.maximum_camber / 100.0f;
    const float edge_distance = (float) instance->naca_specifier.edge_distance / 10.0f;
    const float thickness = (float) instance->naca_specifier.maximum_thickness / 100.0f;

    for (unsigned int h_cell_idx = region->h_interior.begin; h_cell_idx < region->h_interior.end; ++h_cell_idx) {
        for (unsigned int v_cell_idx = region->v_interior.begin; v_cell_idx < region->v_interior.end; ++v_cell_idx) {
            velocity_x[h_cell_idx][v_cell_idx] = region->initial_velocity_x;
            velocity_y[h_cell_idx][v_cell_idx] = region->initial_velocity_y;
            pressure[h_cell_idx][v_cell_idx] = region->initial_pressure;
            flags[h_cell_idx][v_cell_idx] = region->initial_flag;
        }

        // Position along chord, normalised to [0, 1].
        const double x = resolution_reciprocal * (float) (h_cell_idx + region->x_indent) - 0.5f;

        if (x >= 0.0 && x <= 1.0) {
            const double midline_distance = 5.0 * thickness *
                (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x);

            const double mean_camber_line_y = x <= edge_distance ?
                maximum_camber / (edge_distance * edge_distance) * (2.0 * edge_distance * x - x * x) :  // 0 <= x <= p
                maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) *                      // p < x <= 1
                    (1.0 - 2.0 * edge_distance + 2.0 * edge_distance * x - x * x);

            const double norm = x <= edge_distance
                ? 2.0 * maximum_camber / (edge_distance * edge_distance) * (edge_distance - x)
                : 2.0 * maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * (edge_distance - x);

            const double perpendicular_angle = cos(atan(norm));
            const double upper_camber_y = mean_camber_line_y + midline_distance * perpendicular_angle;
            const double lower_camber_y = mean_camber_line_y - midline_distance * perpendicular_angle;

            const unsigned int v_idx_boundary_start = floor((lower_camber_y + instance->global_problem_height / 2.0) *
                region->resolution);
            const unsigned int v_idx_boundary_end = ceil((upper_camber_y + instance->global_problem_height / 2.0) *
                region->resolution);

            for (unsigned int v_cell_idx = v_idx_boundary_start; v_cell_idx < v_idx_boundary_end; ++v_cell_idx)
                flags[h_cell_idx][v_cell_idx] = CELL_BOUNDARY;
        }
    }

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
