#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"

/**
 * @brief Allocate all of the arrays used by the computation.
 *
 */
void allocate_arrays()
{
    velocity_x = alloc_2d_array(h_cell_count + 2, v_cell_count + 2);
    velocity_y = alloc_2d_array(h_cell_count + 2, v_cell_count + 2);
    tentative_velocity_x = alloc_2d_array(h_cell_count + 2, v_cell_count + 2);
    tentative_velocity_y = alloc_2d_array(h_cell_count + 2, v_cell_count + 2);
    pressure = alloc_2d_array(h_cell_count + 2, v_cell_count + 2);
    poisson_source = alloc_2d_array(h_cell_count + 2, v_cell_count + 2);
    flags = alloc_2d_char_array(h_cell_count + 2, v_cell_count + 2);

    if (!velocity_x || !velocity_y || !tentative_velocity_x || !tentative_velocity_y || !pressure || !poisson_source || !flags) {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
        exit(1);
    }
}

/**
 * @brief Free all of the arrays used for the computation.
 *
 */
void free_arrays()
{
    free_2d_array((void **) velocity_x);
    free_2d_array((void **) velocity_y);
    free_2d_array((void **) tentative_velocity_x);
    free_2d_array((void **) tentative_velocity_y);
    free_2d_array((void **) pressure);
    free_2d_array((void **) poisson_source);
    free_2d_array((void **) flags);
}

/**
 * @brief Initialise the velocity arrays and then initialize the flag array,
 * marking any obstacle cells and the edge cells as boundaries. The cells
 * adjacent to boundary cells have their relevant flags set too.
 */
void problem_set_up()
{
    /*
     * Initialise all cells with defined initial parameters:
     *
     * 1. Initial
     */

    for (int h_cell_idx = 0; h_cell_idx < h_cell_count + 2; ++h_cell_idx) {
        for (int v_cell_idx = 0; v_cell_idx < v_cell_count + 2; ++v_cell_idx) {
            velocity_x[h_cell_idx][v_cell_idx] = ui;
            velocity_y[h_cell_idx][v_cell_idx] = vi;
            pressure[h_cell_idx][v_cell_idx] = 0.0;
            flags[h_cell_idx][v_cell_idx] = CELL_FLUID;
        }
    }

    /*
     * Mark the airfoil obstacle outline as boundary cells, and the rest as fluid.
     *
     * 1. Scale the NACA airfoil parameters by constants. This function supports cambered airfoils.
     * 2. Scanning left-to-right along the horizontal axis, compute the camber lines for the upper and lower portions of
     *    the airfoil.
     * 3. Mark airfoil outline boundaries in the 'flags' matrix.
     */

    const double maximum_camber = naca_specifier.maximum_camber / 100.0;
    const double edge_distance = naca_specifier.edge_distance / 10.0;
    const double thickness = naca_specifier.maximum_thickness / 100.0;

    for (int h_cell_idx = 1; h_cell_idx <= h_cell_count; ++h_cell_idx) {
        /*
         * Normalise the current cell index to provide the position along the chord. If it lies beyond [0, 1], it is
         * outside of the problem space and we're not interested.
         */
        const double x = problem_space_width / h_cell_count * h_cell_idx - 0.5;
        if (x < 0.0 || x > 1.0)
            continue;

        /*
         * The midline distance is the half-thickness from the fixed 'x' to the horizontal central line of the airfoil.
         * It is the Euclidean distance from the 'x' co-ordinate to the midline. This is NACA standard formulae.
         */
        const double midline_distance = 5.0 * thickness *
                (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x);

        /*
         * Compute the 'y' co-ordinate of the mean camber line, given the fixed 'x' position. This is NACA standard
         * formulae, represented as a piecewise map over 'x' in intervals [0, p] and (p, 1], where 'p' is the edge
         * distance.
         */
        const double mean_camber_line_y =
            x <= edge_distance ?
                maximum_camber / (edge_distance * edge_distance) * (2.0 * edge_distance * x - x * x) :  // 0 <= x <= p
                maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) *                      // p < x <= 1
                    (1.0 - 2.0 * edge_distance + 2.0 * edge_distance * x - x * x);

        // Use standard calculus formulae to find the numerical derivative of the mean camber line 'y' co-ordinate.
        const double dyc_dx =
            x <= edge_distance
                ? 2.0 * maximum_camber / (edge_distance * edge_distance) * (edge_distance - x)
                : 2.0 * maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * (edge_distance - x);

        /*
         * Thickness is applied perpendicular to the mean camber line. Use standard geometric formulae to compute the
         * 'y' co-ordinates for the upper and lower camber surface lines.
         */
        const double perpendicular_angle_cos = cos(atan(dyc_dx));
        const double upper_camber_y = mean_camber_line_y + midline_distance * perpendicular_angle_cos;
        const double lower_camber_y = mean_camber_line_y - midline_distance * perpendicular_angle_cos;

        /*
         * Fixed on the 'x' position, scan vertically along the interval indicated by the upper and lower camber lines
         * to mark boundary cells.
         */
        const double vertical_scaler = v_cell_count / problem_space_height;
        const unsigned int v_idx_start = (unsigned int) floor((lower_camber_y + problem_space_height / 2.0) *
            vertical_scaler);
        const unsigned int v_idx_end = (unsigned int) ceil((upper_camber_y + problem_space_height / 2.0) *
            vertical_scaler);

        for (unsigned int v_cell_idx = v_idx_start; v_cell_idx < v_idx_end; v_cell_idx++)
            flags[h_cell_idx][v_cell_idx] = CELL_BOUNDARY;
    }

    // Mark the extreme north and south boundary cells
    for (int h_cell_idx = 0; h_cell_idx <= h_cell_count + 1; ++h_cell_idx) {
        flags[h_cell_idx][0] = CELL_BOUNDARY;
        flags[h_cell_idx][v_cell_count + 1] = CELL_BOUNDARY;
    }

    // Mark the extreme east and west boundary cells
    for (int v_celL_idx = 1; v_celL_idx <= v_cell_count; ++v_celL_idx) {
        flags[0][v_celL_idx] = CELL_BOUNDARY;
        flags[h_cell_count + 1][v_celL_idx] = CELL_BOUNDARY;
    }

    fluid_cell_count = h_cell_count * v_cell_count;

    // Mask in additional directional indicator flags for non-fluid cells, describing presence of nearby fluid cells.
    for (int h_cell_idx = 1; h_cell_idx <= h_cell_count; ++h_cell_idx)
        for (int v_cell_idx = 1; v_cell_idx <= v_cell_count; ++v_cell_idx)
            if (!(flags[h_cell_idx][v_cell_idx] & CELL_FLUID)) {
                --fluid_cell_count;
                if (flags[h_cell_idx - 1][v_cell_idx] & CELL_FLUID)
                    flags[h_cell_idx][v_cell_idx] |= CELL_FLUID_WEST;
                if (flags[h_cell_idx + 1][v_cell_idx] & CELL_FLUID)
                    flags[h_cell_idx][v_cell_idx] |= CELL_FLUID_EAST;
                if (flags[h_cell_idx][v_cell_idx - 1] & CELL_FLUID)
                    flags[h_cell_idx][v_cell_idx] |= CELL_FLUID_SOUTH;
                if (flags[h_cell_idx][v_cell_idx + 1] & CELL_FLUID)
                    flags[h_cell_idx][v_cell_idx] |= CELL_FLUID_NORTH;
            }
}
