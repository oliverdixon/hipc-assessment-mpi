#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "boundary.h"
#include "data.h"
#include "vtk.h"

/**
 * @brief Set up some default values before arguments are parsed.
 *
 */
void set_defaults()
{
    set_default_base();
}

/**
 * @brief Set up some values after arguments have been parsed.
 *
 */
void setup()
{
    delx = problem_space_width / h_cell_count;
    dely = problem_space_height / v_cell_count;
}

/**
 * @brief Allocate all of the arrays used by the computation.
 *
 */
void allocate_arrays()
{
    /* Allocate arrays */
    u_size_x = h_cell_count + 2;
    u_size_y = v_cell_count + 2;
    u = alloc_2d_array(u_size_x, u_size_y);
    v_size_x = h_cell_count + 2;
    v_size_y = v_cell_count + 2;
    v = alloc_2d_array(v_size_x, v_size_y);
    f_size_x = h_cell_count + 2;
    f_size_y = v_cell_count + 2;
    f = alloc_2d_array(f_size_x, f_size_y);
    g_size_x = h_cell_count + 2;
    g_size_y = v_cell_count + 2;
    g = alloc_2d_array(g_size_x, g_size_y);
    p_size_x = h_cell_count + 2;
    p_size_y = v_cell_count + 2;
    p = alloc_2d_array(p_size_x, p_size_y);
    rhs_size_x = h_cell_count + 2;
    rhs_size_y = v_cell_count + 2;
    rhs = alloc_2d_array(rhs_size_x, rhs_size_y);
    flag_size_x = h_cell_count + 2;
    flag_size_y = v_cell_count + 2;
    flag = alloc_2d_char_array(flag_size_x, flag_size_y);

    if (!u || !v || !f || !g || !p || !rhs || !flag) {
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
    free_2d_array((void **) u);
    free_2d_array((void **) v);
    free_2d_array((void **) f);
    free_2d_array((void **) g);
    free_2d_array((void **) p);
    free_2d_array((void **) rhs);
    free_2d_array((void **) flag);
}

/**
 * @brief Initialise the velocity arrays and then initialize the flag array,
 * marking any obstacle cells and the edge cells as boundaries. The cells
 * adjacent to boundary cells have their relevant flags set too.
 */
void problem_set_up()
{
    for (int i = 0; i < h_cell_count + 2; i++) {
        for (int j = 0; j < v_cell_count + 2; j++) {
            u[i][j] = ui;
            v[i][j] = vi;
            p[i][j] = 0.0;
            flag[i][j] = CELL_FLUID;
        }
    }

    /*
     * Mark the airfoil obstacle outline as boundary cells, and the rest as fluid.
     *
     * 1. Scale the NACA airfoil parameters by constants. This function supports cambered airfoils.
     * 2. Scanning left-to-right along the horizontal axis,
     */

    const double maximum_camber = naca_specifier.maximum_camber / 100.0;
    const double edge_distance = naca_specifier.edge_distance / 10.0;
    const double thickness = naca_specifier.maximum_thickness / 100.0;

    for (int h_cell_idx = 1; h_cell_idx <= h_cell_count; h_cell_idx++) {
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
            flag[h_cell_idx][v_cell_idx] = CELL_BOUNDARY;
    }

    // Mark the extreme north and south boundary cells
    for (int i = 0; i <= h_cell_count + 1; i++) {
        flag[i][0] = CELL_BOUNDARY;
        flag[i][v_cell_count + 1] = CELL_BOUNDARY;
    }

    // Mark the extreme east and west boundary cells
    for (int j = 1; j <= v_cell_count; j++) {
        flag[0][j] = CELL_BOUNDARY;
        flag[h_cell_count + 1][j] = CELL_BOUNDARY;
    }

    fluid_cells = h_cell_count * v_cell_count;

    // Mask in additional directional indicator flags for non-fluid cells, describing presence of nearby fluid cells.
    for (int i = 1; i <= h_cell_count; i++)
        for (int j = 1; j <= v_cell_count; j++)
            if (!(flag[i][j] & CELL_FLUID)) {
                --fluid_cells;
                if (flag[i - 1][j] & CELL_FLUID)
                    flag[i][j] |= CELL_FLUID_WEST;
                if (flag[i + 1][j] & CELL_FLUID)
                    flag[i][j] |= CELL_FLUID_EAST;
                if (flag[i][j - 1] & CELL_FLUID)
                    flag[i][j] |= CELL_FLUID_SOUTH;
                if (flag[i][j + 1] & CELL_FLUID)
                    flag[i][j] |= CELL_FLUID_NORTH;
            }

    apply_boundary_conditions();
}
