#include <math.h>
#include <stdio.h>

#include "args.h"
#include "data.h"
#include "setup.h"
#include "vtk.h"

/**
 * @brief Given the boundary conditions defined by the flag matrix, update
 * the velocity matrices. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
static void apply_boundary_conditions()
{
    for (int v_cell_idx = 0; v_cell_idx < v_cell_count + 2; ++v_cell_idx) {
        /* Fluid freely flows in from the west */
        velocity_x[0][v_cell_idx] = velocity_x[1][v_cell_idx];
        velocity_y[0][v_cell_idx] = velocity_y[1][v_cell_idx];

        /* Fluid freely flows out to the east */
        velocity_x[h_cell_count][v_cell_idx] = velocity_x[h_cell_count - 1][v_cell_idx];
        velocity_y[h_cell_count + 1][v_cell_idx] = velocity_y[h_cell_count][v_cell_idx];
    }

    for (int h_cell_idx = 0; h_cell_idx < h_cell_count + 2; ++h_cell_idx) {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        velocity_y[h_cell_idx][v_cell_count] = 0.0;
        velocity_x[h_cell_idx][v_cell_count + 1] = velocity_x[h_cell_idx][v_cell_count];

        velocity_y[h_cell_idx][0] = 0.0;
        velocity_x[h_cell_idx][0] = velocity_x[h_cell_idx][1];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the velocities to tend towards zero in these cells.
     */
    for (int i = 1; i < h_cell_count + 1; i++) {
        for (int j = 1; j < v_cell_count + 1; j++) {
            if (flags[i][j] & CELL_FLUID_ALL) {
                switch (flags[i][j]) {
                case CELL_FLUID_NORTH:
                    velocity_y[i][j] = 0.0;
                    velocity_x[i][j] = -velocity_x[i][j + 1];
                    velocity_x[i - 1][j] = -velocity_x[i - 1][j + 1];
                    break;
                case CELL_FLUID_EAST:
                    velocity_x[i][j] = 0.0;
                    velocity_y[i][j] = -velocity_y[i + 1][j];
                    velocity_y[i][j - 1] = -velocity_y[i + 1][j - 1];
                    break;
                case CELL_FLUID_SOUTH:
                    velocity_y[i][j - 1] = 0.0;
                    velocity_x[i][j] = -velocity_x[i][j - 1];
                    velocity_x[i - 1][j] = -velocity_x[i - 1][j - 1];
                    break;
                case CELL_FLUID_WEST:
                    velocity_x[i - 1][j] = 0.0;
                    velocity_y[i][j] = -velocity_y[i - 1][j];
                    velocity_y[i][j - 1] = -velocity_y[i - 1][j - 1];
                    break;
                case CELL_FLUID_NORTHEAST:
                    velocity_y[i][j] = 0.0;
                    velocity_x[i][j] = 0.0;
                    velocity_y[i][j - 1] = -velocity_y[i + 1][j - 1];
                    velocity_x[i - 1][j] = -velocity_x[i - 1][j + 1];
                    break;
                case CELL_FLUID_SOUTHEAST:
                    velocity_y[i][j - 1] = 0.0;
                    velocity_x[i][j] = 0.0;
                    velocity_y[i][j] = -velocity_y[i + 1][j];
                    velocity_x[i - 1][j] = -velocity_x[i - 1][j - 1];
                    break;
                case CELL_FLUID_SOUTHWEST:
                    velocity_y[i][j - 1] = 0.0;
                    velocity_x[i - 1][j] = 0.0;
                    velocity_y[i][j] = -velocity_y[i - 1][j];
                    velocity_x[i][j] = -velocity_x[i][j - 1];
                    break;
                case CELL_FLUID_NORTHWEST:
                    velocity_y[i][j] = 0.0;
                    velocity_x[i - 1][j] = 0.0;
                    velocity_y[i][j - 1] = -velocity_y[i - 1][j - 1];
                    velocity_x[i][j] = -velocity_x[i][j + 1];
                    break;
                default: ;
                }
            }
        }
    }

    // Finally, fix the horizontal velocity at the western edge to have a continual flow of fluid into the simulation.
    velocity_y[0][0] = 2 * initial_y_vel - velocity_y[1][0];
    for (int v_cell_idx = 1; v_cell_idx < v_cell_count + 1; ++v_cell_idx) {
        velocity_x[0][v_cell_idx] = initial_x_vel;
        velocity_y[0][v_cell_idx] = 2 * initial_y_vel - velocity_y[1][v_cell_idx];
    }
}

/**
 * @brief Computation of tentative velocity field.
 */
static void compute_tentative_velocities()
{
    static const double gamma = 0.9; /* Gamma, Upwind differencing factor in PDE discretisation */

    for (int h_cell_idx = 1; h_cell_idx < h_cell_count; ++h_cell_idx)
        for (int v_cell_idx = 1; v_cell_idx < v_cell_count + 1; ++v_cell_idx) {
            if (flags[h_cell_idx][v_cell_idx] & CELL_FLUID && flags[h_cell_idx + 1][v_cell_idx] & CELL_FLUID) {

                const double self_advection_x =
                    (
                        (velocity_x[h_cell_idx][v_cell_idx] + velocity_x[h_cell_idx + 1][v_cell_idx]) *
                        (velocity_x[h_cell_idx][v_cell_idx] + velocity_x[h_cell_idx + 1][v_cell_idx]) +
                        gamma * fabs(velocity_x[h_cell_idx][v_cell_idx] + velocity_x[h_cell_idx + 1][v_cell_idx]) *
                        (velocity_x[h_cell_idx][v_cell_idx] - velocity_x[h_cell_idx + 1][v_cell_idx]) -
                        (velocity_x[h_cell_idx - 1][v_cell_idx] + velocity_x[h_cell_idx][v_cell_idx]) *
                        (velocity_x[h_cell_idx - 1][v_cell_idx] + velocity_x[h_cell_idx][v_cell_idx]) -
                        gamma * fabs(velocity_x[h_cell_idx - 1][v_cell_idx] + velocity_x[h_cell_idx][v_cell_idx]) *
                        (velocity_x[h_cell_idx - 1][v_cell_idx] - velocity_x[h_cell_idx][v_cell_idx])
                    ) /
                        (4.0 * x_grid_spacing);

                const double cross_advection_y =
                    (
                        (velocity_y[h_cell_idx][v_cell_idx] + velocity_y[h_cell_idx + 1][v_cell_idx]) *
                        (velocity_x[h_cell_idx][v_cell_idx] + velocity_x[h_cell_idx][v_cell_idx + 1]) +
                        gamma * fabs(velocity_y[h_cell_idx][v_cell_idx] + velocity_y[h_cell_idx + 1][v_cell_idx]) *
                        (velocity_x[h_cell_idx][v_cell_idx] - velocity_x[h_cell_idx][v_cell_idx + 1]) -
                        (velocity_y[h_cell_idx][v_cell_idx - 1] + velocity_y[h_cell_idx + 1][v_cell_idx - 1]) *
                        (velocity_x[h_cell_idx][v_cell_idx - 1] + velocity_x[h_cell_idx][v_cell_idx]) -
                        gamma * fabs(velocity_y[h_cell_idx][v_cell_idx - 1] +
                            velocity_y[h_cell_idx + 1][v_cell_idx - 1]) *
                        (velocity_x[h_cell_idx][v_cell_idx - 1] - velocity_x[h_cell_idx][v_cell_idx])
                    ) /
                        (4.0 * y_grid_spacing);

                const double diffusion =
                    (
                        velocity_x[h_cell_idx + 1][v_cell_idx] -
                        2.0 * velocity_x[h_cell_idx][v_cell_idx] +
                        velocity_x[h_cell_idx - 1][v_cell_idx]
                    ) /
                        (x_grid_spacing * x_grid_spacing) +
                    (
                        velocity_x[h_cell_idx][v_cell_idx + 1] -
                        2.0 * velocity_x[h_cell_idx][v_cell_idx] +
                        velocity_x[h_cell_idx][v_cell_idx - 1]
                    ) /
                        (y_grid_spacing * y_grid_spacing);

                tentative_velocity_x[h_cell_idx][v_cell_idx] = velocity_x[h_cell_idx][v_cell_idx] + timestep_duration *
                    (diffusion / reynolds - self_advection_x - cross_advection_y);

            } else
                // If both adjacent cells are not fluids, the velocity is unchanged.
                tentative_velocity_x[h_cell_idx][v_cell_idx] = velocity_x[h_cell_idx][v_cell_idx];
        }

    for (int h_cell_idx = 1; h_cell_idx < h_cell_count + 1; ++h_cell_idx)
        for (int v_cell_idx = 1; v_cell_idx < v_cell_count; ++v_cell_idx) {
            if (flags[h_cell_idx][v_cell_idx] & CELL_FLUID && flags[h_cell_idx][v_cell_idx + 1] & CELL_FLUID) {

                const double cross_advection_x =
                    (
                        (velocity_x[h_cell_idx][v_cell_idx] + velocity_x[h_cell_idx][v_cell_idx + 1]) *
                        (velocity_y[h_cell_idx][v_cell_idx] + velocity_y[h_cell_idx + 1][v_cell_idx]) +
                        gamma * fabs(velocity_x[h_cell_idx][v_cell_idx] + velocity_x[h_cell_idx][v_cell_idx + 1]) *
                        (velocity_y[h_cell_idx][v_cell_idx] - velocity_y[h_cell_idx + 1][v_cell_idx]) -
                        (velocity_x[h_cell_idx - 1][v_cell_idx] + velocity_x[h_cell_idx - 1][v_cell_idx + 1]) *
                        (velocity_y[h_cell_idx - 1][v_cell_idx] + velocity_y[h_cell_idx][v_cell_idx]) -
                        gamma * fabs(velocity_x[h_cell_idx - 1][v_cell_idx] +
                            velocity_x[h_cell_idx - 1][v_cell_idx + 1]) *
                        (velocity_y[h_cell_idx - 1][v_cell_idx] - velocity_y[h_cell_idx][v_cell_idx])
                    ) /
                        (4.0 * x_grid_spacing);

                const double self_advection_y =
                    (
                        (velocity_y[h_cell_idx][v_cell_idx] + velocity_y[h_cell_idx][v_cell_idx + 1]) *
                        (velocity_y[h_cell_idx][v_cell_idx] + velocity_y[h_cell_idx][v_cell_idx + 1]) +
                        gamma * fabs(velocity_y[h_cell_idx][v_cell_idx] + velocity_y[h_cell_idx][v_cell_idx + 1]) *
                        (velocity_y[h_cell_idx][v_cell_idx] - velocity_y[h_cell_idx][v_cell_idx + 1]) -
                        (velocity_y[h_cell_idx][v_cell_idx - 1] + velocity_y[h_cell_idx][v_cell_idx]) *
                        (velocity_y[h_cell_idx][v_cell_idx - 1] + velocity_y[h_cell_idx][v_cell_idx]) -
                        gamma * fabs(velocity_y[h_cell_idx][v_cell_idx - 1] + velocity_y[h_cell_idx][v_cell_idx]) *
                        (velocity_y[h_cell_idx][v_cell_idx - 1] - velocity_y[h_cell_idx][v_cell_idx])
                    ) /
                        (4.0 * y_grid_spacing);

                const double diffusion =
                    (
                        velocity_y[h_cell_idx + 1][v_cell_idx] -
                        2.0 * velocity_y[h_cell_idx][v_cell_idx] +
                        velocity_y[h_cell_idx - 1][v_cell_idx]
                    ) /
                        (x_grid_spacing * x_grid_spacing) +
                    (
                        velocity_y[h_cell_idx][v_cell_idx + 1] -
                        2.0 * velocity_y[h_cell_idx][v_cell_idx] +
                        velocity_y[h_cell_idx][v_cell_idx - 1]
                    ) /
                        (y_grid_spacing * y_grid_spacing);

                tentative_velocity_y[h_cell_idx][v_cell_idx] = velocity_y[h_cell_idx][v_cell_idx] + timestep_duration *
                    (diffusion / reynolds - cross_advection_x - self_advection_y);

            } else
                // If both adjacent cells are not fluids, the velocity is unchanged.
                tentative_velocity_y[h_cell_idx][v_cell_idx] = velocity_y[h_cell_idx][v_cell_idx];
        }

    // Tentative velocities along extreme vertical boundaries.
    for (int v_cell_idx = 1; v_cell_idx < v_cell_count + 1; ++v_cell_idx) {
        tentative_velocity_x[0][v_cell_idx] = velocity_x[0][v_cell_idx];
        tentative_velocity_x[h_cell_count][v_cell_idx] = velocity_x[h_cell_count][v_cell_idx];
    }

    // Tentative velocities along extreme horizontal boundaries.
    for (int h_cell_idx = 1; h_cell_idx < h_cell_count + 1; ++h_cell_idx) {
        tentative_velocity_y[h_cell_idx][0] = velocity_y[h_cell_idx][0];
        tentative_velocity_y[h_cell_idx][v_cell_count] = velocity_y[h_cell_idx][v_cell_count];
    }
}

/**
 * @brief Calculate the right hand side of the pressure equation
 */
static void compute_poisson_source()
{
    for (int h_cell_idx = 1; h_cell_idx < h_cell_count + 1; ++h_cell_idx)
        for (int v_cell_idx = 1; v_cell_idx < v_cell_count + 1; ++v_cell_idx)
            if (flags[h_cell_idx][v_cell_idx] & CELL_FLUID) {
                /* only for fluid and non-surface cells */

                const double x_tent_vel_diff = (tentative_velocity_x[h_cell_idx][v_cell_idx] -
                    tentative_velocity_x[h_cell_idx - 1][v_cell_idx]) / x_grid_spacing;

                const double y_tent_vel_diff = (tentative_velocity_y[h_cell_idx][v_cell_idx] -
                    tentative_velocity_y[h_cell_idx][v_cell_idx - 1]) / y_grid_spacing;

                poisson_source[h_cell_idx][v_cell_idx] = (x_tent_vel_diff + y_tent_vel_diff) / timestep_duration;
            }
}

/**
 * @brief Red-black successive order relaxation to solve the poisson equation.
 * @return Calculated residual of the computation
 *
 */
static double compute_pressure()
{
    static const int itermax = 100; /* Maximum number of iterations in SOR */
    static const double epsilon = 0.001; /* Stopping error threshold for SOR */
    static const double omega = 1.7; /* Relaxation parameter for SOR */

    compute_poisson_source();

    const double rdx2 = 1.0 / (x_grid_spacing * x_grid_spacing);
    const double rdy2 = 1.0 / (y_grid_spacing * y_grid_spacing);
    const double beta_2 = -omega / (2.0 * (rdx2 + rdy2));

    double p0 = 0.0;

    /* Calculate sum of squares */
    for (int h_cell_idx = 1; h_cell_idx < h_cell_count + 1; ++h_cell_idx)
        for (int v_cell_idx = 1; v_cell_idx < v_cell_count + 1; ++v_cell_idx)
            if (flags[h_cell_idx][v_cell_idx] & CELL_FLUID)
                p0 += pressure[h_cell_idx][v_cell_idx] * pressure[h_cell_idx][v_cell_idx];

    p0 = sqrt(p0 / fluid_cell_count);
    if (p0 < 0.0001)
        p0 = 1.0;

    /* Red/Black SOR-iteration */

    double residual = 0.0;

    for (int iteration = 0; iteration < itermax; ++iteration) {
        for (int rb = 0; rb < 2; rb++)
            for (int h_cell_idx = 1; h_cell_idx < h_cell_count + 1; h_cell_idx++)
                for (int v_cell_idx = 1; v_cell_idx < v_cell_count + 1; v_cell_idx++) {
                    if ((h_cell_idx + v_cell_idx) % 2 != rb)
                        continue;

                    if (flags[h_cell_idx][v_cell_idx] == (CELL_FLUID | CELL_FLUID_ALL)) {
                        /* five point star for interior fluid cells */
                        pressure[h_cell_idx][v_cell_idx] = (1.0 - omega) * pressure[h_cell_idx][v_cell_idx] -
                                beta_2 *
                                        ((pressure[h_cell_idx + 1][v_cell_idx] + pressure[h_cell_idx - 1][v_cell_idx]) * rdx2 +
                                                (pressure[h_cell_idx][v_cell_idx + 1] + pressure[h_cell_idx][v_cell_idx - 1]) * rdy2 -
                                                poisson_source[h_cell_idx][v_cell_idx]);

                    } else if (flags[h_cell_idx][v_cell_idx] & CELL_FLUID) {
                        /* modified star near boundary */

                        const double epsilon_east = !!(flags[h_cell_idx + 1][v_cell_idx] & CELL_FLUID);
                        const double epsilon_west = !!(flags[h_cell_idx - 1][v_cell_idx] & CELL_FLUID);
                        const double epsilon_north = !!(flags[h_cell_idx][v_cell_idx + 1] & CELL_FLUID);
                        const double epsilon_south = !!(flags[h_cell_idx][v_cell_idx - 1] & CELL_FLUID);

                        const double beta_mod = -omega / ((epsilon_east + epsilon_west) * rdx2 + (epsilon_north + epsilon_south) * rdy2);
                        pressure[h_cell_idx][v_cell_idx] = (1.0 - omega) * pressure[h_cell_idx][v_cell_idx] -
                                beta_mod *
                                        ((epsilon_east * pressure[h_cell_idx + 1][v_cell_idx] + epsilon_west * pressure[h_cell_idx - 1][v_cell_idx]) * rdx2 +
                                                (epsilon_north * pressure[h_cell_idx][v_cell_idx + 1] + epsilon_south * pressure[h_cell_idx][v_cell_idx - 1]) * rdy2 -
                                                poisson_source[h_cell_idx][v_cell_idx]);
                    }
                }

        // Compute the Laplacian residual over fluid cells.

        for (int h_cell_idx = 1; h_cell_idx < h_cell_count + 1; ++h_cell_idx)
            for (int v_cell_idx = 1; v_cell_idx < v_cell_count + 1; ++v_cell_idx)
                if (flags[h_cell_idx][v_cell_idx] & CELL_FLUID) {

                    const double epsilon_east = !!(flags[h_cell_idx + 1][v_cell_idx] & CELL_FLUID);
                    const double epsilon_west = !!(flags[h_cell_idx - 1][v_cell_idx] & CELL_FLUID);
                    const double epsilon_north = !!(flags[h_cell_idx][v_cell_idx + 1] & CELL_FLUID);
                    const double epsilon_south = !!(flags[h_cell_idx][v_cell_idx - 1] & CELL_FLUID);

                    const double add = (epsilon_east * (pressure[h_cell_idx + 1][v_cell_idx] - pressure[h_cell_idx][v_cell_idx]) -
                                         epsilon_west * (pressure[h_cell_idx][v_cell_idx] - pressure[h_cell_idx - 1][v_cell_idx])) *
                                    rdx2 +
                            (epsilon_north * (pressure[h_cell_idx][v_cell_idx + 1] - pressure[h_cell_idx][v_cell_idx]) -
                                    epsilon_south * (pressure[h_cell_idx][v_cell_idx] - pressure[h_cell_idx][v_cell_idx - 1])) *
                                    rdy2 -
                            poisson_source[h_cell_idx][v_cell_idx];
                    residual += add * add;
                }

        residual = sqrt(residual / fluid_cell_count) / p0;
        if (residual < epsilon)
            break; // The residual has converged.
    }

    return residual;
}


/**
 * @brief Update the velocity values based on the tentative velocity values and the new pressure matrix.
 *
 * Mutations are done where both adjacent cells are fluid cells. On the fixed axis, the next velocity value can be
 * computed by Equation (5) in the assignment brief. In particular, the current tentative velocity minus the fixed time
 * increment, multiplied by the pressure differential between the current and next points.
 */
static void update_velocities()
{
    /*
     * The pressure differential factors are the constants implied by the discretisation of the momentum equation. They
     * represent fixed-axis grid spacings, warped by the timestep duration, to numerically approximate the next velocity
     * values in terms of the computed pressures.
     */
    const double x_pressure_diff_factor = timestep_duration / x_grid_spacing;
    const double y_pressure_diff_factor = timestep_duration / y_grid_spacing;

    // X velocities
    for (int h_cell_idx = 1; h_cell_idx < h_cell_count - 2; h_cell_idx++)
        for (int v_cell_idx = 1; v_cell_idx < v_cell_count - 1; v_cell_idx++)
            if (flags[h_cell_idx][v_cell_idx] & CELL_FLUID && flags[h_cell_idx + 1][v_cell_idx] & CELL_FLUID)
                velocity_x[h_cell_idx][v_cell_idx] = tentative_velocity_x[h_cell_idx][v_cell_idx] -
                    (pressure[h_cell_idx + 1][v_cell_idx] - pressure[h_cell_idx][v_cell_idx]) * x_pressure_diff_factor;

    // Y velocities
    for (int h_cell_idx = 1; h_cell_idx < h_cell_count - 1; h_cell_idx++)
        for (int v_cell_idx = 1; v_cell_idx < v_cell_count - 2; v_cell_idx++)
            if (flags[h_cell_idx][v_cell_idx] & CELL_FLUID && flags[h_cell_idx][v_cell_idx + 1] & CELL_FLUID)
                velocity_y[h_cell_idx][v_cell_idx] = tentative_velocity_y[h_cell_idx][v_cell_idx] -
                    (pressure[h_cell_idx][v_cell_idx + 1] - pressure[h_cell_idx][v_cell_idx]) * y_pressure_diff_factor;
}

/**
 * @brief Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy conditions. Otherwise the simulation
 *  becomes unstable.
 */
static void set_timestep_interval()
{
    static const double safety_factor = 0.5;

    /* timestep_duration satisfying CFL conditions */
    if (safety_factor >= 1.0e-10) { /* else no time stepsize control */
        double x_maximum = 1.0e-10;
        double y_maximum = 1.0e-10;

        // Maximise the X velocity
        for (int h_cell_idx = 0; h_cell_idx < h_cell_count + 2; ++h_cell_idx)
            for (int v_cell_idx = 1; v_cell_idx < v_cell_count + 2; ++v_cell_idx)
                x_maximum = fmax(fabs(velocity_x[h_cell_idx][v_cell_idx]), x_maximum);

        // Maximise the Y velocity
        for (int h_cell_idx = 1; h_cell_idx < h_cell_count + 2; ++h_cell_idx)
            for (int v_cell_idx = 0; v_cell_idx < v_cell_count + 2; ++v_cell_idx)
                y_maximum = fmax(fabs(velocity_y[h_cell_idx][v_cell_idx]), y_maximum);

        /*
         * Scale the velocities i.a.w. the problem dimensions and set the timestep duration to not exceed an extent that
         * would cause numerical instability during the PDE solve. Based on the CFL constraints, this ensures that the
         * fluid does not move more than one grid cell per timestamp.
         */
        const double cfl_limit = fmin(x_grid_spacing / x_maximum, y_grid_spacing / y_maximum);
        const double reynolds_delta = 1.0 /
            (1.0 / (x_grid_spacing * x_grid_spacing) + 1 / (y_grid_spacing * y_grid_spacing)) * reynolds / 2.0;

        timestep_duration = safety_factor * fmin(cfl_limit, reynolds_delta);
    }
}

/**
 * @brief The main routine that sets up the problem and executes the solving routines routines
 *
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(const int argc, char **argv)
{
    set_default_base();
    parse_args(argc, argv);

    x_grid_spacing = problem_space_width / h_cell_count;
    y_grid_spacing = problem_space_height / v_cell_count;

    if (verbose)
        print_opts();

    allocate_arrays();
    problem_set_up();
    apply_boundary_conditions();

    double pressure_residual = 0.0;

    int iteration = 0;
    double t;
    for (t = 0.0; t < t_end; t += timestep_duration, ++iteration) {
        if (!fixed_dt)
            set_timestep_interval();

        compute_tentative_velocities();
        pressure_residual = compute_pressure();
        update_velocities();
        apply_boundary_conditions();

        if (iteration % output_freq == 0) {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iteration, t + timestep_duration,
                timestep_duration, pressure_residual);

            if (!no_output && enable_checkpoints)
                write_checkpoint(iteration, t + timestep_duration);
        }
    }

    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iteration, t, pressure_residual);
    printf("Simulation complete.\n");

    if (!no_output)
        write_result(iteration, t);

    free_arrays();

    return 0;
}
