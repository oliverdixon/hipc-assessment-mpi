#include <math.h>
#include <stdio.h>

#include "args.h"
#include "data.h"
#include "setup.h"
#include "vtk.h"

/**
 * @brief Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
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

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    velocity_y[0][0] = 2 * vi - velocity_y[1][0];
    for (int j = 1; j < v_cell_count + 1; j++) {
        velocity_x[0][j] = ui;
        velocity_y[0][j] = 2 * vi - velocity_y[1][j];
    }
}

/**
 * @brief Computation of tentative velocity field (f, g)
 *
 */
static void compute_tentative_velocities()
{
    static const double gamma = 0.9; /* Gamma, Upwind differencing factor in PDE discretisation */

    for (int i = 1; i < h_cell_count; i++) {
        for (int j = 1; j < v_cell_count + 1; j++) {
            /* only if both adjacent cells are fluid cells */
            if (flags[i][j] & CELL_FLUID && flags[i + 1][j] & CELL_FLUID) {
                double du2dx = ((velocity_x[i][j] + velocity_x[i + 1][j]) * (velocity_x[i][j] + velocity_x[i + 1][j]) +
                                       gamma * fabs(velocity_x[i][j] + velocity_x[i + 1][j]) * (velocity_x[i][j] - velocity_x[i + 1][j]) -
                                       (velocity_x[i - 1][j] + velocity_x[i][j]) * (velocity_x[i - 1][j] + velocity_x[i][j]) -
                                       gamma * fabs(velocity_x[i - 1][j] + velocity_x[i][j]) * (velocity_x[i - 1][j] - velocity_x[i][j])) /
                        (4.0 * delx);
                double duvdy = ((velocity_y[i][j] + velocity_y[i + 1][j]) * (velocity_x[i][j] + velocity_x[i][j + 1]) +
                                       gamma * fabs(velocity_y[i][j] + velocity_y[i + 1][j]) * (velocity_x[i][j] - velocity_x[i][j + 1]) -
                                       (velocity_y[i][j - 1] + velocity_y[i + 1][j - 1]) * (velocity_x[i][j - 1] + velocity_x[i][j]) -
                                       gamma * fabs(velocity_y[i][j - 1] + velocity_y[i + 1][j - 1]) * (velocity_x[i][j - 1] - velocity_x[i][j])) /
                        (4.0 * dely);
                double laplu = (velocity_x[i + 1][j] - 2.0 * velocity_x[i][j] + velocity_x[i - 1][j]) / delx / delx +
                        (velocity_x[i][j + 1] - 2.0 * velocity_x[i][j] + velocity_x[i][j - 1]) / dely / dely;

                tentative_velocity_x[i][j] = velocity_x[i][j] + del_t * (laplu / Re - du2dx - duvdy);
            } else {
                tentative_velocity_x[i][j] = velocity_x[i][j];
            }
        }
    }

    for (int i = 1; i < h_cell_count + 1; i++) {
        for (int j = 1; j < v_cell_count; j++) {
            /* only if both adjacent cells are fluid cells */
            if (flags[i][j] & CELL_FLUID && flags[i][j + 1] & CELL_FLUID) {
                double duvdx = ((velocity_x[i][j] + velocity_x[i][j + 1]) * (velocity_y[i][j] + velocity_y[i + 1][j]) +
                                       gamma * fabs(velocity_x[i][j] + velocity_x[i][j + 1]) * (velocity_y[i][j] - velocity_y[i + 1][j]) -
                                       (velocity_x[i - 1][j] + velocity_x[i - 1][j + 1]) * (velocity_y[i - 1][j] + velocity_y[i][j]) -
                                       gamma * fabs(velocity_x[i - 1][j] + velocity_x[i - 1][j + 1]) * (velocity_y[i - 1][j] - velocity_y[i][j])) /
                        (4.0 * delx);
                double dv2dy = ((velocity_y[i][j] + velocity_y[i][j + 1]) * (velocity_y[i][j] + velocity_y[i][j + 1]) +
                                       gamma * fabs(velocity_y[i][j] + velocity_y[i][j + 1]) * (velocity_y[i][j] - velocity_y[i][j + 1]) -
                                       (velocity_y[i][j - 1] + velocity_y[i][j]) * (velocity_y[i][j - 1] + velocity_y[i][j]) -
                                       gamma * fabs(velocity_y[i][j - 1] + velocity_y[i][j]) * (velocity_y[i][j - 1] - velocity_y[i][j])) /
                        (4.0 * dely);
                double laplv = (velocity_y[i + 1][j] - 2.0 * velocity_y[i][j] + velocity_y[i - 1][j]) / delx / delx +
                        (velocity_y[i][j + 1] - 2.0 * velocity_y[i][j] + velocity_y[i][j - 1]) / dely / dely;

                tentative_velocity_y[i][j] = velocity_y[i][j] + del_t * (laplv / Re - duvdx - dv2dy);
            } else {
                tentative_velocity_y[i][j] = velocity_y[i][j];
            }
        }
    }

    /* tentative_velocity_x & tentative_velocity_y at external boundaries */
    for (int j = 1; j < v_cell_count + 1; j++) {
        tentative_velocity_x[0][j] = velocity_x[0][j];
        tentative_velocity_x[h_cell_count][j] = velocity_x[h_cell_count][j];
    }
    for (int i = 1; i < h_cell_count + 1; i++) {
        tentative_velocity_y[i][0] = velocity_y[i][0];
        tentative_velocity_y[i][v_cell_count] = velocity_y[i][v_cell_count];
    }
}


/**
 * @brief Calculate the right hand side of the pressure equation
 *
 */
static void compute_poisson_source()
{
    for (int i = 1; i < h_cell_count + 1; i++) {
        for (int j = 1; j < v_cell_count + 1; j++) {
            if (flags[i][j] & CELL_FLUID) {
                /* only for fluid and non-surface cells */
                poisson_source[i][j] = ((tentative_velocity_x[i][j] - tentative_velocity_x[i - 1][j]) / delx + (tentative_velocity_y[i][j] - tentative_velocity_y[i][j - 1]) / dely) / del_t;
            }
        }
    }
}


/**
 * @brief Red/Black SOR to solve the poisson equation.
 *
 * @return Calculated residual of the computation
 *
 */
static double compute_pressure()
{
    static const int itermax = 100; /* Maximum number of iterations in SOR */
    static const double epsilon = 0.001; /* Stopping error threshold for SOR */
    static const double omega = 1.7; /* Relaxation parameter for SOR */

    compute_poisson_source();

    double rdx2 = 1.0 / (delx * delx);
    double rdy2 = 1.0 / (dely * dely);
    double beta_2 = -omega / (2.0 * (rdx2 + rdy2));

    double p0 = 0.0;
    /* Calculate sum of squares */
    for (int i = 1; i < h_cell_count + 1; i++) {
        for (int j = 1; j < v_cell_count + 1; j++) {
            if (flags[i][j] & CELL_FLUID) {
                p0 += pressure[i][j] * pressure[i][j];
            }
        }
    }

    p0 = sqrt(p0 / fluid_cell_count);
    if (p0 < 0.0001) {
        p0 = 1.0;
    }

    /* Red/Black SOR-iteration */
    double res = 0.0;
    for (int iter = 0; iter < itermax; iter++) {
        for (int rb = 0; rb < 2; rb++) {
            for (int i = 1; i < h_cell_count + 1; i++) {
                for (int j = 1; j < v_cell_count + 1; j++) {
                    if ((i + j) % 2 != rb) {
                        continue;
                    }
                    if (flags[i][j] == (CELL_FLUID | CELL_FLUID_ALL)) {
                        /* five point star for interior fluid cells */
                        pressure[i][j] = (1.0 - omega) * pressure[i][j] -
                                beta_2 *
                                        ((pressure[i + 1][j] + pressure[i - 1][j]) * rdx2 + (pressure[i][j + 1] + pressure[i][j - 1]) * rdy2 -
                                                poisson_source[i][j]);
                    } else if (flags[i][j] & CELL_FLUID) {
                        /* modified star near boundary */

                        double eps_E = flags[i + 1][j] & CELL_FLUID ? 1.0 : 0.0;
                        double eps_W = flags[i - 1][j] & CELL_FLUID ? 1.0 : 0.0;
                        double eps_N = flags[i][j + 1] & CELL_FLUID ? 1.0 : 0.0;
                        double eps_S = flags[i][j - 1] & CELL_FLUID ? 1.0 : 0.0;

                        double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                        pressure[i][j] = (1.0 - omega) * pressure[i][j] -
                                beta_mod *
                                        ((eps_E * pressure[i + 1][j] + eps_W * pressure[i - 1][j]) * rdx2 +
                                                (eps_N * pressure[i][j + 1] + eps_S * pressure[i][j - 1]) * rdy2 - poisson_source[i][j]);
                    }
                }
            }
        }

        /* computation of residual */
        for (int i = 1; i < h_cell_count + 1; i++) {
            for (int j = 1; j < v_cell_count + 1; j++) {
                if (flags[i][j] & CELL_FLUID) {
                    double eps_E = flags[i + 1][j] & CELL_FLUID ? 1.0 : 0.0;
                    double eps_W = flags[i - 1][j] & CELL_FLUID ? 1.0 : 0.0;
                    double eps_N = flags[i][j + 1] & CELL_FLUID ? 1.0 : 0.0;
                    double eps_S = flags[i][j - 1] & CELL_FLUID ? 1.0 : 0.0;

                    /* only fluid cells */
                    double add = (eps_E * (pressure[i + 1][j] - pressure[i][j]) - eps_W * (pressure[i][j] - pressure[i - 1][j])) * rdx2 +
                            (eps_N * (pressure[i][j + 1] - pressure[i][j]) - eps_S * (pressure[i][j] - pressure[i][j - 1])) * rdy2 - poisson_source[i][j];
                    res += add * add;
                }
            }
        }
        res = sqrt(res / fluid_cell_count) / p0;

        /* convergence? */
        if (res < epsilon)
            break;
    }

    return res;
}


/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
static void update_velocities()
{
    for (int i = 1; i < h_cell_count - 2; i++) {
        for (int j = 1; j < v_cell_count - 1; j++) {
            /* only if both adjacent cells are fluid cells */
            if (flags[i][j] & CELL_FLUID && flags[i + 1][j] & CELL_FLUID) {
                velocity_x[i][j] = tentative_velocity_x[i][j] - (pressure[i + 1][j] - pressure[i][j]) * del_t / delx;
            }
        }
    }

    for (int i = 1; i < h_cell_count - 1; i++) {
        for (int j = 1; j < v_cell_count - 2; j++) {
            /* only if both adjacent cells are fluid cells */
            if (flags[i][j] & CELL_FLUID && flags[i][j + 1] & CELL_FLUID) {
                velocity_y[i][j] = tentative_velocity_y[i][j] - (pressure[i][j + 1] - pressure[i][j]) * del_t / dely;
            }
        }
    }
}


/**
 * @brief Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions. Otherwise the simulation becomes unstable.
 */
static void set_timestep_interval()
{
    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        double umax = 1.0e-10;
        double vmax = 1.0e-10;

        for (int i = 0; i < h_cell_count + 2; i++) {
            for (int j = 1; j < v_cell_count + 2; j++) {
                umax = fmax(fabs(velocity_x[i][j]), umax);
            }
        }

        for (int i = 1; i < h_cell_count + 2; i++) {
            for (int j = 0; j < v_cell_count + 2; j++) {
                vmax = fmax(fabs(velocity_y[i][j]), vmax);
            }
        }

        double deltu = delx / umax;
        double deltv = dely / vmax;
        double deltRe = 1.0 / (1.0 / (delx * delx) + 1 / (dely * dely)) * Re / 2.0;

        if (deltu < deltv) {
            del_t = fmin(deltu, deltRe);
        } else {
            del_t = fmin(deltv, deltRe);
        }
        del_t = tau * del_t; /* multiply by safety factor */
    }
}

/**
 * @brief The main routine that sets up the problem and executes the solving routines routines
 *
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(const int argc, char ** argv)
{
    set_default_base();
    parse_args(argc, argv);

    delx = problem_space_width / h_cell_count;
    dely = problem_space_height / v_cell_count;

    if (verbose)
        print_opts();

    allocate_arrays();
    problem_set_up();
    apply_boundary_conditions();

    double pressure_residual = 0.0;

    int iteration = 0;
    double t;
    for (t = 0.0; t < t_end; t += del_t, ++iteration) {
        if (!fixed_dt)
            set_timestep_interval();

        compute_tentative_velocities();
        pressure_residual = compute_pressure();
        update_velocities();
        apply_boundary_conditions();

        if (iteration % output_freq == 0) {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iteration, t + del_t, del_t,
                pressure_residual);

            if (!no_output && enable_checkpoints)
                write_checkpoint(iteration, t + del_t);
        }
    }

    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iteration, t, pressure_residual);
    printf("Simulation complete.\n");

    if (!no_output)
        write_result(iteration, t);

    free_arrays();

    return 0;
}
