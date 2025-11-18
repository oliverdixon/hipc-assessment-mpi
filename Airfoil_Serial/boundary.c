#include "boundary.h"
#include "data.h"

/**
 * @brief Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
void apply_boundary_conditions()
{
    for (int j = 0; j < v_cell_count + 2; j++) {
        /* Fluid freely flows in from the west */
        u[0][j] = u[1][j];
        v[0][j] = v[1][j];

        /* Fluid freely flows out to the east */
        u[h_cell_count][j] = u[h_cell_count - 1][j];
        v[h_cell_count + 1][j] = v[h_cell_count][j];
    }

    for (int i = 0; i < h_cell_count + 2; i++) {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        v[i][v_cell_count] = 0.0;
        u[i][v_cell_count + 1] = u[i][v_cell_count];

        v[i][0] = 0.0;
        u[i][0] = u[i][1];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */
    for (int i = 1; i < h_cell_count + 1; i++) {
        for (int j = 1; j < v_cell_count + 1; j++) {
            if (flag[i][j] & CELL_FLUID_ALL) {
                switch (flag[i][j]) {
                case CELL_FLUID_NORTH:
                    v[i][j] = 0.0;
                    u[i][j] = -u[i][j + 1];
                    u[i - 1][j] = -u[i - 1][j + 1];
                    break;
                case CELL_FLUID_EAST:
                    u[i][j] = 0.0;
                    v[i][j] = -v[i + 1][j];
                    v[i][j - 1] = -v[i + 1][j - 1];
                    break;
                case CELL_FLUID_SOUTH:
                    v[i][j - 1] = 0.0;
                    u[i][j] = -u[i][j - 1];
                    u[i - 1][j] = -u[i - 1][j - 1];
                    break;
                case CELL_FLUID_WEST:
                    u[i - 1][j] = 0.0;
                    v[i][j] = -v[i - 1][j];
                    v[i][j - 1] = -v[i - 1][j - 1];
                    break;
                case CELL_FLUID_NORTHEAST:
                    v[i][j] = 0.0;
                    u[i][j] = 0.0;
                    v[i][j - 1] = -v[i + 1][j - 1];
                    u[i - 1][j] = -u[i - 1][j + 1];
                    break;
                case CELL_FLUID_SOUTHEAST:
                    v[i][j - 1] = 0.0;
                    u[i][j] = 0.0;
                    v[i][j] = -v[i + 1][j];
                    u[i - 1][j] = -u[i - 1][j - 1];
                    break;
                case CELL_FLUID_SOUTHWEST:
                    v[i][j - 1] = 0.0;
                    u[i - 1][j] = 0.0;
                    v[i][j] = -v[i - 1][j];
                    u[i][j] = -u[i][j - 1];
                    break;
                case CELL_FLUID_NORTHWEST:
                    v[i][j] = 0.0;
                    u[i - 1][j] = 0.0;
                    v[i][j - 1] = -v[i - 1][j - 1];
                    u[i][j] = -u[i][j + 1];
                    break;
                }
            }
        }
    }

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    v[0][0] = 2 * vi - v[1][0];
    for (int j = 1; j < v_cell_count + 1; j++) {
        u[0][j] = ui;
        v[0][j] = 2 * vi - v[1][j];
    }
}
