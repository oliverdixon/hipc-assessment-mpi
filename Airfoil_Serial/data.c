#include <stdlib.h>

#include "data.h"

double problem_space_width = 4.0; /* Width of simulated domain */
double problem_space_height = 1.0; /* Height of simulated domain */

int h_cell_count = 1024; /* Number of cells horizontally */
int v_cell_count = 256; /* Number of cells vertically */

struct naca_specifier naca_specifier =
{
    .maximum_camber = 2,
    .edge_distance = 4,
    .maximum_thickness = 12
};

double t_end = 2.0; /* Simulation runtime */
double del_t = 0.003; /* Duration of each timestep */
double tau = 0.5; /* Safety factor for timestep control */

double Re = 500.0; /* Reynolds number */
double ui = 1.0; /* Initial X velocity */
double vi = 0.0; /* Initial Y velocity */

double delx, dely;

int fluid_cell_count = 0;

double **velocity_x;
double **velocity_y;
double **pressure;
double **poisson_source;
double **tentative_velocity_x;
double **tentative_velocity_y;
char **flags;

/**
 * @brief Allocate a 2D array that is addressable using square brackets
 *
 * @param column_count The first dimension of the array
 * @param row_count The second dimension of the array
 * @return double** A 2D array
 */
double **alloc_2d_array(const int column_count, const int row_count)
{
    double **array;

    array = (double **) malloc(column_count * sizeof(double *));
    array[0] = (double *) calloc(column_count * row_count, sizeof(double));

    for (int column_idx = 1; column_idx < column_count; ++column_idx)
        array[column_idx] = &array[0][column_idx * row_count];

    return array;
}


/**
 * @brief Allocate a 2D char array that is addressable using square brackets
 *
 * @param column_count The first dimension of the array
 * @param row_count The second dimension of the array
 * @return char** A 2D array
 */
char **alloc_2d_char_array(const int column_count, const int row_count)
{
    char **array;

    array = (char **) malloc(column_count * sizeof(char *));
    array[0] = (char *) calloc(column_count * row_count, sizeof(char));

    for (int column_idx = 1; column_idx < column_count; ++column_idx)
        array[column_idx] = &array[0][column_idx * row_count];

    return array;
}

/**
 * @brief Free a 2D array
 *
 * @param array The 2D array to free
 */
void free_2d_array(void **array)
{
    free(array[0]);
    free(array);
}
