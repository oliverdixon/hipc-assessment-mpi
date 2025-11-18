#ifndef DATA_H
#define DATA_H

enum cell_flags
{
    CELL_BOUNDARY = 0, /**< Boundary cell */

    CELL_FLUID_NORTH = 1, /**< Boundary cell with fluid to the north */
    CELL_FLUID_SOUTH = 1 << 1, /**< Boundary cell with fluid to the south */
    CELL_FLUID_WEST = 1 << 2, /**< Boundary cell with fluid to the west */
    CELL_FLUID_EAST = 1 << 3, /**< Boundary cell with fluid to the east */

    CELL_FLUID_NORTHWEST = CELL_FLUID_NORTH | CELL_FLUID_WEST,
    CELL_FLUID_SOUTHWEST = CELL_FLUID_SOUTH | CELL_FLUID_WEST,
    CELL_FLUID_NORTHEAST = CELL_FLUID_NORTH | CELL_FLUID_EAST,
    CELL_FLUID_SOUTHEAST = CELL_FLUID_SOUTH | CELL_FLUID_EAST,
    CELL_FLUID_ALL = CELL_FLUID_NORTH | CELL_FLUID_SOUTH | CELL_FLUID_EAST | CELL_FLUID_WEST,

    CELL_FLUID = 1 << 4, /**< Fluid cell */
};

struct naca_specifier
{
    unsigned char maximum_camber;
    unsigned char edge_distance;
    unsigned char maximum_thickness;
};

extern struct naca_specifier naca_specifier;

extern double problem_space_width; /* Width of simulated domain */
extern double problem_space_height; /* Height of simulated domain */

extern int h_cell_count; /* Number of cells horizontally */
extern int v_cell_count; /* Number of cells vertically */

extern double t_end; /* Simulation runtime */
extern double del_t; /* Duration of each timestep */
extern double tau; /* Safety factor for timestep control */

extern double Re; /* Reynolds number */
extern double ui; /* Initial X velocity */
extern double vi; /* Initial Y velocity */

extern int fluid_cell_count;

extern double delx, dely;

extern double **velocity_x;
extern double **velocity_y;
extern double **pressure;
extern double **poisson_source;
extern double **tentative_velocity_x;
extern double **tentative_velocity_y;
extern char **flags;

double **alloc_2d_array(int column_count, int row_count);
char **alloc_2d_char_array(int column_count, int row_count);
void free_2d_array(void **array);

#endif
