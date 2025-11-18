#ifndef DATA_H
#define DATA_H

enum cell_flags
{
    CELL_BOUNDARY = 0x0000, /**< Boundary cell */

    CELL_FLUID_NORTH = 0x0001, /**< Boundary cell with fluid to the north */
    CELL_FLUID_SOUTH = 0x0002, /**< Boundary cell with fluid to the south */
    CELL_FLUID_WEST = 0x0004, /**< Boundary cell with fluid to the west */
    CELL_FLUID_EAST = 0x0008, /**< Boundary cell with fluid to the east */

    CELL_FLUID_NORTHWEST = CELL_FLUID_NORTH | CELL_FLUID_WEST,
    CELL_FLUID_SOUTHWEST = CELL_FLUID_SOUTH | CELL_FLUID_WEST,
    CELL_FLUID_NORTHEAST = CELL_FLUID_NORTH | CELL_FLUID_EAST,
    CELL_FLUID_SOUTHEAST = CELL_FLUID_SOUTH | CELL_FLUID_EAST,
    CELL_FLUID_ALL = CELL_FLUID_NORTH | CELL_FLUID_SOUTH | CELL_FLUID_EAST | CELL_FLUID_WEST,

    CELL_FLUID = 0x0010, /**< Fluid cell */
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

extern int itermax; /* Maximum number of iterations in SOR */
extern double eps; /* Stopping error threshold for SOR */
extern double omega; /* Relaxation parameter for SOR */
extern double y; /* Gamma, Upwind differencing factor in PDE */

extern double Re; /* Reynolds number */
extern double ui; /* Initial X velocity */
extern double vi; /* Initial Y velocity */

extern int fluid_cells;

extern double delx, dely;

// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays
extern int u_size_x, u_size_y;
extern double **u;
extern int v_size_x, v_size_y;
extern double **v;
extern int p_size_x, p_size_y;
extern double **p;
extern int rhs_size_x, rhs_size_y;
extern double **rhs;
extern int f_size_x, f_size_y;
extern double **f;
extern int g_size_x, g_size_y;
extern double **g;
extern int flag_size_x, flag_size_y;
extern char **flag;

double **alloc_2d_array(int m, int n);
char **alloc_2d_char_array(int m, int n);
void free_2d_array(void **array);

#endif
