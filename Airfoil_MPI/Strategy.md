# OpenMPI Implementation

In the OpenMPI variant, a virtual two-dimensional Cartesian topology is used to segment the problem space. Due to the
nature of the boundaries, the topology is non-period in both dimensions. The exact dimensions can be computed by
OpenMPI, given the number of available ranks, by the `MPI_Dims_create` function.

## Initialisation

Once the rank-local regions have been determined, the information grids (velocities, pressure, and flags) are
initialised by a three-stage process:

1. Initialise the interiors of all information grids with defined initial constants:
   * X velocity: 1.0 m/s
   * Y velocity: 0.0 m/s
   * Pressure: 0 Pa
   * Flag: *fluid*
2. ...
3. Initialise the flags array along the relevant exteriors to delimit the problem space borders.
