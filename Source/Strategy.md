# OpenMPI Implementation

In the OpenMPI variant, a virtual two-dimensional Cartesian topology is used to segment the problem space. Due to the
nature of the boundaries, the topology is non-period in both dimensions. The exact dimensions can be computed by
OpenMPI, given the number of available ranks, by the `MPI_Dims_create` function.

## Domain Division and Initialisation

The local regions for each rank must be divided, as uniformly as possible, to collectively cover the entire problem
space. This can be trivially done by integer division, such that each ranks' region has size
$ \left( \lceil Rw \rceil / N_H, \lceil Rh \rceil / N_V \right)$, where $R$ indicates the resolution (number of cells
per unit-distance, nominally metre, in the problem space), $w$ and $h$ indicate the physical extents of the width and
height of the problem space, respectively, and $N_H$ and $N_V$ indicate the number of ranks available in the horizontal
and vertical dimension, respectively. For each terminating rank in dimension $ d = \left\{ H, V \right\} $; the rank
$ r \in \text{Ranks} $ such that $ r = \text{arg max}_\text{Ranks}~N_d $, should be extended to accommodate any cells
that could not be allocated by integer division. The excesses in the respective dimensions are the modulo operation in
C.

**TODO: ghost rows and columns.**

Once the rank-local regions have been determined, the information grids (velocities, pressure, and flags) are
initialised by a three-stage process:

1. Initialise the interiors of all information grids with defined initial constants:
   * X velocity: 1.0 m/s
   * Y velocity: 0.0 m/s
   * Pressure: 0 Pa
   * Flag: *fluid*

2. Initialise the outline of the defined airfoil shape into the corresponding cells in the flags array to the *border*
   type. This requires the computation of a standard cambered four-digit NACA airfoil, implemented as an
   $\mathcal{O}(h)$ loop in the number of horizontal cells $h$.
   
   1. Let $x$ be the position along the chord, normalised into $[0, 1]$. Let $t$ be the maximum thickness as a fraction
      of the chord. Compute the midline distance (half-thickness) at $x$, $y_t$:
      $$ y_t(x) := 5t\left( 0.2969 \sqrt{x} - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1015x^4 \right) $$
   2. Let $p$ be the location of the maximum camber along the chord, and let $m$ be the maximum camber. Compute the mean
      camber line $y_c$ for the upper and lower portions:
      $$
         y_c(x) := \begin{cases}
            m/p^2 \left( 2px - x^2 \right) &\text{ for } 0 \leq x \leq p; \\
            m/(1-p)^2 \left[ \left( 1 - 2p \right) + 2px - x^2 \right] &\text{ for } p < x \leq 1.
         \end{cases}
      $$
   3. Compute the derivative of $y_c$:
      $$
         \frac{\mathrm{d}y_c}{\mathrm{d}x} = \begin{cases}
            2m/p^2 \left( p - x \right) &\text{ for } 0 \leq x \leq p; \\
            2m/(1 - p)^2 \left( p - x \right) &\text{ for } p < x \leq 1.
         \end{cases}
      $$
   4. Use standard geometric formulae to compute the perpendicular vector to the camber line, for the upper ($U$) and
      lower ($L$) portions:
      $$
         U := \begin{pmatrix}
            x - y_t \sin\theta \\
            y_c + y_t \cos\theta
         \end{pmatrix}
         \qquad
         L := \begin{pmatrix}
            x + y_t \sin\theta \\
            y_c - y_t \cos\theta
         \end{pmatrix}
      $$
      where
      $ \theta = \arctan \mathrm{d}y_c/\mathrm{d}x $.
      
   Subject to scaling for the problem dimensions, this procedure provides the $N$ vertical points
   $\left\{ y_1,\ldots,y_N \right\}$ at the fixed $x$ horizontal co-ordinate for which a boundary should exist. This
   effectively traces out the shape of the airfoil along co-ordinates
   $$ \left\{ \left( x, y_1 \right), \ldots, \left( x, y_N \right) \right\}. $$

3. Initialise the flags array along the relevant exteriors to delimit the problem space borders.
   
4. Do a final pass of the entire region to mask in additional flags indicating the presence of fluid in neighbours of
   non-fluid cells.
   
## 
