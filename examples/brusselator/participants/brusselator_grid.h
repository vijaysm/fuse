/* examples/brusselator/participants/brusselator_grid.h -- the ONE place the
 * Brusselator participants' grid and physics constants live.
 *
 * The graph tool writes the meshes into the graph, U and V allocate and
 * register storage of the matching size, and the three must agree. If
 * each file kept its own copy of these numbers, the copies could drift:
 * participants registering 9-element buffers against 256-cell fields
 * would not be refused until a run produced a trajectory that could not
 * be reconciled with the single-process example. A C header both C and
 * C++ include, and that the Fortran participant reads through
 * bruss_grid_dims() in shim.h, cannot drift that way.
 */
#ifndef FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_BRUSSELATOR_GRID_H_
#define FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_BRUSSELATOR_GRID_H_

#define BRUSS_NX_U 16
#define BRUSS_NY_U 16
#define BRUSS_NX_V 24
#define BRUSS_NY_V 24
/* diffusion, A, B and forcing are the single-process example's own
 * (examples/brusselator/main.cpp's defaults). */
#define BRUSS_DIFFUSION 10.0
#define BRUSS_A 3.4
#define BRUSS_B 1.0
#define BRUSS_FORCING 1

#endif /* FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_BRUSSELATOR_GRID_H_ */
