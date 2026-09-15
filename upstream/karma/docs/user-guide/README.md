# Scientific User Guide

This guide describes the active, tested portion of KARMA. It is intended for scientists constructing meshes, finite-element spaces, scalar transport or diffusion systems, mesh-free ODE models, and restartable MPI workflows.

Start with [Getting Started](getting-started.md), then use:

- [Meshes and Finite-Element Spaces](meshes-and-spaces.md) for topology, geometry, variables, bases, and DoFs.
- [Physics Systems](physics-systems.md) for mass, diffusion-reaction, advection-reaction, boundary conditions, and ODEs.
- [Parallel Runs and Checkpoints](parallel-and-checkpoints.md) for partitions, ghosts, H5M files, and rank-count-independent restart.
- [Gray-Scott Coupler](gray-scott-coupler.md) for the native two-component multimesh operator-split prototype.

KARMA is still being modernized. Some legacy headers remain installed but do not have implementations in the active CMake target. The capability tables in the root README and these pages are the supported surface.
