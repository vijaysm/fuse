# Meshes and Finite-Element Spaces

## Meshes

`KARMA::KARMAMesh` owns a MOAB database. It exposes the raw MOAB interface by design, along with cached vertex, element, and owned-element ranges; side and neighbor queries; coordinates and centroids; material and boundary sets; and MPI ownership metadata.

`generate_box()` creates explicit Edge2, Quad4, or Hex8 meshes in one, two, or three dimensions. `read()` and `write()` delegate to MOAB. H5M is preferred for exact and parallel workflows; VTK and Gmsh are interchange or visualization formats when supported by the linked MOAB build.

Supported geometry mappings are Edge2/3, Tri3/6, Quad4/8/9, Tet4/10, Hex8/27, Prism6/15/18, and Pyramid5/13/14. Exact derivative evaluation at a pyramid apex is undefined and rejected.

## Variables and Bases

Add variables to `KARMASystem` before `init()`. A `VariableSpec` contains a name, basis family, polynomial order, continuity, and component count. The built-in basis is nodal Lagrange:

- P0 for dG and SE labels
- P1 and P2 for cG, dG, and SE labels
- scalar reference bases composed with any supported geometry map
- multiple variables and components at the system/algebra level

Other basis families or orders require registration in a `KARMABasisRegistry`. The SE continuity label currently uses the same built-in nodal basis machinery; it is not a separate built-in spectral basis.

## Degree-of-Freedom Layout

`KARMADofMap` numbers unknowns independently of DMMoab:

- element order is variable-major, component-major, then basis-minor;
- cG DoFs on shared subentities use canonical stable identities;
- dG and SE DoFs are element-local;
- owners receive contiguous owner-major global ranges;
- visible off-rank DoFs are PETSc ghosts;
- exact diagonal and off-diagonal sparsity is available for owned rows.

Hanging-node, periodic, and other algebraic constraints are not implemented.

## Finite-Element Evaluation

`KARMAFEBase` combines a geometry map, field basis, and PETSc-backed quadrature for volume, face, and arbitrary physical-point evaluation. It provides shape values, physical gradients, quadrature points, measures, and face normals. FE execution requires exceptions to be enabled.
