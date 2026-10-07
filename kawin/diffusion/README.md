## Diffusion Module

Classes
- `DiffusionState`
  - Stores u-fraction
- `DiffusionModel`
  - Handles a diffusion model using explicit schemes to evolve the state
  - `SinglePhaseDiffusionModel`
  - `HomogenizationModel`
- `MeshBase`
  - Handles a mesh and defines:
    - Spatial representation of nodes
    - Computing dx/dt
    - Initial profile
    - Boundary conditions
  - `FiniteVolume1D`
    - `Cartesian1D`
    - `Cylindrical1D`
    - `Spherical1D`
  - `FiniteVolume2D`