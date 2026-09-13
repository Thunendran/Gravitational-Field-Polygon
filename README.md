# Gravitational-Field-Polygon

**Analytical Gravitational Field of a Homogeneous Polygon**

Multi-language implementation (Python, MATLAB, Julia) of analytical formulations for the gravitational potential, acceleration, and gravity-gradient tensor of homogeneous triangular and rectangular laminae. Python cuboid routines are included for numerical comparisons.

---

## Reference

> Periyandy, T., & Bevis, M. (2026).  
> *The Gravitational Field of a Homogeneous Polygon.*  
> Mathematical Geosciences.  
> https://doi.org/10.1007/s11004-026-10325-6

**Authors:**

- Thunendran Periyandy — The Ohio State University / Sabaragamuwa University of Sri Lanka
- Michael Bevis — The Ohio State University

**Correspondence:** thunendran@gmail.com

---

## Overview

The triangle and rectangle implementations compute:

| Quantity | Symbol | Output per observation point |
|---|---|---|
| Gravitational potential | U(P) | Scalar |
| Gravitational acceleration | g(P) | Three components |
| Gravity-gradient tensor | Γ(P) | 3 × 3 matrix, returned as an array or component vectors |

The repository includes:

- **Triangle elements:** specified by three vertices in three-dimensional Cartesian coordinates.
- **Rectangle elements:** specified by two half-dimensions, with observations expressed in the rectangle's coordinate frame.
- **Batch evaluation:** evaluation at multiple observation points, with parallel options in the supplied implementations.
- **Validation notebooks:** thin-cuboid, rectangle, and triangle comparisons, plots, and benchmark workflows.

A general planar polygon can be represented by a suitable triangulation and summation of element contributions. Automatic polygon triangulation is not included. Review the sign-convention note under **Validation and Numerical Notes** before combining outputs.

---

## Repository Structure

```text
Gravitational-Field-Polygon-main/
├── Python/
│   ├── TriangleGravitationalField/
│   │   ├── __init__.py
│   │   └── Triangle_GP.py               # TriangleLaminaGravitation class
│   ├── RectangleGravitationalField/
│   │   ├── rectangle_potential.py
│   │   ├── rectangle_acceleration.py
│   │   ├── rectangle_tensor.py
│   │   └── safe_math.py
│   ├── CuboidGravitationalField/
│   │   ├── __init__.py
│   │   ├── potential.py
│   │   ├── acceleration.py
│   │   ├── tensor.py
│   │   ├── laplacian.py
│   │   ├── surfaces.py
│   │   └── safe_math.py
│   ├── example_basic.py                 # Runnable Python quick start
│   ├── Validation_Cuboid_Rectangle_Triangle.ipynb
│   └── bench_rectangle_py_vs_matlab.ipynb
├── MATLAB/
│   ├── triangle_gravity_matlab.m
│   ├── rectangle_gravity_matlab.m
│   ├── Example_Trinangle_Basic.m         # Existing filename spelling
│   └── Example_Rectangle_Basic.m
├── Julia/
│   ├── TriangleGravity.jl
│   ├── RectangleGravity.jl
│   ├── Example_Triangle_Basic_JULIA.jl
│   └── Example_Rectangle_Basic_JULIA.jl
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## Requirements

### Python

- Python 3 with NumPy for the numerical routines.
- Matplotlib and JupyterLab for notebook execution and plotting.
- A LaTeX installation for notebook plots configured with `text.usetex=True`, or change that setting to `False` before plotting.

The Python quick-start script was exercised with Python 3.13 and NumPy 2.3.3. Minimum supported versions have not been established. The requirements file is unpinned and is not a frozen reproduction environment.

### MATLAB

- MATLAB for the `.m` implementations and examples.
- Parallel Computing Toolbox for parallel `parfor` evaluation.
- For serial rectangle evaluation, set `parallel_threshold = Inf` in the example or batch call.

### Julia

- Julia for the `.jl` implementations and examples.
- The supplied modules import `Base.Threads`; no third-party packages are required.

Minimum MATLAB and Julia release compatibility has not been established.

---

## Quick Start

Download and extract the repository, or clone it using the URL in the GitHub **Code** menu. The commands below assume you start in the repository root.

### Python: Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python Python/example_basic.py
```

On Windows, create the environment with `py -m venv .venv` and activate it in PowerShell with `.venv\Scripts\Activate.ps1`.

The script prints potential, acceleration, tensor components, and tensor traces for a triangle and a rectangle. For the interactive examples below, first change into `Python/` and start Python, or use a notebook located in that directory. This makes the local modules available for import.

### Python: Triangle

```python
import numpy as np
from TriangleGravitationalField.Triangle_GP import TriangleLaminaGravitation

# Rows are the triangle vertices [x, y, z].
vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

# Normalized gravitational constant and surface density.
model = TriangleLaminaGravitation(vertices, G=1.0, sigma=1.0, eps=1e-15)

# Two observation points away from the mass plane.
points = np.array([[0.2, 0.2, 1.0], [0.2, 0.2, -1.0]])
U = model.potential(points)           # (N,)
g = model.acceleration(points)       # (N, 3)
Gamma = model.gravity_tensor(points) # (N, 3, 3)

print(U)
print(g)
print(Gamma)
```

Use three non-collinear vertices. A single observation point of shape `(3,)` returns a scalar potential, an acceleration of shape `(3,)`, and a tensor of shape `(3, 3)`.

### Python: Rectangle

```python
import numpy as np
from RectangleGravitationalField.rectangle_potential import potential_batch_rectangle
from RectangleGravitationalField.rectangle_acceleration import acceleration_batch_rectangle
from RectangleGravitationalField.rectangle_tensor import tensor_batch_rectangle

points = np.array([[0.2, 0.2, 1.0], [0.2, 0.2, -1.0]])
L, B = 1.0, 1.0  # Half-dimensions: full rectangle is 2L × 2B.
G, sigma = 1.0, 1.0

U = potential_batch_rectangle(points, L, B, sigma, G=G)
gx, gy, gz = acceleration_batch_rectangle(points, L, B, sigma, G=G)
Vxx, Vyy, Vzz, Vxy, Vxz, Vyz = tensor_batch_rectangle(
    points, L, B, z_r=0.0, G_const=G, rho_surf=sigma
)
g = np.column_stack((gx, gy, gz))
trace = Vxx + Vyy + Vzz
```

Each returned component has shape `(N,)`. The symmetric tensor at point `i` is:

```python
i = 0
Gamma_i = np.array([
    [Vxx[i], Vxy[i], Vxz[i]],
    [Vxy[i], Vyy[i], Vyz[i]],
    [Vxz[i], Vyz[i], Vzz[i]],
])
```

### MATLAB

Set MATLAB's Current Folder to the repository root, then run:

```matlab
cd MATLAB
Example_Trinangle_Basic
Example_Rectangle_Basic
```

`Trinangle` is the spelling in the existing triangle example filename. Edit the geometry, density, gravitational constant, and observation-point blocks in these scripts for your model.

For direct triangle evaluation:

```matlab
vertices = [0 0 0; 1 0 0; 0 1 0];
points = [0.2 0.2 1.0; 0.2 0.2 -1.0];
tri = triangle_gravity_matlab('init', vertices, 1.0, 1.0, 1e-15);
U = triangle_gravity_matlab('potential_batch', tri, points);
[gx, gy, gz] = triangle_gravity_matlab('acceleration_batch', tri, points);
[G11,G12,G13,G21,G22,G23,G31,G32,G33] = ...
    triangle_gravity_matlab('tensor_batch', tri, points);
```

Triangle tensors return nine component vectors in row-major order. Rectangle tensors return six component vectors in the order `xx, yy, zz, xy, xz, yz`.

### Julia

Run the supplied examples from the repository root:

```bash
julia Julia/Example_Triangle_Basic_JULIA.jl
julia Julia/Example_Rectangle_Basic_JULIA.jl
```

For direct triangle evaluation in a Julia session started at the repository root:

```julia
include("Julia/TriangleGravity.jl")
using .TriangleGravity

vertices = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
points = [0.2 0.2 1.0; 0.2 0.2 -1.0]
tri = TriangleLaminaGravitation(vertices; G=1.0, sigma=1.0, eps=1e-15)
U = potential_batch_triangle(tri, points)
gx, gy, gz = acceleration_batch_triangle(tri, points)
G11,G12,G13,G21,G22,G23,G31,G32,G33 = tensor_batch_triangle(tri, points)
```

---

## Input Geometry and Units

| Parameter | Meaning |
|---|---|
| `vertices` | Triangle vertices, with shape 3 × 3 |
| `points` | Cartesian observation points, with shape N × 3 |
| `L`, `B` | Rectangle or cuboid half-dimensions along x and y |
| `D` | Cuboid half-thickness along z |
| `sigma`, `rho_surf` | Surface mass density |
| Cuboid `rho` / `rho_vol` | Volume mass density |
| `G` / `G_const` | Gravitational constant in the chosen unit system |
| `eps` / `epsz` | Numerical regularization parameter; behavior depends on the routine |

The rectangle potential and acceleration routines use a rectangle occupying `[-L,L] × [-B,B]` in the plane `z=0`. The tensor also accepts a plane height `z_r`. For a rectangle at height `z0`, a consistent approach is to subtract `z0` from observation-point z coordinates for all calculations and use `z_r=0`. For a rotated rectangle, transform coordinates into its local frame, then transform vectors and tensors back.

Examples use normalized `G=1` and density `1`. For physical calculations, supply consistent units. With SI inputs, surface density is kg/m², volume density is kg/m³, potential is m²/s², acceleration is m/s², and tensor components are s⁻².

For a thin cuboid of full thickness `2D`, the equivalent surface density is `sigma = 2 * D * rho`.

---

## Examples and Notebooks

| File | Description |
|---|---|
| `Python/example_basic.py` | Small triangle and rectangle calculation |
| `Python/Validation_Cuboid_Rectangle_Triangle.ipynb` | Cuboid, rectangle, and triangle comparisons and plots |
| `Python/bench_rectangle_py_vs_matlab.ipynb` | Rectangle and triangle timing and cross-language report comparisons |
| `MATLAB/Example_Trinangle_Basic.m` | Triangle potential, acceleration, tensor, and trace |
| `MATLAB/Example_Rectangle_Basic.m` | Rectangle field quantities and trace |
| `Julia/Example_Triangle_Basic_JULIA.jl` | Triangle field quantities and trace |
| `Julia/Example_Rectangle_Basic_JULIA.jl` | Rectangle field quantities and trace |

Launch notebooks from the repository root with the Python environment activated:

```bash
cd Python
python -m jupyterlab
```

Inspect the configuration and run the relevant sections in order. Plotting cells that enable `text.usetex=True` need LaTeX; use `False` if LaTeX is unavailable. Generated files are written relative to the working directory.

The benchmark comparison sections require matching Python, MATLAB, and Julia reports in `Results/`. Those reports are not included in this folder, and the basic MATLAB/Julia examples do not generate every report expected by the comparison cells. Prepare the required reports before running those sections.

---

## Parallel Evaluation

- **Python triangle:** `potential_parallel`, `acceleration_parallel`, and `gravity_tensor_parallel` accept `n_workers` and `chunk_size`.
- **Python rectangle:** batch functions switch to multiprocessing at 50,000 points by default. Set `parallel_threshold=float('inf')` to remain serial.
- **MATLAB rectangle:** large batches use `parfor`; set `parallel_threshold=Inf` for serial evaluation.
- **Julia:** enable threads at startup, for example `julia --threads=4 Julia/Example_Rectangle_Basic_JULIA.jl`.

For Python multiprocessing, use a saved script with calls inside an `if __name__ == '__main__':` guard, especially on Windows and macOS. Start with serial methods in notebooks.

---

## Validation and Numerical Notes

The notebooks contain numerical comparisons and tensor-trace checks. Away from the mass, an approximately zero tensor trace is a useful consistency check. It does not by itself establish the sign or accuracy of all components. No independently reproduced cross-language performance or accuracy figures are claimed here.

Exact sheet, edge, and vertex evaluations depend on the implemented limiting and regularization conventions. Safe logarithm and arctangent handling differs between routines; begin validation with observation points away from the sheet and inspect boundary behavior separately.

**Sign convention requiring review:** At `(0.2, 0.2, 1.0)`, the Python triangle defined in the quick start returns positive potential and acceleration approximately `(-0.04985, -0.04985, +0.41637)`. A central finite-difference check gives acceleration equal to the negative gradient of that potential. The rectangle example returns negative vertical acceleration above its sheet. Reconcile these conventions before combining triangle and rectangle vector/tensor results in a physical model. The documentation update leaves the numerical implementations unchanged.

---

## Citation

If you use this code, please cite:

```text
Periyandy, T., & Bevis, M. (2026).
The Gravitational Field of a Homogeneous Polygon.
Mathematical Geosciences.
https://doi.org/10.1007/s11004-026-10325-6
```

```bibtex
@article{PeriyandyBevis2026Polygon,
  author  = {Periyandy, Thunendran and Bevis, Michael},
  title   = {The Gravitational Field of a Homogeneous Polygon},
  journal = {Mathematical Geosciences},
  year    = {2026},
  doi     = {10.1007/s11004-026-10325-6},
  url     = {https://doi.org/10.1007/s11004-026-10325-6}
}
```

[Download the publisher's citation](https://citation-needed.springer.com/v2/references/10.1007/s11004-026-10325-6?format=refman&flavour=citation).

---

## License

This code is distributed under the [MIT License](LICENSE). See the license file for permission and attribution requirements. The software license does not specify the article's publication license.
