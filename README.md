## About pyDVC

pyDVC is a Python-based implementation of the Discrete Voronoi Chain tessellation algorithm, originally introduced by Velic, May and Moresi (2009) [[1]](#1).
This algorithm will perform a bounded tessellation of 3D point-cloud centre-of-mass data on a 3D voxel grid to a desired resolution.
Although the algorithm is single-threaded, pyDVC utilises the Numba Python library for speed, and is usually reasonably quick for N~= 1000-10000 particles.

pyDVC can be found on [pyPI](https://pypi.org/project/pyDVC/) and [GitHub](https://github.com/jadball/pyDVC)

### Dependencies

```
numpy
numba
python>=3.6
```

### Installation

If you want to install pyDVC to an existing Python virtual environment, simply run `pip install pyDVC` to install pyDVC and its dependencies.

If you want to set up a dedicated virtual environment for pyDVC, follow the instructions below, which assume you already have Python >= 3.6 installed on some sort of UNIX system which is accessible via `python3`.

1. Create a folder to put your new virtual environment in:
    ```bash
    mkdir env_pyDVC
    ```
2. Create a virtual environment using your system Python:
    ```bash
    python3 -m venv ./env_pyDVC
    ```
3. Activate your new virtual environment:
    ```bash
    source ./env_pyDVC/bin/activate
    ```
4. Install pyDVC with pip inside your new virtual environment:
    ```bash
    pip install pyDVC
    ```

<!-- USAGE EXAMPLES -->
## Usage

You can find an example Jupyter Notebook with some plotting on [GitHub](https://github.com/jadball/pyDVC) in [examples/simple_example.ipynb](https://github.com/jadball/pyDVC/blob/main/examples/simple_example.ipynb)
A simpler example is below:

```py
import numpy as np
from pyDVC import tessellate

# Generate 1000 particles
n_particles = 1000

# Give each particle a random 3D position from -0.5 to 0.5 in x, y and z
positions = np.random.rand(n_particles, 3) - 0.5

# Give each particle a random radius to use as the weight
weights = np.random.rand(n_particles) * 0.05

# Define the bounds of the tessellation
bounds = np.array([
    [-0.5, 0.5],
    [-0.5, 0.5],
    [-0.5, 0.5]
])

# Define the number of cells (grid points) along each axis
n_cells = 100

# The bounding box side length is 1 (-0.5 to 0.5), so the size of each cell is 1/n_cells
cell_length = 1/n_cells

# Call the tessellation
ownership_array = tessellate(positions, bounds, cell_length, weights)

# Print the result
print(ownership_array)
```

## References
<a id="1">[1]</a>
Velic, May and Moresi (2009).
A Fast Robust Algorithm for Computing Discrete Voronoi Diagrams.
J Math Model Algor. 8. 343-355.
http://dx.doi.org/10.1007/s10852-008-9097-6
