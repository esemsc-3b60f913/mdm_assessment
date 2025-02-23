# acsefunctions
A numerical software package for computing transcendental and special functions using series expansions and integral definitions.

## Features
- **Transcendental Functions:** Compute `exp(x)`, `sinh(x)`, `cosh(x)`, and `tanh(x)` using Taylor series expansions with customizable precision.
- **Special Functions:** Calculate `factorial(n)`, `gamma(z)`, and `bessel_j(alpha, x)` with efficient and accurate implementations.
- **Vectorized Operations:** Functions seamlessly handle both scalar inputs and NumPy arrays for efficient computation.
- **Precision Control:** Optional `tol` parameter allows users to specify convergence tolerance for series-based functions.
- **Comprehensive Testing:** Includes unit tests, docstring examples, and CI workflows to ensure reliability.
- **Documentation:** Features Sphinx-generated API docs and Jupyter notebooks with usage examples and performance insights.

## Installation
Follow these steps to set up and install the `acsefunctions` package:

### Clone the Repository:
```bash
git clone <repository_url>
cd acsefunctions
```

### Create and Activate the Conda Environment:
```bash
conda env create -f environment.yml
conda activate acsefunctions_env
```

### Install the Package:
```bash
pip install -e .
```
This installs the package in editable mode, along with all dependencies listed in `environment.yml` and `requirements.txt`, such as NumPy.

## Running Tests
To verify the package’s functionality, run the full test suite (unit tests and docstring tests):
```bash
pytest tests/
```
Tests check accuracy against NumPy/SciPy, handle edge cases (e.g., `exp(0) = 1`), and ensure proper scalar/array output types.

## Usage
### Importing Functions
```python
from acsefunctions import exp, sinh, cosh, tanh, factorial, gamma, bessel_j
import numpy as np
```

### Examples
#### Transcendental Functions
```python
print(exp(1))                  # Approx. 2.71828
print(sinh(np.array([0, 1])))  # Array([0.0, 1.1752...])
print(cosh(0))                 # 1.0
print(tanh(np.array([-1, 1]))) # Array([-0.7615..., 0.7615...])
```

#### Special Functions
```python
print(factorial(5))         # 120
print(gamma(0.5))           # Approx. 1.77245 (sqrt(pi))
print(bessel_j(0, 0))       # 1.0
```

#### Vectorized Inputs
```python
x = np.linspace(-1, 1, 5)
print(exp(x))               # Array of exp values
print(gamma(x + 1))         # Array of gamma values for x > 0
```
Functions accept a `tol` parameter (default `1e-10`) to control series convergence precision.

## Documentation
### API Documentation
Built with Sphinx. To generate locally:
```bash
cd docs
make html
```
Open `docs/build/html/index.html` in a browser for detailed function descriptions, parameters, and examples.

### Usage Examples
Check `docs/source/examples.ipynb` for interactive demonstrations, including plots of function outputs.

### Performance Analysis
See `performance.ipynb` for execution time comparisons and error analysis against NumPy/SciPy implementations.

## Continuous Integration (CI)
The repository leverages GitHub Actions for CI, providing:
- **Testing and Linting:** Executes `pytest` and `flake8` across Ubuntu and Windows with Python 3.8 and 3.10.
- **Environment Consistency:** Validates `environment.yml` and `requirements.txt`.
- **Documentation Automation:** Builds Sphinx docs and commits a PDF version on updates to documentation files.
- **Notebook Validation:** Ensures `examples.ipynb` runs correctly using `nbval`.

This ensures code quality and reliability for users and contributors.

## Development
- **Code Quality:** Adheres to PEP8 standards, enforced via Flake8 linting.
- **Dependencies:** Managed through `environment.yml` (Conda) and `requirements.txt` (Pip).
- **Version Control:** Use feature branches (e.g., `feature/part1`, `feature/part2`) and commit changes regularly for collaborative development.

## Function Details
### Implementations
- **Taylor Series:** `exp`, `sinh`, and `cosh` compute terms iteratively until `|term| < tol`. `tanh` is derived as `sinh(x) / cosh(x)` for efficiency.
- **Special Functions:** `factorial` uses iterative multiplication, `gamma` leverages integral or recursive definitions, and `bessel_j` implements the Bessel function series.
- **Vectorization:** NumPy’s array operations enable simultaneous computation across elements, with convergence based on the maximum term magnitude.

### API Design
- **Parameters:** Functions take an input (`x`, `n`, etc.) and an optional `tol` for precision.
- **Docstrings:** Provide descriptions, parameter details, return types, and usage examples (scalar and array).

### Verification
- **Accuracy:** Tested against NumPy/SciPy over ranges like `[-10, 10]` using `assert_allclose`.
- **Edge Cases:** Confirmed for key inputs (e.g., `sinh(0) = 0`, `bessel_j(0, 0) = 1`).
- **Output Types:** Scalars return as `float`, arrays as `ndarray`.

## License
This project is licensed under the MIT License.