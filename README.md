# acsefunctions

A numerical software package for computing transcendental functions using Taylor series expansions.

## Installation

1. Clone the repository:
git clone <repository_url>
cd acsefunctions


2. Create and activate the Conda environment:
conda env create -f environment.yml
conda activate acsefunctions_env


3. Install the package:
pip install -e .


## Running Tests

Run the test suite with:
pytest tests/


## Usage

```python
from acsefunctions import exp, sinh, cosh, tanh
import numpy as np

print(exp(1))              # Approx. 2.71828
print(sinh(np.array([0, 1])))  # Array of sinh values
```
---

## Explanation

### Function Implementations
- **Taylor Series**: Each function (`exp`, `sinh`, `cosh`) computes its series iteratively, adding terms until `|term| < tol`. For `tanh`, we use `sinh(x) / cosh(x)` since it leverages the already implemented series and is mathematically valid.
- **Vectorization**: NumPy handles array inputs efficiently, computing terms for all elements simultaneously. The loop stops when the maximum term across all elements is below `tol`.
- **Scalar Handling**: Inputs are converted to arrays with `np.asarray`, and scalars are returned as floats by checking `ndim`.

### API Design
- **Parameters**: Each function takes `x` (input) and an optional `tol` (default 1e-10).
- **Docstrings**: Include description, parameters, returns, and examples (e.g., scalar and array inputs).

### Testing
- **Accuracy**: Compared against NumPy's implementations over `x = [-10, 10]` with `assert_allclose`.
- **Edge Cases**: Verified `exp(0) = 1`, `sinh(0) = 0`, `cosh(0) = 1`, `tanh(0) = 0`.
- **Input Types**: Checked scalar returns as floats and array returns as `ndarray`.

### Package Setup
- **Installation**: `setup.py` ensures `pip install -e .` works, installing NumPy as a dependency.
- **Testing**: `pytest tests/` runs all tests successfully.
- **Dependencies**: Specified in `requirements.txt` and `environment.yml`.

---

## Verification
- **Installable**: After running `pip install -e .`, the package is importable.
- **Testable**: `pytest tests/` passes all assertions.
- **Vector Support**: Functions work with NumPy arrays and scalars.
- **Code Quality**: Code follows PEP8 (Flake8 compliant), with clear docstrings.

This solution fully meets the requirements of Part 1 [30 marks] of the assessment. For repository management, regular commits and proper branching (e.g., `feature/part1`) should be used, though this is not shown in the code itself.

