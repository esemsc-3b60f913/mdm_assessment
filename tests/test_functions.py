# tests/test_functions.py
import numpy as np
from acsefunctions import exp, sinh, cosh, tanh
from numpy.testing import assert_allclose


# tests/test_functions.py
import doctest
import acsefunctions.functions


def test_docstrings():
    doctest.testmod(acsefunctions.functions)


def test_exp():
    """Test the exp function for accuracy and input handling."""
    x = np.linspace(-10, 10, 100)
    assert_allclose(exp(x), np.exp(x), rtol=1e-8, atol=1e-10)
    assert exp(0) == 1.0
    assert isinstance(exp(1), float)
    result = exp(np.array([0, 1, 2]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_sinh():
    """Test the sinh function for accuracy and input handling."""
    x = np.linspace(-10, 10, 100)
    assert_allclose(sinh(x), np.sinh(x), rtol=1e-8, atol=1e-10)
    assert sinh(0) == 0.0


def test_cosh():
    """Test the cosh function for accuracy and input handling."""
    x = np.linspace(-10, 10, 100)
    assert_allclose(cosh(x), np.cosh(x), rtol=1e-8, atol=1e-10)
    assert cosh(0) == 1.0


def test_tanh():
    """Test the tanh function for accuracy and input handling."""
    x = np.linspace(-10, 10, 100)
    assert_allclose(tanh(x), np.tanh(x), rtol=1e-8, atol=1e-10)
    assert tanh(0) == 0.0
