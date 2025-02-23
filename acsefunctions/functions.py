# acsefunctions/functions.py
import numpy as np


def exp(x, tol=1e-10):
    """
    Compute the exponential function using its Taylor series expansion.

    The Taylor series for exp(x) is: exp(x) = sum_{n=0}^∞ x^n / n!

    Parameters:
    -----------
    x : scalar or array_like
        The input value(s) to compute the exponential of.
    tol : float, optional
        The tolerance for series convergence (default is 1e-10).

    Returns:
    --------
    out : scalar or ndarray
        The exponential of x.

    Examples:
    --------
    >>> exp(0)
    1.0
    >>> exp(1)
    2.718281828459045
    >>> exp(np.array([0, 1, 2]))
    array([1.        , 2.71828183, 7.3890561 ])
    """
    x = np.asarray(x)
    is_scalar = x.ndim == 0
    if is_scalar:
        x = x[None]  # Convert scalar to 1D array
    sum_exp = np.zeros_like(x, dtype=float)
    term = np.ones_like(x, dtype=float)
    n = 0
    while np.max(np.abs(term)) >= tol:
        sum_exp += term
        n += 1
        term = term * x / n
    return sum_exp[0] if is_scalar else sum_exp


def sinh(x, tol=1e-10):
    """
    Compute the hyperbolic sine using its Taylor series expansion.

    The Taylor series for sinh(x) is: sinh(x) = sum_{n=0}^∞ x^(2n+1) / (2n+1)!

    Parameters:
    -----------
    x : scalar or array_like
        The input value(s) to compute the hyperbolic sine of.
    tol : float, optional
        The tolerance for series convergence (default is 1e-10).

    Returns:
    --------
    out : scalar or ndarray
        The hyperbolic sine of x.

    Examples:
    --------
    >>> sinh(0)
    0.0
    >>> sinh(1)
    1.1752011936438014
    >>> sinh(np.array([-1, 0, 1]))
    array([-1.17520119,  0.        ,  1.17520119])
    """
    x = np.asarray(x)
    is_scalar = x.ndim == 0
    if is_scalar:
        x = x[None]
    sum_sinh = np.zeros_like(x, dtype=float)
    term = x
    n = 0
    while np.max(np.abs(term)) >= tol:
        sum_sinh += term
        n += 1
        term = term * x**2 / ((2 * n) * (2 * n + 1))
    return sum_sinh[0] if is_scalar else sum_sinh


def cosh(x, tol=1e-10):
    """
    Compute the hyperbolic cosine using its Taylor series expansion.

    The Taylor series for cosh(x) is: cosh(x) = sum_{n=0}^∞ x^(2n) / (2n)!

    Parameters:
    -----------
    x : scalar or array_like
        The input value(s) to compute the hyperbolic cosine of.
    tol : float, optional
        The tolerance for series convergence (default is 1e-10).

    Returns:
    --------
    out : scalar or ndarray
        The hyperbolic cosine of x.

    Examples:
    --------
    >>> cosh(0)
    1.0
    >>> cosh(1)
    1.5430806348152437
    >>> cosh(np.array([0, 1, 2]))
    array([1.        , 1.54308063, 3.76219569])
    """
    x = np.asarray(x)
    is_scalar = x.ndim == 0
    if is_scalar:
        x = x[None]
    sum_cosh = np.zeros_like(x, dtype=float)
    term = np.ones_like(x, dtype=float)
    n = 0
    while np.max(np.abs(term)) >= tol:
        sum_cosh += term
        n += 1
        term = term * x**2 / ((2 * n - 1) * (2 * n))
    return sum_cosh[0] if is_scalar else sum_cosh


def tanh(x, tol=1e-10):
    """
    Compute the hyperbolic tangent as sinh(x) / cosh(x).

    Uses the Taylor series expansions of sinh(x) and cosh(x) to compute tanh(x).

    Parameters:
    -----------
    x : scalar or array_like
        The input value(s) to compute the hyperbolic tangent of.
    tol : float, optional
        The tolerance for series convergence (default is 1e-10).

    Returns:
    --------
    out : scalar or ndarray
        The hyperbolic tangent of x.

    Examples:
    --------
    >>> tanh(0)
    0.0
    >>> tanh(1)
    0.7615941559557649
    >>> tanh(np.array([0, 1, 2]))
    array([0.        , 0.76159416, 0.96402758])
    """
    return sinh(x, tol) / cosh(x, tol)


def factorial(n):
    """
    Compute the factorial of non-negative integers.

    Parameters
    ----------
    n : int or array_like of ints
        The input value(s) to compute the factorial of.

    Returns
    -------
    out : int or ndarray
        The factorial of n.

    Raises
    ------
    ValueError
        If n contains negative numbers or non-integers.

    Examples
    --------
    >>> factorial(5)
    120
    >>> factorial([0, 1, 2, 3])
    array([1, 1, 2, 6])
    """
    n = np.asarray(n)
    if np.any(n < 0) or np.any(n != n.astype(int)):
        raise ValueError("Factorial is only defined for non-negative integers.")
    if n.ndim == 0:  # Scalar case
        return 1 if n == 0 else np.prod(np.arange(1, n + 1))
    else:  # Array case
        max_n = np.max(n)
        if max_n == 0:
            return np.ones_like(n)
        factorials = [1]
        for i in range(1, max_n + 1):
            factorials.append(factorials[-1] * i)
        return np.array(factorials)[n]


def simpson_rule(f, a, b, N):
    """
    Approximate the integral of f from a to b using Simpson's rule.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a, b : float
        Lower and upper integration limits.
    N : int
        Number of intervals (must be even).

    Returns
    -------
    float
        Approximate integral value.
    """
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's rule.")
    dt = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    weights = np.ones(N + 1)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    return np.sum(y * weights) * dt / 3


def gamma(z, T=100, N=1000, epsilon=1e-10):
    """
    Compute the gamma function for z > 0.

    Parameters
    ----------
    z : scalar or array_like
        The input value(s) to compute the gamma function of.
    T : float, optional
        Upper limit for integration (default 100).
    N : int, optional
        Number of intervals for Simpson's rule (default 1000).
    epsilon : float, optional
        Lower limit to handle singularity for z < 1 (default 1e-10).

    Returns
    -------
    out : scalar or ndarray
        The gamma function of z.

    Raises
    ------
    ValueError
        If any z <= 0.

    Examples
    --------
    >>> gamma(1)
    1.0
    >>> gamma(0.5)
    1.7724538509055159
    >>> gamma([1, 2, 3])
    array([1., 1., 2.])
    """
    z = np.atleast_1d(z)
    if np.any(z <= 0):
        raise ValueError("Gamma function is only defined for z > 0.")
    t = np.linspace(epsilon, T, N + 1)
    dt = (T - epsilon) / N
    integrand = t[np.newaxis, :] ** (z[:, np.newaxis] - 1) * np.exp(-t[np.newaxis, :])
    weights = np.ones(N + 1)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    weights *= dt / 3
    integral = np.sum(integrand * weights, axis=1)
    contrib = np.where(z < 1, epsilon**z / z, 0)
    result = integral + contrib
    return result[0] if z.shape == (1,) else result


def bessel_j(alpha, x, tol=1e-10, max_iter=1000):
    """
    Compute the Bessel function of the first kind J_alpha(x).

    Parameters
    ----------
    alpha : scalar
        The order of the Bessel function.
    x : scalar or array_like
        The input value(s) to compute the Bessel function at.
    tol : float, optional
        Tolerance for series convergence (default 1e-10).
    max_iter : int, optional
        Maximum number of iterations (default 1000).

    Returns
    -------
    out : scalar or ndarray
        The Bessel function J_alpha(x).

    Examples
    --------
    >>> bessel_j(0, 0)
    1.0
    >>> bessel_j(1, 0)
    0.0
    >>> bessel_j(0, np.array([0, 1, 2]))
    array([1.        , 0.76519769, 0.22389078])
    """
    x = np.atleast_1d(x)
    sum_j = np.zeros_like(x, dtype=float)
    term = (x / 2) ** alpha / gamma(alpha + 1)
    m = 0
    while np.max(np.abs(term)) >= tol and m < max_iter:
        sum_j += term
        m += 1
        term = term * (-((x / 2) ** 2)) / (m * (m + alpha))
    return sum_j[0] if x.shape == (1,) else sum_j
