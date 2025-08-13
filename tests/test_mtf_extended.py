# Tests for the MTFExtended module
import pytest
import numpy as np
import pandas as pd
import MTFLibrary
from MTFLibrary import MTFExtended
from MTFLibrary import initialize_mtf_globals, set_global_etol, get_global_max_dimension
from MTFLibrary.MTFExtended import MultivariateTaylorFunction, Var, compose, mtfarray
from MTFLibrary.taylor_function import MultivariateTaylorFunctionBase

@pytest.fixture(scope="function", autouse=True)
def setup_function():
    """Initialize MTF globals for each test in this module."""
    initialize_mtf_globals(max_order=5, max_dimension=3)
    set_global_etol(1e-12)
    yield
    MTFLibrary.taylor_function._INITIALIZED = False

def test_array_ufunc():
    """
    Test the __array_ufunc__ implementation with a numpy ufunc.
    """
    x = Var(1)
    y = np.sin(x)

    # The result should be a MultivariateTaylorFunction object
    from MTFLibrary.taylor_function import MultivariateTaylorFunctionBase
    assert isinstance(y, MultivariateTaylorFunctionBase)

    # Check if the result is correct by evaluating at a point
    eval_point = [0.1, 0, 0]
    numerical_result = y.eval(eval_point)
    analytical_result = np.sin(eval_point[0])

    assert np.isclose(numerical_result[0], analytical_result, rtol=1e-5)

def test_compose():
    """
    Test the compose function.
    """
    x = Var(1)
    y = Var(2)

    f = x**2 + y
    g = x + 1

    # Compose f with g, replacing x with g
    h = compose(f, {1: g})

    # The result should be (x+1)^2 + y = x^2 + 2x + 1 + y

    # Evaluate at a point
    eval_point = [1, 1, 0]
    numerical_result = h.eval(eval_point)
    analytical_result = (eval_point[0] + 1)**2 + eval_point[1]

    assert np.isclose(numerical_result[0], analytical_result)

def test_mtfarray():
    """
    Test the mtfarray function.
    """
    x = Var(1)
    y = Var(2)

    mtf1 = x + 2*y
    mtf2 = x**2

    df = mtfarray([mtf1, mtf2], column_names=['f1', 'f2'])

    assert isinstance(df, pd.DataFrame)
    assert 'Coeff_f1' in df.columns
    assert 'Coeff_f2' in df.columns
    assert 'Order' in df.columns
    assert 'Exponents' in df.columns

    # Check the number of rows.
    # mtf1 has terms (1,0,0) and (0,1,0)
    # mtf2 has term (2,0,0)
    # The constant term (0,0,0) is also present.
    # So there should be 4 rows.
    # Actually, the constant term is not always present if the coefficient is zero.
    # Let's check the number of non-zero coefficients.
    num_coeffs = mtf1.coeffs.size + mtf2.coeffs.size
    # There might be overlapping coefficients, but in this case there are not.
    # The mtfarray function will merge the terms, so the number of rows will be the number of unique exponents.
    # mtf1 has terms (1,0,0) and (0,1,0). mtf2 has term (2,0,0).
    # The exponents are (1,0,0), (0,1,0), (2,0,0). So there should be 3 rows.
    # Let's get the number of unique exponents.
    all_exponents = np.vstack([mtf1.exponents, mtf2.exponents])
    num_unique_exponents = len(np.unique(all_exponents, axis=0))
    assert len(df) == num_unique_exponents
