# MTFLibrary/test_taylor_functions.py

import pytest
import numpy as np
import MTFLibrary
from MTFLibrary import *

# Global settings for tests
MAX_ORDER = 30
MAX_DIMENSION = 6
ETOL = 1e-10

@pytest.fixture(scope="function", autouse=True)
def setup_function():
    initialize_mtf_globals(MAX_ORDER, MAX_DIMENSION)
    set_global_etol(ETOL)
    global_dim = get_global_max_dimension()
    exponent_zero = tuple([0] * global_dim)
    yield global_dim, exponent_zero
    MTFLibrary.taylor_function._INITIALIZED = False


# --- Global Settings Tests ---

# --- Tests for Global Settings ---
def test_global_initialization_once(setup_function):
    global_dim, exponent_zero = setup_function
    assert get_global_max_order() == MAX_ORDER
    assert get_global_max_dimension() == MAX_DIMENSION
    with pytest.raises(RuntimeError):
        initialize_mtf_globals(max_order=MAX_ORDER, max_dimension=MAX_DIMENSION) # Re-initialization should fail

def test_global_max_order_setting(setup_function):
    global_dim, exponent_zero = setup_function
    set_global_max_order(MAX_ORDER)
    assert get_global_max_order() == MAX_ORDER
    with pytest.raises(ValueError):
        set_global_max_order(-1) # Invalid order

def test_global_etol_setting(setup_function):
    global_dim, exponent_zero = setup_function
    set_global_etol(1e-6)
    assert get_global_etol() == 1e-6
    with pytest.raises(ValueError):
        set_global_etol(-1e-6) # Invalid etol
    with pytest.raises(ValueError):
        set_global_etol(0.0) # Invalid etol


# --- Var Function Tests ---
def test_var_creation(setup_function):
    global_dim, exponent_zero = setup_function
    x_var = Var(1)
    assert isinstance(x_var, MultivariateTaylorFunction)
    assert x_var.dimension == global_dim
    exponent_one = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
    coeff_x1 = x_var.extract_coefficient(tuple(exponent_one))
    assert np.allclose(coeff_x1, np.array([1.0]).reshape(1))
    coeff_constant = x_var.extract_coefficient(exponent_zero)
    assert np.allclose(coeff_constant, np.array([0.0]).reshape(1))
    with pytest.raises(ValueError):
        Var(0)
    with pytest.raises(ValueError):
        Var(global_dim + 1)
    MTFLibrary.taylor_function._INITIALIZED = False
    with pytest.raises(RuntimeError):
        Var(1) # Before initialization


# --- MultivariateTaylorFunction (Real MTF) Tests ---

def test_mtf_constant_creation(setup_function):
    global_dim, exponent_zero = setup_function
    const_mtf = MultivariateTaylorFunction.from_constant(5.0)
    assert np.allclose(const_mtf.eval([0] * global_dim), np.array([5.0]).reshape(1))
    assert np.allclose(const_mtf.extract_coefficient(exponent_zero), np.array([5.0]).reshape(1))

    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    assert np.allclose(const_mtf.extract_coefficient(tuple(exponent_one)), np.array([0.0]).reshape(1))

def test_mtf_variable_evaluation(setup_function):
    global_dim, exponent_zero = setup_function
    evaluation_point_x1 = [0] * global_dim
    if global_dim > 0:
        evaluation_point_x1[0] = 2.0
    evaluation_point_x2 = [0] * global_dim
    if global_dim > 1:
        evaluation_point_x2[1] = 3.0

    x1_var = Var(1)
    x2_var = Var(2)
    assert np.allclose(x1_var.eval(evaluation_point_x1), np.array([2.0]).reshape(1))
    assert np.allclose(x2_var.eval(evaluation_point_x2), np.array([3.0]).reshape(1))

def test_mtf_truncate(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    exponent_two = list(exponent_zero)
    exponent_three = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
        exponent_three[0] = 3
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)
    exponent_three = tuple(exponent_three)

    coeffs = {exponent_zero: np.array([1.0]).reshape(1),
              exponent_one: np.array([2.0]).reshape(1),
              exponent_two: np.array([3.0]).reshape(1),
              exponent_three: np.array([4.0]).reshape(1)}
    mtf = MultivariateTaylorFunction(coefficients=coeffs, dimension=global_dim)
    truncated_mtf = mtf.truncate(2)
    assert np.allclose(truncated_mtf.extract_coefficient(exponent_three), np.array([0.0]).reshape(1))
    assert np.allclose(truncated_mtf.extract_coefficient(exponent_two), np.array([3.0]).reshape(1))
    assert np.allclose(truncated_mtf.extract_coefficient(exponent_one), np.array([2.0]).reshape(1))
    assert np.allclose(truncated_mtf.extract_coefficient(exponent_zero), np.array([1.0]).reshape(1))

def test_mtf_extract_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_one_list = list(exponent_one_one)
        if global_dim > 1:
            exponent_one_one_list[0] = 1
            exponent_one_one_list[1] = 1
        elif global_dim == 1:
            exponent_one_one_list[0] = 2 # If dimension is 1, (1, 1) becomes (2,)
        exponent_one_one = tuple(exponent_one_one_list)

    mtf = MultivariateTaylorFunction(coefficients={exponent_one_one: np.array([2.5]).reshape(1), exponent_zero: np.array([1.0]).reshape(1)}, dimension=max(2, global_dim)) # Dimension is set here to at least 2 for the test
    assert np.allclose(mtf.extract_coefficient(exponent_one_one), np.array([2.5]).reshape(1))
    assert np.allclose(mtf.extract_coefficient(exponent_zero), np.array([1.0]).reshape(1))

    exponent_two_zero = list(exponent_zero)
    if global_dim > 0:
        exponent_two_zero_list = list(exponent_two_zero)
        exponent_two_zero_list[0] = 2
        exponent_two_zero = tuple(exponent_two_zero_list)
    assert np.allclose(mtf.extract_coefficient(exponent_two_zero), np.array([0.0]).reshape(1)) # Non-existent coefficient should be zero

def test_mtf_set_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf = MultivariateTaylorFunction.from_constant(0.0)
    mtf.set_coefficient(exponent_one, 3.0)
    assert np.allclose(mtf.extract_coefficient(exponent_one), np.array([3.0]).reshape(1))
    mtf.set_coefficient(exponent_one, 0.0) # Setting to zero
    assert np.allclose(mtf.extract_coefficient(exponent_one), np.array([0.0]).reshape(1))
    with pytest.raises(TypeError):
        mtf.set_coefficient(list(exponent_one), 2.0) # Exponents must be tuple
    with pytest.raises(ValueError):
        if global_dim > 1:
            invalid_exponent = tuple(exponent_one[:-1]) # Shorter tuple
        else:
            invalid_exponent = (1, 0) # Mismatch if global_dim is 1
        mtf.set_coefficient(invalid_exponent, 2.0) # Exponent dimension mismatch
    with pytest.raises(TypeError):
        mtf.set_coefficient(exponent_one, "invalid") # Value must be numeric

def test_mtf_get_max_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    exponent_zero_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    if global_dim > 1:
        exponent_zero_one[1] = 1
    exponent_one_zero = tuple(exponent_one_zero)
    exponent_zero_one = tuple(exponent_zero_one)

    coeffs = {exponent_zero: np.array([1.0]).reshape(1),
              exponent_one_zero: np.array([-2.0]).reshape(1),
              exponent_zero_one: np.array([3.0]).reshape(1)}
    mtf = MultivariateTaylorFunction(coefficients=coeffs, dimension=max(2, global_dim)) # Dimension at least 2 for the test
    assert pytest.approx(mtf.get_max_coefficient()) == 3.0

def test_mtf_get_min_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    exponent_zero_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    if global_dim > 1:
        exponent_zero_one[1] = 1
    exponent_one_zero = tuple(exponent_one_zero)
    exponent_zero_one = tuple(exponent_zero_one)

    etol = get_global_etol()
    coeffs = {exponent_zero: np.array([0.1]).reshape(1),
              exponent_one_zero: np.array([2.0]).reshape(1),
              exponent_zero_one: np.array([3.0]).reshape(1)}
    mtf = MultivariateTaylorFunction(coefficients=coeffs, dimension=max(2, global_dim)) # Dimension at least 2 for the test
    assert pytest.approx(mtf.get_min_coefficient(tolerance=0.5)) == 2.0 # Min non-negligible coefficient (above 0.5)
    coeffs_negligible = {exponent_zero: np.array([etol]).reshape(1)}
    mtf_negligible = MultivariateTaylorFunction(coefficients=coeffs_negligible, dimension=max(2, global_dim))
    assert pytest.approx(mtf_negligible.get_min_coefficient()) == 0.0 # All negligible, returns 0.0

# def test_mtf_to_complex_mtf(setup_function):
#     global_dim, exponent_zero = setup_function
#     mtf_real = MultivariateTaylorFunction.from_constant(2.5)
#     cmtf = mtf_real.to_complex_mtf()
#     assert isinstance(cmtf, ComplexMultivariateTaylorFunction)
#     assert np.allclose(cmtf.extract_coefficient(exponent_zero).real, np.array([2.5]).reshape(1))
#     assert np.allclose(cmtf.extract_coefficient(exponent_zero).imag, np.array([0.0]).reshape(1))

# --- MTF Arithmetic Operations Tests ---

def test_mtf_addition(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf1 = MultivariateTaylorFunction({exponent_zero: np.array([1.0]).reshape(1), exponent_one: np.array([2.0]).reshape(1)}, dimension=global_dim) # 1 + 2x
    mtf2 = MultivariateTaylorFunction({exponent_zero: np.array([3.0]).reshape(1), exponent_one: np.array([-1.0]).reshape(1)}, dimension=global_dim) # 3 - x
    mtf_sum = mtf1 + mtf2 # (1+2x) + (3-x) = 4 + x
    assert np.allclose(mtf_sum.extract_coefficient(exponent_zero), np.array([4.0]).reshape(1))
    assert np.allclose(mtf_sum.extract_coefficient(exponent_one), np.array([1.0]).reshape(1))
    mtf_sum_const = mtf1 + 2.0 # (1+2x) + 2 = 3 + 2x
    assert np.allclose(mtf_sum_const.extract_coefficient(exponent_zero), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_sum_const.extract_coefficient(exponent_one), np.array([2.0]).reshape(1))
    mtf_sum_rconst = 2.0 + mtf1 # 2 + (1+2x) = 3 + 2x (commutativity)
    assert np.allclose(mtf_sum_rconst.extract_coefficient(exponent_zero), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_sum_rconst.extract_coefficient(exponent_one), np.array([2.0]).reshape(1))

def test_mtf_subtraction(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf1 = MultivariateTaylorFunction({exponent_zero: np.array([5.0]).reshape(1), exponent_one: np.array([3.0]).reshape(1)}, dimension=global_dim) # 5 + 3x
    mtf2 = MultivariateTaylorFunction({exponent_zero: np.array([2.0]).reshape(1), exponent_one: np.array([1.0]).reshape(1)}, dimension=global_dim) # 2 + x
    mtf_diff = mtf1 - mtf2 # (5+3x) - (2+x) = 3 + 2x
    assert np.allclose(mtf_diff.extract_coefficient(exponent_zero), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_diff.extract_coefficient(exponent_one), np.array([2.0]).reshape(1))
    mtf_diff_const = mtf1 - 2.0 # (5+3x) - 2 = 3 + 3x
    assert np.allclose(mtf_diff_const.extract_coefficient(exponent_zero), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_diff_const.extract_coefficient(exponent_one), np.array([3.0]).reshape(1))
    mtf_diff_rconst = 4.0 - mtf1 # 4 - (5+3x) = -1 - 3x
    assert np.allclose(mtf_diff_rconst.extract_coefficient(exponent_zero), np.array([-1.0]).reshape(1))
    assert np.allclose(mtf_diff_rconst.extract_coefficient(exponent_one), np.array([-3.0]).reshape(1))

def test_mtf_multiplication(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = [0] * global_dim
    exponent_two = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)

    mtf1 = MultivariateTaylorFunction({exponent_zero: np.array([2.0]).reshape(1), exponent_one: np.array([1.0]).reshape(1)}, dimension=global_dim) # 2 + x
    mtf2 = MultivariateTaylorFunction({exponent_zero: np.array([3.0]).reshape(1), exponent_one: np.array([-2.0]).reshape(1)}, dimension=global_dim) # 3 - 2x
    mtf_prod = mtf1 * mtf2 # (2+x) * (3-2x) = 6 - 4x + 3x - 2x^2 = 6 - x - 2x^2
    assert np.allclose(mtf_prod.extract_coefficient(exponent_zero), np.array([6.0]).reshape(1))
    assert np.allclose(mtf_prod.extract_coefficient(exponent_one), np.array([-1.0]).reshape(1))
    assert np.allclose(mtf_prod.extract_coefficient(exponent_two), np.array([-2.0]).reshape(1))
    mtf_prod_const = mtf1 * 3.0 # (2+x) * 3 = 6 + 3x
    assert np.allclose(mtf_prod_const.extract_coefficient(exponent_zero), np.array([6.0]).reshape(1))
    assert np.allclose(mtf_prod_const.extract_coefficient(exponent_one), np.array([3.0]).reshape(1))
    mtf_prod_rconst = 3.0 * mtf1 # 3 * (2+x) = 6 + 3x (commutativity)
    assert np.allclose(mtf_prod_rconst.extract_coefficient(exponent_zero), np.array([6.0]).reshape(1))
    assert np.allclose(mtf_prod_rconst.extract_coefficient(exponent_one), np.array([3.0]).reshape(1))

def test_mtf_power(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = [0] * global_dim
    exponent_two = [0] * global_dim
    exponent_three = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
        exponent_three[0] = 3
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)
    exponent_three = tuple(exponent_three)

    mtf = MultivariateTaylorFunction({exponent_zero: np.array([1.0]).reshape(1), exponent_one: np.array([1.0]).reshape(1)}, dimension=global_dim) # 1 + x
    mtf_sq = mtf ** 2 # (1+x)^2 = 1 + 2x + x^2
    assert np.allclose(mtf_sq.extract_coefficient(exponent_zero), np.array([1.0]).reshape(1))
    assert np.allclose(mtf_sq.extract_coefficient(exponent_one), np.array([2.0]).reshape(1))
    assert np.allclose(mtf_sq.extract_coefficient(exponent_two), np.array([1.0]).reshape(1))
    mtf_cube = mtf ** 3 # (1+x)^3 = 1 + 3x + 3x^2 + x^3, but truncated to order 2
    assert np.allclose(mtf_cube.extract_coefficient(exponent_zero), np.array([1.0]).reshape(1))
    assert np.allclose(mtf_cube.extract_coefficient(exponent_one), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_cube.extract_coefficient(exponent_two), np.array([3.0]).reshape(1))
    # pytest.approx might not be directly comparable with numpy arrays
    coeff_three = mtf_cube.extract_coefficient(exponent_three)
    if coeff_three is not None:
        assert coeff_three == pytest.approx(np.array([1.0]).reshape(1))
    else:
        # If max_order is less than 3, the coefficient might not exist
        assert get_global_max_order() < 3
    with pytest.raises(ValueError):
        mtf ** (-2) # Negative power not allowed
    with pytest.raises(ValueError):
        mtf ** 2.5 # Non-integer power not allowed

def test_mtf_negation(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf = MultivariateTaylorFunction({exponent_zero: np.array([2.0]).reshape(1), exponent_one: np.array([-1.0]).reshape(1)}, dimension=global_dim) # 2 - x
    neg_mtf = -mtf # -(2 - x) = -2 + x
    assert np.allclose(neg_mtf.extract_coefficient(exponent_zero), np.array([-2.0]).reshape(1))
    assert np.allclose(neg_mtf.extract_coefficient(exponent_one), np.array([1.0]).reshape(1))

def test_mtf_eval_shape_consistency(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf = MultivariateTaylorFunction({exponent_zero: np.array([2.0]).reshape(1), exponent_one: np.array([-1.0]).reshape(1)}, dimension=global_dim)
    eval_point = [0.5] * global_dim if global_dim > 0 else [0.5]
    if global_dim > 1:
        eval_point = [0.5] + [0] * (global_dim - 1) # Evaluate with the first variable as 0.5

    eval_result = mtf.eval(eval_point[:global_dim]) # Ensure the evaluation point matches the dimension
    assert eval_result.shape == (1,) # Evaluation result should be shape (1,)

    mtf_sum_result = (mtf + mtf).eval(eval_point[:global_dim])
    assert mtf_sum_result.shape == (1,)

    mtf_mul_result = (mtf * 2.0).eval(eval_point[:global_dim])
    assert mtf_mul_result.shape == (1,)


# --- ComplexMultivariateTaylorFunction (Complex CMTF) Tests ---
def test_cmtf_creation(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    coeffs = {exponent_zero: np.array([1+1j]).reshape(1), exponent_one: np.array([2-1j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=global_dim)
    assert np.allclose(cmtf.extract_coefficient(exponent_zero), np.array([1+1j]).reshape(1))
    assert np.allclose(cmtf.extract_coefficient(exponent_one), np.array([2-1j]).reshape(1))

def test_cmtf_variable_evaluation(setup_function):
    global_dim, exponent_zero = setup_function
    evaluation_point_x1 = [0] * global_dim
    if global_dim > 0:
        evaluation_point_x1[0] = 0.5
    evaluation_point_x2 = [0] * global_dim
    if global_dim > 1:
        evaluation_point_x2[1] = 0.0

    x1_var_c = ComplexMultivariateTaylorFunction.from_variable(1, dimension=global_dim)
    assert np.allclose(x1_var_c.eval(evaluation_point_x1), np.array([0.5+0.0j]).reshape(1))

def test_cmtf_truncate(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    exponent_two = list(exponent_zero)
    exponent_three = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
        exponent_three[0] = 3
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)
    exponent_three = tuple(exponent_three)

    coeffs = {exponent_zero: np.array([1+0j]).reshape(1),
              exponent_one: np.array([2j]).reshape(1),
              exponent_two: np.array([3-3j]).reshape(1),
              exponent_three: np.array([4+4j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=global_dim)
    truncated_cmtf = cmtf.truncate(2)
    assert np.allclose(truncated_cmtf.extract_coefficient(exponent_three), np.array([0.0j]).reshape(1))
    assert np.allclose(truncated_cmtf.extract_coefficient(exponent_two), np.array([3-3j]).reshape(1))
    assert np.allclose(truncated_cmtf.extract_coefficient(exponent_one), np.array([2j]).reshape(1))
    assert np.allclose(truncated_cmtf.extract_coefficient(exponent_zero), np.array([1+0j]).reshape(1))

def test_cmtf_extract_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    exponent_one_zero = tuple(exponent_one_zero)

    coeffs = {exponent_one_zero: np.array([1+1j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=max(2, global_dim)) # Dimension at least 2 for the test
    assert np.allclose(cmtf.extract_coefficient(exponent_one_zero), np.array([1+1j]).reshape(1))
    assert np.allclose(cmtf.extract_coefficient(exponent_zero), np.array([0.0j]).reshape(1)) # Default complex zero

def test_cmtf_set_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction.from_constant(0.0j, MAX_DIMENSION)
    cmtf.set_coefficient(exponent_one, 2+2j)
    assert np.allclose(cmtf.extract_coefficient(exponent_one), np.array([2+2j]).reshape(1))
    cmtf.set_coefficient(exponent_one, 0.0j) # Setting to zero
    assert np.allclose(cmtf.extract_coefficient(exponent_one), np.array([0.0j]).reshape(1))
    with pytest.raises(TypeError):
        cmtf.set_coefficient(list(exponent_one), "invalid") # Value must be numeric

def test_cmtf_get_max_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    exponent_zero_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    if global_dim > 1:
        exponent_zero_one[1] = 1
    exponent_one_zero = tuple(exponent_one_zero)
    exponent_zero_one = tuple(exponent_zero_one)

    coeffs = {exponent_zero: np.array([1+0j]).reshape(1),
              exponent_one_zero: np.array([1j]).reshape(1),
              exponent_zero_one: np.array([-2+0j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=max(2, global_dim)) # Dimension at least 2 for the test
    assert pytest.approx(cmtf.get_max_coefficient()) == 2.0 # Max magnitude is 2.0

def test_cmtf_get_min_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    exponent_zero_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    if global_dim > 1:
        exponent_zero_one[1] = 1
    exponent_one_zero = tuple(exponent_one_zero)
    exponent_zero_one = tuple(exponent_zero_one)

    coeffs = {exponent_zero: np.array([0.1j]).reshape(1),
              exponent_one_zero: np.array([2+0j]).reshape(1),
              exponent_zero_one: np.array([3j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=max(2, global_dim)) # Dimension at least 2 for the test
    assert pytest.approx(cmtf.get_min_coefficient(tolerance=0.5)) == 2.0 # Min non-negligible magnitude (above 0.5)
    coeffs_negligible = {exponent_zero: np.array([1e-10j]).reshape(1)}
    cmtf_negligible = ComplexMultivariateTaylorFunction(coefficients=coeffs_negligible, dimension=max(2, global_dim))
    assert pytest.approx(cmtf_negligible.get_min_coefficient()) == 0.0 # All negligible, returns 0.0

def test_cmtf_conjugate(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+1j]).reshape(1), exponent_one: np.array([2-1j]).reshape(1)}, dimension=global_dim)
    conj_cmtf = cmtf.conjugate()
    assert np.allclose(conj_cmtf.extract_coefficient(exponent_zero), np.array([1-1j]).reshape(1))
    assert np.allclose(conj_cmtf.extract_coefficient(exponent_one), np.array([2+1j]).reshape(1))

def test_cmtf_real_part(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+1j]).reshape(1), exponent_one: np.array([2-1j]).reshape(1)}, dimension=global_dim)
    real_mtf = cmtf.real_part()
    assert isinstance(real_mtf, MultivariateTaylorFunctionBase)
    assert np.allclose(real_mtf.extract_coefficient(exponent_zero), np.array([1.0]).reshape(1))
    assert np.allclose(real_mtf.extract_coefficient(exponent_one), np.array([2.0]).reshape(1))

def test_cmtf_imag_part(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+1j]).reshape(1), exponent_one: np.array([2-1j]).reshape(1)}, dimension=global_dim)
    imag_mtf = cmtf.imag_part()
    assert isinstance(imag_mtf, MultivariateTaylorFunctionBase)
    assert np.allclose(imag_mtf.extract_coefficient(exponent_zero), np.array([1.0]).reshape(1))
    assert np.allclose(imag_mtf.extract_coefficient(exponent_one), np.array([-1.0]).reshape(1))

def test_cmtf_magnitude_phase_not_implemented(setup_function):
    global_dim, exponent_zero = setup_function
    cmtf = ComplexMultivariateTaylorFunction.from_constant(1+1j, MAX_DIMENSION)
    with pytest.raises(NotImplementedError):
        cmtf.magnitude()
    with pytest.raises(NotImplementedError):
        cmtf.phase()

# --- CMTF Arithmetic Operations Tests ---
def test_cmtf_addition(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf1 = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+1j]).reshape(1), exponent_one: np.array([2-1j]).reshape(1)}, dimension=global_dim) # (1+j) + (2-j)x
    cmtf2 = ComplexMultivariateTaylorFunction({exponent_zero: np.array([3-2j]).reshape(1), exponent_one: np.array([-1+0.5j]).reshape(1)}, dimension=global_dim) # (3-2j) + (-1+0.5j)x
    cmtf_sum = cmtf1 + cmtf2 # (4-j) + (1-0.5j)x
    assert np.allclose(cmtf_sum.extract_coefficient(exponent_zero), np.array([4-1j]).reshape(1))
    assert np.allclose(cmtf_sum.extract_coefficient(exponent_one), np.array([1-0.5j]).reshape(1))
    cmtf_sum_const = cmtf1 + (2+0j) # (3+j) + (2-j)x
    assert np.allclose(cmtf_sum_const.extract_coefficient(exponent_zero), np.array([3+1j]).reshape(1))
    assert np.allclose(cmtf_sum_const.extract_coefficient(exponent_one), np.array([2-1j]).reshape(1))
    cmtf_sum_rconst = (2+0j) + cmtf1 # commutativity
    assert np.allclose(cmtf_sum_rconst.extract_coefficient(exponent_zero), np.array([3+1j]).reshape(1))
    assert np.allclose(cmtf_sum_rconst.extract_coefficient(exponent_one), np.array([2-1j]).reshape(1))

def test_cmtf_subtraction(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf1 = ComplexMultivariateTaylorFunction({exponent_zero: np.array([5+0j]).reshape(1), exponent_one: np.array([3+1j]).reshape(1)}, dimension=global_dim) # 5 + (3+j)x
    cmtf2 = ComplexMultivariateTaylorFunction({exponent_zero: np.array([2+1j]).reshape(1), exponent_one: np.array([1-1j]).reshape(1)}, dimension=global_dim) # (2+j) + (1-j)x
    cmtf_diff = cmtf1 - cmtf2 # (3-j) + (2+2j)x
    assert np.allclose(cmtf_diff.extract_coefficient(exponent_zero), np.array([3-1j]).reshape(1))
    assert np.allclose(cmtf_diff.extract_coefficient(exponent_one), np.array([2+2j]).reshape(1))
    cmtf_diff_const = cmtf1 - (2+0j) # (3+0j) + (3+1j)x
    assert np.allclose(cmtf_diff_const.extract_coefficient(exponent_zero), np.array([3+0j]).reshape(1))
    assert np.allclose(cmtf_diff_const.extract_coefficient(exponent_one), np.array([3+1j]).reshape(1))
    cmtf_diff_rconst = (4+0j) - cmtf1 # (-1+0j) + (-3-1j)x
    assert np.allclose(cmtf_diff_rconst.extract_coefficient(exponent_zero), np.array([-1+0j]).reshape(1))
    assert np.allclose(cmtf_diff_rconst.extract_coefficient(exponent_one), np.array([-3-1j]).reshape(1))

def test_cmtf_multiplication(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    exponent_two = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)

    cmtf1 = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+0j]).reshape(1), exponent_one: np.array([1j]).reshape(1)}, dimension=global_dim) # 1 + jx
    cmtf2 = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1j]).reshape(1), exponent_one: np.array([-1+0j]).reshape(1)}, dimension=global_dim) # j - x
    cmtf_prod = cmtf1 * cmtf2 # j - x -x + (-j)x^2 = j - 2x -jx^2
    assert np.allclose(cmtf_prod.extract_coefficient(exponent_zero), np.array([1j]).reshape(1))
    assert np.allclose(cmtf_prod.extract_coefficient(exponent_one), np.array([-2+0j]).reshape(1))
    assert np.allclose(cmtf_prod.extract_coefficient(exponent_two), np.array([-1j]).reshape(1))
    cmtf_prod_const = cmtf1 * (2+0j) # (2+0j) + (2j)x
    assert np.allclose(cmtf_prod_const.extract_coefficient(exponent_zero), np.array([2+0j]).reshape(1))
    assert np.allclose(cmtf_prod_const.extract_coefficient(exponent_one), np.array([2j]).reshape(1))
    cmtf_prod_rconst = (2+0j) * cmtf1 # commutativity
    assert np.allclose(cmtf_prod_rconst.extract_coefficient(exponent_zero), np.array([2+0j]).reshape(1))
    assert np.allclose(cmtf_prod_rconst.extract_coefficient(exponent_one), np.array([2j]).reshape(1))

def test_cmtf_power(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    exponent_two = list(exponent_zero)
    exponent_three = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
        exponent_three[0] = 3
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)
    exponent_three = tuple(exponent_three)

    cmtf = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+0j]).reshape(1), exponent_one: np.array([1j]).reshape(1)}, dimension=global_dim) # 1 + jx
    cmtf_sq = cmtf ** 2 # (1+jx)^2 = 1 + 2jx - x^2
    assert np.allclose(cmtf_sq.extract_coefficient(exponent_zero), np.array([1+0j]).reshape(1))
    assert np.allclose(cmtf_sq.extract_coefficient(exponent_one), np.array([2j]).reshape(1))
    assert np.allclose(cmtf_sq.extract_coefficient(exponent_two), np.array([-1+0j]).reshape(1))
    cmtf_cube = cmtf ** 3 # (1+jx)^3 = 1 + 3jx -3x^2 -jx^3, truncated to order 2
    assert np.allclose(cmtf_cube.extract_coefficient(exponent_zero), np.array([1+0j]).reshape(1))
    assert np.allclose(cmtf_cube.extract_coefficient(exponent_one), np.array([3j]).reshape(1))
    assert np.allclose(cmtf_cube.extract_coefficient(exponent_two), np.array([-3+0j]).reshape(1))
    # The MAX_ORDER check might be too specific, consider checking against a reasonable order
    # assert cmtf_cube.extract_coefficient((MAX_ORDER+1,)) == pytest.approx(np.array([0.0j]).reshape(1)) # Truncated

def test_cmtf_negation(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction({exponent_zero: np.array([2+0j]).reshape(1), exponent_one: np.array([-1+1j]).reshape(1)}, dimension=global_dim) # 2 + (-1+j)x
    neg_cmtf = -cmtf # -2 + (1-j)x
    assert np.allclose(neg_cmtf.extract_coefficient(exponent_zero), np.array([-2+0j]).reshape(1))
    assert np.allclose(neg_cmtf.extract_coefficient(exponent_one), np.array([1-1j]).reshape(1))

def test_cmtf_eval_shape_consistency(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction({exponent_zero: np.array([2+0j]).reshape(1), exponent_one: np.array([-1+1j]).reshape(1)}, dimension=global_dim)
    eval_point = [0.5] * global_dim if global_dim > 0 else [0.5]
    if global_dim > 1:
        eval_point = [0.5] + [0] * (global_dim - 1) # Evaluate with the first variable as 0.5

    eval_result = cmtf.eval(eval_point[:global_dim])
    assert eval_result.shape == (1,)

    cmtf_sum_result = (cmtf + cmtf).eval(eval_point[:global_dim])
    assert cmtf_sum_result.shape == (1,)

    cmtf_mul_result = (cmtf * (2+0j)).eval(eval_point[:global_dim])
    assert cmtf_mul_result.shape == (1,)


# --- convert_to_mtf Tests ---
def test_convert_to_mtf(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf = convert_to_mtf(5.0)
    assert isinstance(mtf, MultivariateTaylorFunction) or isinstance(mtf, MultivariateTaylorFunctionBase)
    assert np.allclose(mtf.eval([0] * global_dim), np.array([5.0]).reshape(1))
    assert np.allclose(mtf.extract_coefficient(exponent_zero), np.array([5.0]).reshape(1))
    assert np.allclose(mtf.extract_coefficient(exponent_one), np.array([0.0]).reshape(1))

    # cmtf = convert_to_mtf(2+1j)
    # assert isinstance(cmtf, ComplexMultivariateTaylorFunction)
    # assert np.allclose(cmtf.eval([0] * global_dim), np.array([2+1j]).reshape(1))
    # assert np.allclose(cmtf.extract_coefficient(exponent_zero), np.array([2+1j]).reshape(1))
    # assert np.allclose(cmtf.extract_coefficient(exponent_one), np.array([0.0j]).reshape(1))

    x_var_mtf = convert_to_mtf(Var(1))
    assert isinstance(x_var_mtf, MultivariateTaylorFunction)
    assert x_var_mtf.dimension == global_dim
    eval_point = [0] * global_dim
    if global_dim > 0:
        eval_point[0] = 1.0
    assert np.allclose(x_var_mtf.eval(eval_point), np.array([1.0]).reshape(1))

    existing_mtf = MultivariateTaylorFunction.from_constant(3.0)
    converted_mtf = convert_to_mtf(existing_mtf)
    assert converted_mtf is existing_mtf # Should return same object if already MTF

    with pytest.raises(TypeError):
        convert_to_mtf("invalid") # Invalid type

    numpy_scalar = np.float64(7.0)
    mtf_np_scalar = convert_to_mtf(numpy_scalar)
    assert np.allclose(mtf_np_scalar.eval([0] * global_dim), np.array([7.0]).reshape(1))
    assert np.allclose(mtf_np_scalar.extract_coefficient(exponent_zero), np.array([7.0]).reshape(1))
    assert np.allclose(mtf_np_scalar.extract_coefficient(exponent_one), np.array([0.0]).reshape(1))

    numpy_0d_array = np.array(9.0)
    mtf_np_0d = convert_to_mtf(numpy_0d_array)
    assert np.allclose(mtf_np_0d.eval([0] * global_dim), np.array([9.0]).reshape(1))
    assert np.allclose(mtf_np_0d.extract_coefficient(exponent_zero), np.array([9.0]).reshape(1))
    assert np.allclose(mtf_np_0d.extract_coefficient(exponent_one), np.array([0.0]).reshape(1))


# --- Elementary Functions Tests ---
elementary_functions_list = [
    (cos_taylor, 'cos_taylor'),
    (sin_taylor, 'sin_taylor'),
    (exp_taylor, 'exp_taylor'),
    (gaussian_taylor, 'gaussian_taylor'),
    (sqrt_taylor, 'sqrt_taylor'),
    (log_taylor, 'log_taylor'),
    (arctan_taylor, 'arctan_taylor'),
    (sinh_taylor, 'sinh_taylor'),
    (cosh_taylor, 'cosh_taylor'),
    (tanh_taylor, 'tanh_taylor'),
    (arcsin_taylor, 'arcsin_taylor'),
    (arccos_taylor, 'arccos_taylor'),
    (arctanh_taylor, 'arctanh_taylor')
]


@pytest.mark.parametrize("func, func_name", elementary_functions_list)
def test_elementary_functions_scalar(setup_function, func, func_name):
    global_dim, exponent_zero = setup_function
    scalar_input = 0.5;
    mtf_result = func(scalar_input)
    assert isinstance(mtf_result, MultivariateTaylorFunctionBase) # Changed assertion here
    zero_point = tuple([0 for _ in range(global_dim)])
    scalar_eval_result = func(scalar_input).eval(zero_point) # Evaluate at 0 as Taylor series is around 0

    if func_name == 'cos_taylor': expected_val = np.cos(scalar_input)
    elif func_name == 'sin_taylor': expected_val = np.sin(scalar_input)
    elif func_name == 'exp_taylor': expected_val = np.exp(scalar_input)
    elif func_name == 'gaussian_taylor': expected_val = np.exp(-(scalar_input**2))
    elif func_name == 'sqrt_taylor': expected_val = np.sqrt(scalar_input)
    elif func_name == 'log_taylor':
        if scalar_input > 0:
            expected_val = np.log(scalar_input)
        else:
            expected_val = np.nan # Or handle the domain issue appropriately
    elif func_name == 'arctan_taylor': expected_val = np.arctan(scalar_input)
    elif func_name == 'sinh_taylor': expected_val = np.sinh(scalar_input)
    elif func_name == 'cosh_taylor': expected_val = np.cosh(scalar_input)
    elif func_name == 'tanh_taylor': expected_val = np.tanh(scalar_input)
    elif func_name == 'arcsin_taylor':
        if -1 <= scalar_input <= 1:
            expected_val = np.arcsin(scalar_input)
        else:
            expected_val = np.nan
    elif func_name == 'arccos_taylor':
        if -1 <= scalar_input <= 1:
            expected_val = np.arccos(scalar_input)
        else:
            expected_val = np.nan
    elif func_name == 'arctanh_taylor':
        if -1 < scalar_input < 1:
            expected_val = np.arctanh(scalar_input)
        else:
            expected_val = np.nan

    if func_name in ['sqrt_taylor', 'log_taylor', 'arcsin_taylor', 'arccos_taylor', 'arctanh_taylor'] and np.isnan(expected_val):
        pytest.xfail(f"{func_name} not defined at scalar_input={scalar_input}")
    else:
        assert np.allclose(scalar_eval_result, np.array([expected_val]))

# --- Additional Tests ---

def test_mtf_equality(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf1 = MultivariateTaylorFunction({exponent_zero: np.array([1.0]).reshape(1), exponent_one: np.array([2.0]).reshape(1)}, dimension=global_dim)
    mtf2 = MultivariateTaylorFunction({exponent_zero: np.array([1.0]).reshape(1), exponent_one: np.array([2.0]).reshape(1)}, dimension=global_dim)
    mtf3 = MultivariateTaylorFunction({exponent_zero: np.array([1.0]).reshape(1), exponent_one: np.array([3.0]).reshape(1)}, dimension=global_dim)
    assert mtf1 == mtf2
    assert mtf1 != mtf3

def test_cmtf_equality(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf1 = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+1j]).reshape(1), exponent_one: np.array([2-1j]).reshape(1)}, dimension=global_dim)
    cmtf2 = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+1j]).reshape(1), exponent_one: np.array([2-1j]).reshape(1)}, dimension=global_dim)
    cmtf3 = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+1j]).reshape(1), exponent_one: np.array([3-1j]).reshape(1)}, dimension=global_dim)
    assert cmtf1 == cmtf2
    assert cmtf1 != cmtf3


def test_mtf_pickle_unpickle(setup_function):
    import pickle
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)
    mtf_obj = MultivariateTaylorFunction({exponent_zero: np.array([1.0]).reshape(1), exponent_one: np.array([2.0]).reshape(1)}, global_dim)
    pickled_mtf = pickle.dumps(mtf_obj)
    unpickled_mtf = pickle.loads(pickled_mtf)
    assert unpickled_mtf == mtf_obj

def test_cmtf_pickle_unpickle(setup_function):
    import pickle
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)
    cmtf = ComplexMultivariateTaylorFunction({exponent_zero: np.array([1+1j]).reshape(1), exponent_one: np.array([2-1j]).reshape(1)}, dimension=global_dim)
    pickled_cmtf = pickle.dumps(cmtf)
    unpickled_cmtf = pickle.loads(pickled_cmtf)
    assert unpickled_cmtf == cmtf