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
    yield
    # global _INITIALIZED
    # from MTFLibrary.taylor_function import _INITIALIZED
    # _INITIALIZED = False
    MTFLibrary.taylor_function._INITIALIZED = False


# --- Global Settings Tests ---

# --- Tests for Global Settings ---
def test_global_initialization(setup_function):
    print("test_global_initialization STARTS") # Debug 4: Test function start
    print(f"test_global_initialization: Initial value of _INITIALIZED: {get_mtf_initialized_status()}") # Debug 5: Check _INITIALIZED before re-init
    assert get_global_max_order() == MAX_ORDER # Use getter function
    assert get_global_max_dimension() == MAX_DIMENSION # Use getter function
    print("test_global_initialization: First assertions PASSED") # Debug 6: After first assertions
    try:
        print("test_global_initialization: Attempting re-initialization...") # Debug 7: Before pytest.raises
        with pytest.raises(RuntimeError):
            print("test_global_initialization: Inside pytest.raises block") # Debug 8: Inside pytest.raises block
            print(f"test_global_initialization (in pytest.raises): Value of _INITIALIZED before re-init call: {get_mtf_initialized_status()}") # Debug 9: _INITIALIZED before re-init call
            initialize_mtf_globals(max_order=6, max_dimension=3) # Re-initialization attempt
        print("test_global_initialization: RuntimeError EXPECTEDLY raised and caught") # Debug 10: After pytest.raises
    except Exception as e:
        print(f"test_global_initialization: Error during pytest.raises check: {e}")
        raise
    print("test_global_initialization ENDS: Test PASSED completely.") # Debug 11: Test function end

def test_global_initialization(setup_function):
    assert get_global_max_order() == MAX_ORDER
    assert get_global_max_dimension() == MAX_DIMENSION
    with pytest.raises(RuntimeError):
        initialize_mtf_globals(max_order=MAX_ORDER, max_dimension=MAX_DIMENSION) # Re-initialization should fail

def test_global_max_order_setting(setup_function):
    set_global_max_order(MAX_ORDER)
    assert get_global_max_order() == MAX_ORDER
    with pytest.raises(ValueError):
        set_global_max_order(-1) # Invalid order

def test_global_etol_setting(setup_function):
    set_global_etol(1e-6)
    assert get_global_etol() == 1e-6
    with pytest.raises(ValueError):
        set_global_etol(-1e-6) # Invalid etol
    with pytest.raises(ValueError):
        set_global_etol(0.0) # Invalid etol


# --- Var Function Tests ---
def test_var_creation(setup_function):
    x_var = Var(1)
    assert isinstance(x_var, MultivariateTaylorFunction) # This should still be MultivariateTaylorFunction
    assert x_var.dimension == 6
    coeff_x1 = x_var.extract_coefficient((1,0,0,0,0,0))
    assert np.allclose(coeff_x1, np.array([1.0]).reshape(1))
    coeff_constant = x_var.extract_coefficient((0,0,0,0,0,0))
    assert np.allclose(coeff_constant, np.array([0.0]).reshape(1))
    with pytest.raises(ValueError):
        Var(0)
    with pytest.raises(ValueError):
        Var(7)
    MTFLibrary.taylor_function._INITIALIZED = False
    with pytest.raises(RuntimeError):
        Var(1) # Before initialization


# --- MultivariateTaylorFunction (Real MTF) Tests ---
def test_mtf_constant_creation(setup_function):
    const_mtf = MultivariateTaylorFunction.from_constant(5.0, dimension=2)
    assert np.allclose(const_mtf.eval([0, 0]), np.array([5.0]).reshape(1))
    assert np.allclose(const_mtf.extract_coefficient((0, 0)), np.array([5.0]).reshape(1))
    assert np.allclose(const_mtf.extract_coefficient((1, 0)), np.array([0.0]).reshape(1))

def test_mtf_variable_evaluation(setup_function):
    x1_var = Var(1)
    x2_var = Var(2)
    assert np.allclose(x1_var.eval([2.0, 0.0, 0, 0, 0, 0]), np.array([2.0]).reshape(1))
    assert np.allclose(x2_var.eval([0.0, 3.0, 0, 0, 0, 0]), np.array([3.0]).reshape(1))

def test_mtf_truncate(setup_function):
    coeffs = {(0,): np.array([1.0]).reshape(1), (1,): np.array([2.0]).reshape(1), (2,): np.array([3.0]).reshape(1), (3,): np.array([4.0]).reshape(1)}
    mtf = MultivariateTaylorFunction(coefficients=coeffs, dimension=1)
    truncated_mtf = mtf.truncate(2)
    assert truncated_mtf.extract_coefficient((3,)) == pytest.approx(np.array([0.0]).reshape(1))
    assert truncated_mtf.extract_coefficient((2,)) == pytest.approx(np.array([3.0]).reshape(1))
    assert truncated_mtf.extract_coefficient((1,)) == pytest.approx(np.array([2.0]).reshape(1))
    assert truncated_mtf.extract_coefficient((0,)) == pytest.approx(np.array([1.0]).reshape(1))

def test_mtf_extract_coefficient(setup_function):
    mtf = MultivariateTaylorFunction(coefficients={(1, 1): np.array([2.5]).reshape(1), (0, 0): np.array([1.0]).reshape(1)}, dimension=2) #Dimension is set here
    assert np.allclose(mtf.extract_coefficient((1, 1)), np.array([2.5]).reshape(1))
    assert np.allclose(mtf.extract_coefficient((0, 0)), np.array([1.0]).reshape(1))
    assert np.allclose(mtf.extract_coefficient((2, 0)), np.array([0.0]).reshape(1)) # Non-existent coefficient should be zero

def test_mtf_set_coefficient(setup_function):
    mtf = MultivariateTaylorFunction.from_constant(0.0, dimension=2)
    mtf.set_coefficient((1, 0), 3.0)
    assert np.allclose(mtf.extract_coefficient((1, 0)), np.array([3.0]).reshape(1))
    mtf.set_coefficient((1, 0), 0.0) # Setting to zero
    assert np.allclose(mtf.extract_coefficient((1, 0)), np.array([0.0]).reshape(1))
    with pytest.raises(TypeError):
        mtf.set_coefficient([1, 0], 2.0) # Exponents must be tuple
    with pytest.raises(ValueError):
        mtf.set_coefficient((1, 0, 0), 2.0) # Exponent dimension mismatch
    with pytest.raises(TypeError):
        mtf.set_coefficient((1, 0), "invalid") # Value must be numeric

def test_mtf_get_max_coefficient(setup_function):
    coeffs = {(0, 0): np.array([1.0]).reshape(1), (1, 0): np.array([-2.0]).reshape(1), (0, 1): np.array([3.0]).reshape(1)}
    mtf = MultivariateTaylorFunction(coefficients=coeffs, dimension=2)
    assert mtf.get_max_coefficient() == pytest.approx(3.0)


def test_mtf_get_min_coefficient(setup_function):
    etol = get_global_etol()
    print(f"Value of etol: {etol}") # Debug print: Check the value of etol
    coeffs = {(0, 0): np.array([0.1]).reshape(1), (1, 0): np.array([2.0]).reshape(1), (0, 1): np.array([3.0]).reshape(1)}
    mtf = MultivariateTaylorFunction(coefficients=coeffs, dimension=2)
    assert mtf.get_min_coefficient(tolerance=0.5) == pytest.approx(2.0) # Min non-negligible coefficient (above 0.5)
    coeffs_negligible = {(0, 0): np.array([etol]).reshape(1)}
    mtf_negligible = MultivariateTaylorFunction(coefficients=coeffs_negligible, dimension=2)
    assert mtf_negligible.get_min_coefficient() == pytest.approx(0.0) # All negligible, returns 0.0

# def test_mtf_get_min_coefficient(setup_function):
#     etol = get_global_etol()
#     coeffs = {(0, 0): np.array([0.1]).reshape(1), (1, 0): np.array([2.0]).reshape(1), (0, 1): np.array([3.0]).reshape(1)}
#     mtf = MultivariateTaylorFunction(coefficients=coeffs, dimension=2)
#     assert mtf.get_min_coefficient(tolerance=0.5) == pytest.approx(2.0) # Min non-negligible coefficient (above 0.5)
#     coeffs_negligible = {(0, 0): np.array([etol]).reshape(1)}
#     mtf_negligible = MultivariateTaylorFunction(coefficients=coeffs_negligible, dimension=2)
#     assert mtf_negligible.get_min_coefficient() == pytest.approx(0.0) # All negligible, returns 0.0

# def test_mtf_to_complex_mtf(setup_function):
#     mtf_real = MultivariateTaylorFunction.from_constant(2.5, dimension=1)
#     cmtf = mtf_real.to_complex_mtf()
#     assert isinstance(cmtf, ComplexMultivariateTaylorFunction)
#     assert np.allclose(cmtf.extract_coefficient((0,)).real, np.array([2.5]).reshape(1))
#     assert np.allclose(cmtf.extract_coefficient((0,)).imag, np.array([0.0]).reshape(1))

# --- MTF Arithmetic Operations Tests ---
def test_mtf_addition(setup_function):
    mtf1 = MultivariateTaylorFunction({(0,): np.array([1.0]).reshape(1), (1,): np.array([2.0]).reshape(1)}, dimension=1) # 1 + 2x
    mtf2 = MultivariateTaylorFunction({(0,): np.array([3.0]).reshape(1), (1,): np.array([-1.0]).reshape(1)}, dimension=1) # 3 - x
    mtf_sum = mtf1 + mtf2 # (1+2x) + (3-x) = 4 + x
    assert np.allclose(mtf_sum.extract_coefficient((0,)), np.array([4.0]).reshape(1))
    assert np.allclose(mtf_sum.extract_coefficient((1,)), np.array([1.0]).reshape(1))
    mtf_sum_const = mtf1 + 2.0 # (1+2x) + 2 = 3 + 2x
    assert np.allclose(mtf_sum_const.extract_coefficient((0,)), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_sum_const.extract_coefficient((1,)), np.array([2.0]).reshape(1))
    mtf_sum_rconst = 2.0 + mtf1 # 2 + (1+2x) = 3 + 2x (commutativity)
    assert np.allclose(mtf_sum_rconst.extract_coefficient((0,)), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_sum_rconst.extract_coefficient((1,)), np.array([2.0]).reshape(1))

def test_mtf_subtraction(setup_function):
    mtf1 = MultivariateTaylorFunction({(0,): np.array([5.0]).reshape(1), (1,): np.array([3.0]).reshape(1)}, dimension=1) # 5 + 3x
    mtf2 = MultivariateTaylorFunction({(0,): np.array([2.0]).reshape(1), (1,): np.array([1.0]).reshape(1)}, dimension=1) # 2 + x
    mtf_diff = mtf1 - mtf2 # (5+3x) - (2+x) = 3 + 2x
    assert np.allclose(mtf_diff.extract_coefficient((0,)), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_diff.extract_coefficient((1,)), np.array([2.0]).reshape(1))
    mtf_diff_const = mtf1 - 2.0 # (5+3x) - 2 = 3 + 3x
    assert np.allclose(mtf_diff_const.extract_coefficient((0,)), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_diff_const.extract_coefficient((1,)), np.array([3.0]).reshape(1))
    mtf_diff_rconst = 4.0 - mtf1 # 4 - (5+3x) = -1 - 3x
    assert np.allclose(mtf_diff_rconst.extract_coefficient((0,)), np.array([-1.0]).reshape(1))
    assert np.allclose(mtf_diff_rconst.extract_coefficient((1,)), np.array([-3.0]).reshape(1))

def test_mtf_multiplication(setup_function):
    mtf1 = MultivariateTaylorFunction({(0,): np.array([2.0]).reshape(1), (1,): np.array([1.0]).reshape(1)}, dimension=1) # 2 + x
    mtf2 = MultivariateTaylorFunction({(0,): np.array([3.0]).reshape(1), (1,): np.array([-2.0]).reshape(1)}, dimension=1) # 3 - 2x
    mtf_prod = mtf1 * mtf2 # (2+x) * (3-2x) = 6 - 4x + 3x - 2x^2 = 6 - x - 2x^2
    assert np.allclose(mtf_prod.extract_coefficient((0,)), np.array([6.0]).reshape(1))
    assert np.allclose(mtf_prod.extract_coefficient((1,)), np.array([-1.0]).reshape(1))
    assert np.allclose(mtf_prod.extract_coefficient((2,)), np.array([-2.0]).reshape(1))
    mtf_prod_const = mtf1 * 3.0 # (2+x) * 3 = 6 + 3x
    assert np.allclose(mtf_prod_const.extract_coefficient((0,)), np.array([6.0]).reshape(1))
    assert np.allclose(mtf_prod_const.extract_coefficient((1,)), np.array([3.0]).reshape(1))
    mtf_prod_rconst = 3.0 * mtf1 # 3 * (2+x) = 6 + 3x (commutativity)
    assert np.allclose(mtf_prod_rconst.extract_coefficient((0,)), np.array([6.0]).reshape(1))
    assert np.allclose(mtf_prod_rconst.extract_coefficient((1,)), np.array([3.0]).reshape(1))

def test_mtf_power(setup_function):
    mtf = MultivariateTaylorFunction({(0,): np.array([1.0]).reshape(1), (1,): np.array([1.0]).reshape(1)}, dimension=1) # 1 + x
    mtf_sq = mtf ** 2 # (1+x)^2 = 1 + 2x + x^2
    assert np.allclose(mtf_sq.extract_coefficient((0,)), np.array([1.0]).reshape(1))
    assert np.allclose(mtf_sq.extract_coefficient((1,)), np.array([2.0]).reshape(1))
    assert np.allclose(mtf_sq.extract_coefficient((2,)), np.array([1.0]).reshape(1))
    mtf_cube = mtf ** 3 # (1+x)^3 = 1 + 3x + 3x^2 + x^3, but truncated to order 2
    assert np.allclose(mtf_cube.extract_coefficient((0,)), np.array([1.0]).reshape(1))
    assert np.allclose(mtf_cube.extract_coefficient((1,)), np.array([3.0]).reshape(1))
    assert np.allclose(mtf_cube.extract_coefficient((2,)), np.array([3.0]).reshape(1))
    assert mtf_cube.extract_coefficient((3,)) == pytest.approx(np.array([1.0]).reshape(1))
    with pytest.raises(ValueError):
        mtf ** (-2) # Negative power not allowed
    with pytest.raises(ValueError):
        mtf ** 2.5 # Non-integer power not allowed

def test_mtf_negation(setup_function):
    mtf = MultivariateTaylorFunction({(0,): np.array([2.0]).reshape(1), (1,): np.array([-1.0]).reshape(1)}, dimension=1) # 2 - x
    neg_mtf = -mtf # -(2 - x) = -2 + x
    assert np.allclose(neg_mtf.extract_coefficient((0,)), np.array([-2.0]).reshape(1))
    assert np.allclose(neg_mtf.extract_coefficient((1,)), np.array([1.0]).reshape(1))

def test_mtf_eval_shape_consistency(setup_function):
    mtf = MultivariateTaylorFunction({(0,): np.array([2.0]).reshape(1), (1,): np.array([-1.0]).reshape(1)}, dimension=1)
    eval_result = mtf.eval([0.5])
    assert eval_result.shape == (1,) # Evaluation result should be shape (1,)

    mtf_sum_result = (mtf + mtf).eval([0.5])
    assert mtf_sum_result.shape == (1,)

    mtf_mul_result = (mtf * 2.0).eval([0.5])
    assert mtf_mul_result.shape == (1,)


# --- ComplexMultivariateTaylorFunction (Complex CMTF) Tests ---
def test_cmtf_creation(setup_function):
    coeffs = {(0,): np.array([1+1j]).reshape(1), (1,): np.array([2-1j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=1)
    assert np.allclose(cmtf.extract_coefficient((0,)), np.array([1+1j]).reshape(1))
    assert np.allclose(cmtf.extract_coefficient((1,)), np.array([2-1j]).reshape(1))

def test_cmtf_variable_evaluation(setup_function):
    x1_var_c = ComplexMultivariateTaylorFunction.from_variable(1, dimension=2)
    eval_point = [0.5, 0.0]
    assert np.allclose(x1_var_c.eval(eval_point), np.array([0.5+0.0j]).reshape(1))

def test_cmtf_truncate(setup_function):
    coeffs = {(0,): np.array([1+0j]).reshape(1), (1,): np.array([2j]).reshape(1), (2,): np.array([3-3j]).reshape(1), (3,): np.array([4+4j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=1)
    truncated_cmtf = cmtf.truncate(2)
    assert truncated_cmtf.extract_coefficient((3,)) == pytest.approx(np.array([0.0j]).reshape(1))
    assert truncated_cmtf.extract_coefficient((2,)), np.array([3-3j]).reshape(1)

def test_cmtf_extract_coefficient(setup_function):
    coeffs = {(1, 0): np.array([1+1j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=2)
    assert np.allclose(cmtf.extract_coefficient((1, 0)), np.array([1+1j]).reshape(1))
    assert np.allclose(cmtf.extract_coefficient((0, 0)), np.array([0.0j]).reshape(1)) # Default complex zero

def test_cmtf_set_coefficient(setup_function):
    cmtf = ComplexMultivariateTaylorFunction.from_constant(0.0j, dimension=1)
    cmtf.set_coefficient((1,), 2+2j)
    assert np.allclose(cmtf.extract_coefficient((1,)), np.array([2+2j]).reshape(1))
    cmtf.set_coefficient((1,), 0.0j) # Setting to zero
    assert np.allclose(cmtf.extract_coefficient((1,)), np.array([0.0j]).reshape(1))
    with pytest.raises(TypeError):
        cmtf.set_coefficient((1,), "invalid") # Value must be numeric

def test_cmtf_get_max_coefficient(setup_function):
    coeffs = {(0, 0): np.array([1+0j]).reshape(1), (1, 0): np.array([1j]).reshape(1), (0, 1): np.array([-2+0j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=2)
    assert cmtf.get_max_coefficient() == pytest.approx(2.0) # Max magnitude is 2.0

def test_cmtf_get_min_coefficient(setup_function):
    coeffs = {(0, 0): np.array([0.1j]).reshape(1), (1, 0): np.array([2+0j]).reshape(1), (0, 1): np.array([3j]).reshape(1)}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs)
    assert cmtf.get_min_coefficient(tolerance=0.5) == pytest.approx(2.0) # Min non-negligible magnitude (above 0.5)
    coeffs_negligible = {(0, 0): np.array([1e-10j]).reshape(1)}
    cmtf_negligible = ComplexMultivariateTaylorFunction(coefficients=coeffs_negligible, dimension=2)
    assert cmtf_negligible.get_min_coefficient() == pytest.approx(0.0) # All negligible, returns 0.0

def test_cmtf_conjugate(setup_function):
    cmtf = ComplexMultivariateTaylorFunction({(0,): np.array([1+1j]).reshape(1), (1,): np.array([2-1j]).reshape(1)}, dimension=1)
    conj_cmtf = cmtf.conjugate()
    assert np.allclose(conj_cmtf.extract_coefficient((0,)), np.array([1-1j]).reshape(1))
    assert np.allclose(conj_cmtf.extract_coefficient((1,)), np.array([2+1j]).reshape(1))

def test_cmtf_real_part(setup_function):
    cmtf = ComplexMultivariateTaylorFunction({(0,): np.array([1+1j]).reshape(1), (1,): np.array([2-1j]).reshape(1)}, dimension=1)
    real_mtf = cmtf.real_part()
    assert isinstance(real_mtf, MultivariateTaylorFunctionBase)
    assert np.allclose(real_mtf.extract_coefficient((0,)), np.array([1.0]).reshape(1))
    assert np.allclose(real_mtf.extract_coefficient((1,)), np.array([2.0]).reshape(1))

def test_cmtf_imag_part(setup_function):
    cmtf = ComplexMultivariateTaylorFunction({(0,): np.array([1+1j]).reshape(1), (1,): np.array([2-1j]).reshape(1)}, dimension=1)
    imag_mtf = cmtf.imag_part()
    assert isinstance(imag_mtf, MultivariateTaylorFunctionBase)
    assert np.allclose(imag_mtf.extract_coefficient((0,)), np.array([1.0]).reshape(1))
    assert np.allclose(imag_mtf.extract_coefficient((1,)), np.array([-1.0]).reshape(1))

def test_cmtf_magnitude_phase_not_implemented(setup_function):
    cmtf = ComplexMultivariateTaylorFunction.from_constant(1+1j, dimension=1)
    with pytest.raises(NotImplementedError):
        cmtf.magnitude()
    with pytest.raises(NotImplementedError):
        cmtf.phase()

# --- CMTF Arithmetic Operations Tests ---
def test_cmtf_addition(setup_function):
    cmtf1 = ComplexMultivariateTaylorFunction({(0,): np.array([1+1j]).reshape(1), (1,): np.array([2-1j]).reshape(1)}, dimension=1) # (1+j) + (2-j)x
    cmtf2 = ComplexMultivariateTaylorFunction({(0,): np.array([3-2j]).reshape(1), (1,): np.array([-1+0.5j]).reshape(1)}, dimension=1) # (3-2j) + (-1+0.5j)x
    cmtf_sum = cmtf1 + cmtf2 # (4-j) + (1-0.5j)x
    assert np.allclose(cmtf_sum.extract_coefficient((0,)), np.array([4-1j]).reshape(1))
    assert np.allclose(cmtf_sum.extract_coefficient((1,)), np.array([1-0.5j]).reshape(1))
    cmtf_sum_const = cmtf1 + (2+0j) # (3+j) + (2-j)x
    assert np.allclose(cmtf_sum_const.extract_coefficient((0,)), np.array([3+1j]).reshape(1))
    assert np.allclose(cmtf_sum_const.extract_coefficient((1,)), np.array([2-1j]).reshape(1))
    cmtf_sum_rconst = (2+0j) + cmtf1 # commutativity
    assert np.allclose(cmtf_sum_rconst.extract_coefficient((0,)), np.array([3+1j]).reshape(1))
    assert np.allclose(cmtf_sum_rconst.extract_coefficient((1,)), np.array([2-1j]).reshape(1))

def test_cmtf_subtraction(setup_function):
    cmtf1 = ComplexMultivariateTaylorFunction({(0,): np.array([5+0j]).reshape(1), (1,): np.array([3+1j]).reshape(1)}, dimension=1) # 5 + (3+j)x
    cmtf2 = ComplexMultivariateTaylorFunction({(0,): np.array([2+1j]).reshape(1), (1,): np.array([1-1j]).reshape(1)}, dimension=1) # (2+j) + (1-j)x
    cmtf_diff = cmtf1 - cmtf2 # (3-j) + (2+2j)x
    assert np.allclose(cmtf_diff.extract_coefficient((0,)), np.array([3-1j]).reshape(1))
    assert np.allclose(cmtf_diff.extract_coefficient((1,)), np.array([2+2j]).reshape(1))
    cmtf_diff_const = cmtf1 - (2+0j) # (3+0j) + (3+1j)x
    assert np.allclose(cmtf_diff_const.extract_coefficient((0,)), np.array([3+0j]).reshape(1))
    assert np.allclose(cmtf_diff_const.extract_coefficient((1,)), np.array([3+1j]).reshape(1))
    cmtf_diff_rconst = (4+0j) - cmtf1 # (-1+0j) + (-3-1j)x
    assert np.allclose(cmtf_diff_rconst.extract_coefficient((0,)), np.array([-1+0j]).reshape(1))
    assert np.allclose(cmtf_diff_rconst.extract_coefficient((1,)), np.array([-3-1j]).reshape(1))

def test_cmtf_multiplication(setup_function):
    cmtf1 = ComplexMultivariateTaylorFunction({(0,): np.array([1+0j]).reshape(1), (1,): np.array([1j]).reshape(1)}, dimension=1) # 1 + jx
    cmtf2 = ComplexMultivariateTaylorFunction({(0,): np.array([1j]).reshape(1), (1,): np.array([-1+0j]).reshape(1)}, dimension=1) # j - x
    cmtf_prod = cmtf1 * cmtf2 # j - x -x + (-j)x^2 = j - 2x -jx^2
    assert np.allclose(cmtf_prod.extract_coefficient((0,)), np.array([1j]).reshape(1))
    assert np.allclose(cmtf_prod.extract_coefficient((1,)), np.array([-2+0j]).reshape(1))
    assert np.allclose(cmtf_prod.extract_coefficient((2,)), np.array([-1j]).reshape(1))
    cmtf_prod_const = cmtf1 * (2+0j) # (2+0j) + (2j)x
    assert np.allclose(cmtf_prod_const.extract_coefficient((0,)), np.array([2+0j]).reshape(1))
    assert np.allclose(cmtf_prod_const.extract_coefficient((1,)), np.array([2j]).reshape(1))
    cmtf_prod_rconst = (2+0j) * cmtf1 # commutativity
    assert np.allclose(cmtf_prod_rconst.extract_coefficient((0,)), np.array([2+0j]).reshape(1))
    assert np.allclose(cmtf_prod_rconst.extract_coefficient((1,)), np.array([2j]).reshape(1))

def test_cmtf_power(setup_function):
    cmtf = ComplexMultivariateTaylorFunction({(0,): np.array([1+0j]).reshape(1), (1,): np.array([1j]).reshape(1)}, dimension=1) # 1 + jx
    cmtf_sq = cmtf ** 2 # (1+jx)^2 = 1 + 2jx - x^2
    assert np.allclose(cmtf_sq.extract_coefficient((0,)), np.array([1+0j]).reshape(1))
    assert np.allclose(cmtf_sq.extract_coefficient((1,)), np.array([2j]).reshape(1))
    assert np.allclose(cmtf_sq.extract_coefficient((2,)), np.array([-1+0j]).reshape(1))
    cmtf_cube = cmtf ** 3 # (1+jx)^3 = 1 + 3jx -3x^2 -jx^3, truncated to order 2
    assert np.allclose(cmtf_cube.extract_coefficient((0,)), np.array([1+0j]).reshape(1))
    assert np.allclose(cmtf_cube.extract_coefficient((1,)), np.array([3j]).reshape(1))
    assert np.allclose(cmtf_cube.extract_coefficient((2,)), np.array([-3+0j]).reshape(1))
    assert cmtf_cube.extract_coefficient((MAX_ORDER+1,)) == pytest.approx(np.array([0.0j]).reshape(1)) # Truncated

def test_cmtf_negation(setup_function):
    cmtf = ComplexMultivariateTaylorFunction({(0,): np.array([2+0j]).reshape(1), (1,): np.array([-1+1j]).reshape(1)}, dimension=1) # 2 + (-1+j)x
    neg_cmtf = -cmtf # -2 + (1-j)x
    assert np.allclose(neg_cmtf.extract_coefficient((0,)), np.array([-2+0j]).reshape(1))
    assert np.allclose(neg_cmtf.extract_coefficient((1,)), np.array([1-1j]).reshape(1))

def test_cmtf_eval_shape_consistency(setup_function):
    cmtf = ComplexMultivariateTaylorFunction({(0,): np.array([2+0j]).reshape(1), (1,): np.array([-1+1j]).reshape(1)}, dimension=1)
    eval_result = cmtf.eval([0.5])
    assert eval_result.shape == (1,)

    cmtf_sum_result = (cmtf + cmtf).eval([0.5])
    assert cmtf_sum_result.shape == (1,)

    cmtf_mul_result = (cmtf * (2+0j)).eval([0.5])
    assert cmtf_mul_result.shape == (1,)


# --- convert_to_mtf Tests ---
# def test_convert_to_mtf(setup_function):
#     mtf = convert_to_mtf(5.0, dimension=1)
#     assert isinstance(mtf, MultivariateTaylorFunction)
#     assert np.allclose(mtf.eval([0]), np.array([5.0]).reshape(1))

#     cmtf = convert_to_mtf(2+1j, dimension=1)
#     assert isinstance(cmtf, ComplexMultivariateTaylorFunction)
#     assert np.allclose(cmtf.eval([0]), np.array([2+1j]).reshape(1))

#     x_var_mtf = convert_to_mtf(Var(1))
#     assert isinstance(x_var_mtf, MultivariateTaylorFunction)
#     assert x_var_mtf.dimension == 6 # Dimension from global setting
#     assert np.allclose(x_var_mtf.eval([1.0, 0, 0, 0, 0, 0]), np.array([1.0]).reshape(1))

#     existing_mtf = MultivariateTaylorFunction.from_constant(3.0, dimension=1)
#     converted_mtf = convert_to_mtf(existing_mtf)
#     assert converted_mtf is existing_mtf # Should return same object if already MTF

#     with pytest.raises(TypeError):
#         convert_to_mtf("invalid") # Invalid type

#     numpy_scalar = np.float64(7.0)
#     mtf_np_scalar = convert_to_mtf(numpy_scalar, dimension=1)
#     assert np.allclose(mtf_np_scalar.eval([0]), np.array([7.0]).reshape(1))

#     numpy_0d_array = np.array(9.0)
#     mtf_np_0d = convert_to_mtf(numpy_0d_array, dimension=1)
#     assert np.allclose(mtf_np_0d.eval([0]), np.array([9.0]).reshape(1))


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
    scalar_input = 0.5;
    mtf_result = func(scalar_input)
    assert isinstance(mtf_result, MultivariateTaylorFunctionBase) # Changed assertion here
    zero_point = tuple([0 for _ in range(get_global_max_dimension())])
    scalar_eval_result = func(scalar_input).eval(zero_point) # Evaluate at 0 as Taylor series is around 0

    if func_name == 'cos_taylor': expected_val = np.cos(scalar_input)
    elif func_name == 'sin_taylor': expected_val = np.sin(scalar_input)
    elif func_name == 'exp_taylor': expected_val = np.exp(scalar_input)
    elif func_name == 'gaussian_taylor': expected_val = np.exp(-(scalar_input**2))
    elif func_name == 'sqrt_taylor': expected_val = np.sqrt(scalar_input)
    elif func_name == 'log_taylor': expected_val = np.log(scalar_input)
    elif func_name == 'arctan_taylor': expected_val = np.arctan(scalar_input)
    elif func_name == 'sinh_taylor': expected_val = np.sinh(scalar_input)
    elif func_name == 'cosh_taylor': expected_val = np.cosh(scalar_input)
    elif func_name == 'tanh_taylor': expected_val = np.tanh(scalar_input)
    elif func_name == 'arcsin_taylor': expected_val = np.arcsin(scalar_input)
    elif func_name == 'arccos_taylor': expected_val = np.arccos(scalar_input)
    elif func_name == 'arctanh_taylor': expected_val = np.arctanh(scalar_input)
    else: expected_val = np.nan # Should not reach here

    assert np.allclose(scalar_eval_result, np.array([expected_val]).reshape(1), rtol=1e-10)


@pytest.mark.parametrize("func, func_name", elementary_functions_list)
def test_elementary_functions_mtf(setup_function, func, func_name):
    x_var = Var(1)
    if func_name == 'sqrt_taylor' or func_name == 'log_taylor':
        mtf_input = 1 + x_var
    else:
        mtf_input = x_var # Input MTF is just x
    mtf_result = func(mtf_input)
    assert isinstance(mtf_result, MultivariateTaylorFunctionBase) # Changed assertion here
    # Basic check: non-zero constant term for cosh, cos, exp, sqrt, gaussian, arccos, else zero/small
    constant_term = mtf_result.extract_coefficient((0,0,0,0,0,0))
    if func_name in ['cosh_taylor', 'cos_taylor', 'exp_taylor', 'sqrt_taylor', 'gaussian_taylor', 'arccos_taylor']:
    # if func_name in ['cos_taylor']:
        assert abs(constant_term[0]) > 0.1 # Constant term exists
    else:
        assert abs(constant_term[0]) < 0.1 # Constant term close to zero

# # No CMTF elementary function tests yet as all are real-valued in elementary_functions.py


# --- Error Handling Tests ---
def test_error_handling_global_init(setup_function):
    MTFLibrary.taylor_function._INITIALIZED = False
    with pytest.raises(ValueError):
        initialize_mtf_globals(max_order=-1, max_dimension=2) # Invalid max_order

    MTFLibrary.taylor_function._INITIALIZED = False
    with pytest.raises(ValueError):
        initialize_mtf_globals(max_order=2, max_dimension=0) # Invalid max_dimension

    MTFLibrary.taylor_function._INITIALIZED = True
    with pytest.raises(RuntimeError):
        initialize_mtf_globals(max_order=3, max_dimension=3) # Re-initialization error

def test_error_handling_var(setup_function):
    with pytest.raises(ValueError):
        Var(-1) # Invalid var_index
    with pytest.raises(ValueError):
        Var(7)     # var_index exceeds dimension

def test_error_handling_mtf_dimension(setup_function):
    with pytest.raises(ValueError):
        MultivariateTaylorFunction({}, dimension=0) # Invalid dimension
    mtf = MultivariateTaylorFunction.from_constant(1.0, dimension=2)
    with pytest.raises(ValueError):
        mtf.eval([1.0, 1.0, 1.0, 0, 0, 0]) # Eval dimension mismatch

def test_error_handling_mtf_power(setup_function):
    mtf = MultivariateTaylorFunction.from_constant(2.0, dimension=1)
    with pytest.raises(ValueError):
        mtf**(-2) # Negative power

def test_error_handling_convert_to_mtf(setup_function):
    with pytest.raises(TypeError):
        convert_to_mtf("invalid_input") # Invalid input type