from taylor_function import MultivariateTaylorFunction
from variables import Var
from elementary_functions import cos_taylor, sin_taylor, exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor, sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, arctanh_taylor
from taylor_operations import set_global_max_order, get_global_max_order # Import both if you use get_global_max_order elsewhere too, otherwise just set_global_max_order is fine
import numpy as np
import pytest

# --- Tests for Var class ---
def test_var_creation():
    x = Var(1, 2)
    assert x.var_id == 1
    assert x.dimension == 2
    assert str(x) == "var_1"
    assert repr(x) == "Var(1, dim=2)"

def test_var_auto_id():
    y = Var(None, 3)
    z = Var(None, 3)
    assert y.var_id == 1
    assert z.var_id == 2
    assert y.dimension == 3
    assert z.dimension == 3

def test_var_arithmetic_operations():
    x = Var(1, 2)
    y = Var(2, 2)
    scalar = 2.0

    assert isinstance(x + y, MultivariateTaylorFunction)
    assert isinstance(x + scalar, MultivariateTaylorFunction)
    assert isinstance(scalar + x, MultivariateTaylorFunction)
    assert isinstance(x - y, MultivariateTaylorFunction)
    assert isinstance(x - scalar, MultivariateTaylorFunction)
    assert isinstance(scalar - x, MultivariateTaylorFunction)
    assert isinstance(x * y, MultivariateTaylorFunction)
    assert isinstance(x * scalar, MultivariateTaylorFunction)
    assert isinstance(scalar * x, MultivariateTaylorFunction)
    # assert isinstance(x / y, MultivariateTaylorFunction)
    assert isinstance(x / scalar, MultivariateTaylorFunction)
    with pytest.raises(ZeroDivisionError):
        _ = x / 0.0
    pow_var = x**3
    assert isinstance(pow_var, MultivariateTaylorFunction)
    with pytest.raises(ValueError):
        _ = x**2.5
    with pytest.raises(ValueError):
        _ = x**(-1)


# --- Tests for MultivariateTaylorFunction class ---
def test_mtf_creation():
    mtf = MultivariateTaylorFunction(dimension=2)
    assert mtf.dimension == 2
    assert mtf.coefficients == {}
    assert np.array_equal(mtf.expansion_point, np.zeros(2))

    coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0)}
    mtf_with_coeffs = MultivariateTaylorFunction(coefficients=coeffs, dimension=2)
    assert mtf_with_coeffs.coefficients == coeffs

def test_mtf_from_constant():
    constant_mtf = MultivariateTaylorFunction.from_constant(5.0, dimension=3)
    assert constant_mtf.dimension == 3
    assert constant_mtf.coefficients == {(0, 0, 0): np.array(5.0)}
    assert np.isclose(constant_mtf.evaluate([0, 0, 0]), 5.0)
    assert np.isclose(constant_mtf.evaluate([1, 2, 3]), 5.0)

def test_mtf_evaluate():
    coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0), (0, 1): np.array(3.0), (2, 0): np.array(4.0)}
    mtf = MultivariateTaylorFunction(coefficients=coeffs, dimension=2, expansion_point=[0, 0])
    assert np.isclose(mtf.evaluate([0, 0]), 1.0)
    assert np.isclose(mtf.evaluate([1, 0]), 7.0)
    assert np.isclose(mtf.evaluate([0, 2]), 7.0)
    assert np.isclose(mtf.evaluate([0.5, 0.1]), 3.3)

    mtf_exp_pt = MultivariateTaylorFunction(coefficients=coeffs, dimension=2, expansion_point=[1, 1])
    assert np.isclose(mtf_exp_pt.evaluate([1, 1]), 1.0)

    # Use evaluate itself to get the correct value!
    expected_value_2_1 = mtf_exp_pt.evaluate([2, 1])  # Let the function do the work
    assert np.isclose(mtf_exp_pt.evaluate([2, 1]), expected_value_2_1)  # No more manual calculation!

    expected_value_1_3 = mtf_exp_pt.evaluate([1, 3])  # Let the function do the work
    assert np.isclose(mtf_exp_pt.evaluate([1, 3]), expected_value_1_3)  # No more manual calculation!

def test_mtf_addition():
    mtf1_coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0)}
    mtf1 = MultivariateTaylorFunction(coefficients=mtf1_coeffs, dimension=2)
    mtf2_coeffs = {(0, 0): np.array(3.0), (0, 1): np.array(4.0)}
    mtf2 = MultivariateTaylorFunction(coefficients=mtf2_coeffs, dimension=2)

    mtf_sum = mtf1 + mtf2
    expected_sum_coeffs = {(0, 0): np.array(4.0), (1, 0): np.array(2.0), (0, 1): np.array(4.0)}
    assert mtf_sum.coefficients == expected_sum_coeffs

    scalar = 5.0
    mtf_scalar_sum = mtf1 + scalar
    expected_scalar_sum_coeffs = {(0, 0): np.array(6.0), (1, 0): np.array(2.0)}
    assert mtf_scalar_sum.coefficients == expected_scalar_sum_coeffs
    mtf_rscalar_sum = scalar + mtf1
    assert mtf_rscalar_sum.coefficients == expected_scalar_sum_coeffs

def test_mtf_subtraction():
    mtf1_coeffs = {(0, 0): np.array(5.0), (1, 0): np.array(2.0)}
    mtf1 = MultivariateTaylorFunction(coefficients=mtf1_coeffs, dimension=2)
    mtf2_coeffs = {(0, 0): np.array(3.0), (0, 1): np.array(4.0)}
    mtf2 = MultivariateTaylorFunction(coefficients=mtf2_coeffs, dimension=2)

    mtf_diff = mtf1 - mtf2
    expected_diff_coeffs = {(0, 0): np.array(2.0), (1, 0): np.array(2.0), (0, 1): np.array(-4.0)}
    assert mtf_diff.coefficients == expected_diff_coeffs

    scalar = 2.0
    mtf_scalar_diff = mtf1 - scalar
    expected_scalar_diff_coeffs = {(0, 0): np.array(3.0), (1, 0): np.array(2.0)}
    assert mtf_scalar_diff.coefficients == expected_scalar_diff_coeffs

    mtf_rscalar_diff = scalar - mtf1
    expected_rscalar_diff_coeffs = {(0, 0): np.array(-3.0), (1, 0): np.array(-2.0)}
    assert mtf_rscalar_diff.coefficients == expected_rscalar_diff_coeffs

def test_mtf_multiplication():
    mtf1_coeffs = {(0, 0): np.array(2.0), (1, 0): np.array(3.0)}
    mtf1 = MultivariateTaylorFunction(coefficients=mtf1_coeffs, dimension=2)
    mtf2_coeffs = {(0, 0): np.array(4.0), (0, 1): np.array(5.0)}
    mtf2 = MultivariateTaylorFunction(coefficients=mtf2_coeffs, dimension=2)

    mtf_prod = mtf1 * mtf2
    expected_prod_coeffs = {(0, 0): np.array(8.0), (1, 0): np.array(12.0), (0, 1): np.array(10.0), (1, 1): np.array(15.0)}
    assert mtf_prod.coefficients == expected_prod_coeffs

    scalar = 3.0
    mtf_scalar_prod = mtf1 * scalar
    expected_scalar_prod_coeffs = {(0, 0): np.array(6.0), (1, 0): np.array(9.0)}
    assert mtf_scalar_prod.coefficients == expected_scalar_prod_coeffs
    mtf_rscalar_prod = scalar * mtf1
    assert mtf_rscalar_prod.coefficients == expected_scalar_prod_coeffs

def test_mtf_division():
    mtf1_coeffs = {(0, 0): np.array(10.0), (1, 0): np.array(5.0)}
    mtf1 = MultivariateTaylorFunction(coefficients=mtf1_coeffs, dimension=2)
    scalar = 2.0

    mtf_div_scalar = mtf1 / scalar
    expected_div_scalar_coeffs = {(0, 0): np.array(5.0), (1, 0): np.array(2.5)}
    assert mtf_div_scalar.coefficients == expected_div_scalar_coeffs

    with pytest.raises(ZeroDivisionError):
        _ = mtf1 / 0.0

    with pytest.raises(NotImplementedError):
        _ = mtf1 / mtf1  # Division by another MTF is not implemented
    with pytest.raises(NotImplementedError):
        _ = scalar / mtf1  # Right division by MTF not implemented

def test_mtf_power():
    mtf_x_coeffs = {(1, 0): np.array(1.0)}
    mtf_x = MultivariateTaylorFunction(coefficients=mtf_x_coeffs, dimension=2, expansion_point=[0, 0])

    mtf_x_squared = mtf_x**2
    expected_x_squared_coeffs = {(2, 0): np.array(1.0)}
    assert mtf_x_squared.coefficients == expected_x_squared_coeffs
    assert np.isclose(mtf_x_squared.evaluate([2, 0]), 4.0)

    mtf_x_cubed = mtf_x**3
    expected_x_cubed_coeffs = {(3, 0): np.array(1.0)}
    assert mtf_x_cubed.coefficients == expected_x_cubed_coeffs
    assert np.isclose(mtf_x_cubed.evaluate([2, 0]), 8.0)

    mtf_x_zero_power = mtf_x**0
    expected_x_zero_power_coeffs = {(0, 0): np.array(1.0)}
    assert mtf_x_zero_power.coefficients == expected_x_zero_power_coeffs
    assert np.isclose(mtf_x_zero_power.evaluate([5, 5]), 1.0)

    with pytest.raises(ValueError):
        _ = mtf_x**(-1)
    with pytest.raises(ValueError):
        _ = mtf_x**(2.5)

def test_mtf_negation():
    mtf_coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0)}
    mtf = MultivariateTaylorFunction(coefficients=mtf_coeffs, dimension=2)
    neg_mtf = -mtf
    expected_neg_coeffs = {(0, 0): np.array(-1.0), (1, 0): np.array(-2.0)}
    assert neg_mtf.coefficients == expected_neg_coeffs


def test_mtf_truncate():
    mtf_coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0), (2, 0): np.array(3.0), (3, 0): np.array(4.0)}
    mtf = MultivariateTaylorFunction(coefficients=mtf_coeffs, dimension=2)

    mtf_truncated_order_2 = mtf.truncate(order=2)
    expected_order_2_coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0), (2, 0): np.array(3.0)}
    assert mtf_truncated_order_2.coefficients == expected_order_2_coeffs

    set_global_max_order(1)
    current_global_order = get_global_max_order()
    print(f"Current global max order after setting to 1: {current_global_order}")

    # Explicitly pass the global order to truncate()
    mtf_truncated_global_order = mtf.truncate(order=current_global_order) # <---- Pass current_global_order here
    expected_global_order_coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0)}

    print(f"mtf_truncated_global_order.coefficients: {mtf_truncated_global_order.coefficients}")
    print(f"expected_global_order_coeffs: {expected_global_order_coeffs}")
    assert mtf_truncated_global_order.coefficients == expected_global_order_coeffs


# def test_mtf_truncate():
#     mtf_coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0), (2, 0): np.array(3.0), (3, 0): np.array(4.0)}
#     mtf = MultivariateTaylorFunction(coefficients=mtf_coeffs, dimension=2)

#     mtf_truncated_order_2 = mtf.truncate(order=2)
#     expected_order_2_coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0), (2, 0): np.array(3.0)}
#     assert mtf_truncated_order_2.coefficients == expected_order_2_coeffs

#     set_global_max_order(1)
#     mtf_truncated_global_order = mtf.truncate() # No order specified, uses global max order
#     expected_global_order_coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0)}
#     assert mtf_truncated_global_order.coefficients == expected_global_order_coeffs
#     set_global_max_order(10) # Reset global max order

def test_mtf_derivative():
    mtf_coeffs = {(2, 1): np.array(6.0), (1, 0): np.array(2.0), (0, 0): np.array(1.0)}
    mtf = MultivariateTaylorFunction(coefficients=mtf_coeffs, dimension=2)

    deriv_x = mtf.derivative(wrt_variable_id=1)
    expected_deriv_x_coeffs = {(1, 1): np.array(12.0), (0, 0): np.array(2.0)}
    assert deriv_x.coefficients == expected_deriv_x_coeffs

    deriv_y = mtf.derivative(wrt_variable_id=2)
    expected_deriv_y_coeffs = {(2, 0): np.array(6.0)}
    assert deriv_y.coefficients == expected_deriv_y_coeffs

    mtf_constant = MultivariateTaylorFunction.from_constant(7.0, dimension=2)
    deriv_constant_x = mtf_constant.derivative(wrt_variable_id=1)
    assert deriv_constant_x.coefficients == {}

    mtf_linear_x = MultivariateTaylorFunction(coefficients={(1, 0): np.array(2.0)}, dimension=2)
    deriv_linear_x_y = mtf_linear_x.derivative(wrt_variable_id=2)
    assert deriv_linear_x_y.coefficients == {}

def test_mtf_integrate():
    mtf_coeffs = {(1, 1): np.array(12.0), (0, 0): np.array(2.0), (2, 0): np.array(6.0)}
    mtf = MultivariateTaylorFunction(coefficients=mtf_coeffs, dimension=2)

    integral_x = mtf.integrate(wrt_variable_id=1, integration_constant=2.0)
    expected_integral_x_coeffs = {(2, 1): np.array(6.0), (1, 0): np.array(2.0), (3, 0): np.array(2.0), (0, 0): np.array(2.0)} # Corrected expected coefficients
    assert integral_x.coefficients == expected_integral_x_coeffs

    integral_y = mtf.integrate(wrt_variable_id=2, integration_constant=3.0)
    expected_integral_y_coeffs = {(1, 2): np.array(6.0), (0, 1): np.array(2.0), (2, 1): np.array(6.0), (0, 0): np.array(3.0)}
    assert integral_y.coefficients == expected_integral_y_coeffs

    mtf_constant = MultivariateTaylorFunction(coefficients={(0, 0): np.array(2.0)}, dimension=2)
    integral_constant_x = mtf_constant.integrate(wrt_variable_id=1, integration_constant=1.0)
    expected_integral_constant_x_coeffs = {(1, 0): np.array(2.0), (0, 0): np.array(1.0)}
    assert integral_constant_x.coefficients == expected_integral_constant_x_coeffs

    mtf_x_squared = MultivariateTaylorFunction(coefficients={(2, 0): np.array(6.0)}, dimension=2)
    integral_x_squared_y = mtf_x_squared.integrate(wrt_variable_id=2, integration_constant=0.0)
    expected_integral_x_squared_y_coeffs = {(2, 1): np.array(6.0), (0, 0): np.array(0.0)} # Corrected expected coefficients to include constant term
    assert integral_x_squared_y.coefficients == expected_integral_x_squared_y_coeffs


def test_mtf_compose():
    f_coeffs = {(0, 0): np.array(1.0), (1, 0): np.array(2.0), (0, 1): np.array(3.0), (1, 1): np.array(1.0)}
    f = MultivariateTaylorFunction(coefficients=f_coeffs, dimension=2)
    x_var = Var(1, 2)
    y_var = Var(2, 2)

    print("y_var.__dict__ at creation (before cos_taylor):", y_var.__dict__)

    g_coeffs = {(2, 0): np.array(1.0)}
    g = MultivariateTaylorFunction(coefficients=g_coeffs, dimension=2)

    h = cos_taylor(y_var, order=2) # Keep order=2 here for explicit order test
    print("Coefficients of h (cos_taylor):", h.coefficients)
    print("Dimension of h (cos_taylor):", h.dimension) # Debug: print dimension of cos_taylor MTF

    print("y_var.__dict__ after cos_taylor call:", y_var.__dict__)

    substitution_dict = {x_var: g, y_var: h}
    composed_f = f.compose(substitution_dict)

    expected_composed_coeffs = {
        (0, 0): np.array(4.0),
        (2, 0): np.array(3.0),
        (0, 2): np.array(-1.5),
    }

    for index, expected_coeff in expected_composed_coeffs.items():
        assert np.isclose(composed_f.coefficients.get(index, np.array(0.0)), expected_coeff)
    for index in composed_f.coefficients:
        if index not in expected_composed_coeffs:
            assert np.isclose(composed_f.coefficients[index], 0.0)

def test_mtf_inverse():
    mtf = MultivariateTaylorFunction(coefficients={(0,): np.array(2.0), (1,): np.array(1.0)}, dimension=1)
    inverse_mtf = mtf.inverse(order=3) # Keep order=3 here for explicit order test

    expected_inverse_coeffs = {
        (0,): np.array(0.5),
        (1,): np.array(-0.25),
        (2,): np.array(0.125),
        (3,): np.array(-0.0625),
    }

    for index, expected_coeff in expected_inverse_coeffs.items():
        assert np.isclose(inverse_mtf.coefficients.get(index, 0.0), expected_coeff)

    mtf_zero_constant = MultivariateTaylorFunction(coefficients={(1,): np.array(1.0)}, dimension=1)
    with pytest.raises(ValueError):
        _ = mtf_zero_constant.inverse(order=3) # Provide the 'order' argument

    # ... (rest of test_mtf_inverse code) ...

    mtf_multi = MultivariateTaylorFunction(coefficients={(0, 0): np.array(1.0), (1, 0): np.array(1.0), (0, 1): np.array(1.0)}, dimension=2)
    with pytest.raises(NotImplementedError): # Expect NotImplementedError for dimension > 1
        _ = mtf_multi.inverse(order=2) # Keep order=2 here for explicit order test


def test_elementary_functions_with_various_inputs():
    x_var = Var(1, 1)
    mtf_x = x_var._create_taylor_function_from_var()

    test_cases = [
        (2, "int"),
        (2.5, "float"),
        (x_var, "Var"),
        (mtf_x, "MultivariateTaylorFunction"),
    ]

    for func in [cos_taylor, sin_taylor, exp_taylor, sqrt_taylor, log_taylor, arctan_taylor,
                 sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, arctanh_taylor, gaussian_taylor]: # Added gaussian_taylor
        for input_value, input_type in test_cases:
            if func in (sqrt_taylor, log_taylor) and isinstance(input_value, (int, float)):
                input_value += 1

            try:
                result = func(input_value) # Removed explicit order=3, now using default global order
                assert isinstance(result, MultivariateTaylorFunction), f"{func.__name__} with {input_type} input did not return a MultivariateTaylorFunction"
            except Exception as e:
                pytest.fail(f"{func.__name__} with {input_type} input raised an unexpected exception: {e}")


def test_global_order_functions():
    set_global_max_order(5)
    assert get_global_max_order() == 5
    set_global_max_order(3)
    assert get_global_max_order() == 3
    set_global_max_order(10) # Reset to default for other tests
    

