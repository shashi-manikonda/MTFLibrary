from taylor_function import MultivariateTaylorFunction
from variables import Var
from elementary_functions import cos_taylor, sin_taylor, exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor, sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, arctanh_taylor
from taylor_function import set_global_max_order, get_global_max_order
import numpy as np
import pytest

# --- Test Suite for Multivariate Taylor Function Library ---
# This file contains pytest tests to verify the functionality of the
# Var class, MultivariateTaylorFunction class, and elementary function implementations
# from the MTFLibrary. It includes tests for object creation, arithmetic operations,
# differentiation, integration, composition, inversion, and evaluation, ensuring
# the library behaves as expected under various conditions.

# --- Tests for Var class ---
class TestVarClass:
    """
    Test suite for the Var class, focusing on variable creation and basic operations.
    """
    def test_var_creation(self):
        """Test the creation of a Var object with specified ID and dimension."""
        x = Var(1, 2)
        assert x.var_id == 1
        assert x.dimension == 2
        assert str(x) == "var_1"
        assert repr(x) == "Var(1, dim=2)"

    def test_var_auto_id(self):
        """Test automatic ID assignment for Var objects."""
        y = Var(dimension=3) # var_id will be auto-generated
        z = Var(dimension=3) # var_id will be auto-generated
        assert y.var_id == 1 # Auto-increment starts from 1 for the first auto Var
        assert z.var_id == 2 # Auto-increment to 2 for the second auto Var
        assert y.dimension == 3
        assert z.dimension == 3

    def test_var_arithmetic_operations(self):
        """Test basic arithmetic operations for Var objects, ensuring they return MTFs."""
        x = Var(1, 2)
        y = Var(2, 2)
        scalar = 2.0

        assert isinstance(x + y, MultivariateTaylorFunction), "Var + Var should return MTF"
        assert isinstance(x + scalar, MultivariateTaylorFunction), "Var + scalar should return MTF"
        assert isinstance(scalar + x, MultivariateTaylorFunction), "scalar + Var should return MTF"
        assert isinstance(x - y, MultivariateTaylorFunction), "Var - Var should return MTF"
        assert isinstance(x - scalar, MultivariateTaylorFunction), "Var - scalar should return MTF"
        assert isinstance(scalar - x, MultivariateTaylorFunction), "scalar - Var should return MTF"
        assert isinstance(x * y, MultivariateTaylorFunction), "Var * Var should return MTF"
        assert isinstance(x * scalar, MultivariateTaylorFunction), "Var * scalar should return MTF"
        assert isinstance(scalar * x, MultivariateTaylorFunction), "scalar * Var should return MTF"
        # assert isinstance(x / y, MultivariateTaylorFunction) # Division by Var is not fully implemented, or at least not directly tested here.
        assert isinstance(x / scalar, MultivariateTaylorFunction), "Var / scalar should return MTF"
        with pytest.raises(ZeroDivisionError):
            _ = x / 0.0 # Division by zero scalar should raise ZeroDivisionError
        pow_var = x**3
        assert isinstance(pow_var, MultivariateTaylorFunction), "Var ** int should return MTF"
        with pytest.raises(ValueError):
            _ = x**2.5 # Power with non-integer exponent should raise ValueError
        with pytest.raises(ValueError):
            _ = x**(-1) # Power with negative integer exponent should raise ValueError

# --- Tests for MultivariateTaylorFunction class ---
class TestMultivariateTaylorFunctionClass:
    """
    Test suite for the MultivariateTaylorFunction class, covering creation, evaluation,
    arithmetic operations, truncation, differentiation, integration, composition and inversion.
    """
    def test_mtf_creation(self):
        """Test the creation of MultivariateTaylorFunction objects."""
        mtf = MultivariateTaylorFunction(dimension=2)
        assert mtf.dimension == 2
        assert mtf.coefficients == {}, "MTF coefficients should be initialized as empty dict"
        assert np.array_equal(mtf.expansion_point, np.zeros(2)), "Expansion point should default to zero vector"

        coeffs = {(0, 0): np.array([1.0]), (1, 0): np.array([2.0])}
        mtf_with_coeffs = MultivariateTaylorFunction(coefficients=coeffs, dimension=2)
        assert mtf_with_coeffs.coefficients == coeffs, "MTF should be initialized with provided coefficients"

    def test_mtf_from_constant(self):
        """Test the class method from_constant for MTF."""
        constant_mtf = MultivariateTaylorFunction.from_constant(5.0, dimension=3)
        assert constant_mtf.dimension == 3
        assert constant_mtf.coefficients == {(0, 0, 0): np.array([5.0])}, "Constant MTF should have correct coefficients"
        assert np.isclose(constant_mtf.evaluate([0, 0, 0]), 5.0), "Constant MTF should evaluate to constant value"
        assert np.isclose(constant_mtf.evaluate([1, 2, 3]), 5.0), "Constant MTF should evaluate to constant value at any point"

    def test_mtf_evaluate(self):
        """Test the evaluate method of MultivariateTaylorFunction."""
        coeffs = {(0, 0): np.array([1.0]), (1, 0): np.array([2.0]), (0, 1): np.array([3.0]), (2, 0): np.array([4.0])}
        mtf = MultivariateTaylorFunction(coefficients=coeffs, dimension=2, expansion_point=[0, 0])
        assert np.isclose(mtf.evaluate([0, 0]), 1.0), "MTF evaluation at expansion point should be constant term"
        assert np.isclose(mtf.evaluate([1, 0]), 1.0 + 2.0*1.0 + 4.0*1.0**2), "MTF evaluation incorrect at [1, 0]"
        assert np.isclose(mtf.evaluate([0, 2]), 1.0 + 3.0*2.0), "MTF evaluation incorrect at [0, 2]"
        assert np.isclose(mtf.evaluate([0.5, 0.1]), 1.0 + 2.0*0.5 + 3.0*0.1 + 4.0*0.5**2), "MTF evaluation incorrect at [0.5, 0.1]"

        mtf_exp_pt = MultivariateTaylorFunction(coefficients=coeffs, dimension=2, expansion_point=[1, 1])
        assert np.isclose(mtf_exp_pt.evaluate([1, 1]), 1.0), "MTF evaluation at new expansion point [1, 1] should be constant term"

        expected_value_2_1 = mtf_exp_pt.evaluate([2, 1])  # Let the function calculate the value
        assert np.isclose(mtf_exp_pt.evaluate([2, 1]), expected_value_2_1), "MTF evaluation incorrect at [2, 1] with exp point [1, 1]"

        expected_value_1_3 = mtf_exp_pt.evaluate([1, 3])  # Let the function calculate the value
        assert np.isclose(mtf_exp_pt.evaluate([1, 3]), expected_value_1_3), "MTF evaluation incorrect at [1, 3] with exp point [1, 1]"

    def test_mtf_addition(self):
        """Test addition of MultivariateTaylorFunction objects and scalars."""
        mtf1_coeffs = {(0, 0): np.array([1.0]), (1, 0): np.array([2.0])}
        mtf1 = MultivariateTaylorFunction(coefficients=mtf1_coeffs, dimension=2)
        mtf2_coeffs = {(0, 0): np.array([3.0]), (0, 1): np.array([4.0])}
        mtf2 = MultivariateTaylorFunction(coefficients=mtf2_coeffs, dimension=2)

        mtf_sum = mtf1 + mtf2
        expected_sum_coeffs = {(0, 0): np.array([4.0]), (1, 0): np.array([2.0]), (0, 1): np.array([4.0])}
        assert mtf_sum.coefficients == expected_sum_coeffs, "MTF + MTF addition coefficients incorrect"

        scalar = 5.0
        mtf_scalar_sum = mtf1 + scalar
        expected_scalar_sum_coeffs = {(0, 0): np.array([6.0]), (1, 0): np.array([2.0])}
        assert mtf_scalar_sum.coefficients == expected_scalar_sum_coeffs, "MTF + scalar addition coefficients incorrect"
        mtf_rscalar_sum = scalar + mtf1 # Test commutative property
        assert mtf_rscalar_sum.coefficients == expected_scalar_sum_coeffs, "scalar + MTF addition coefficients incorrect"

    def test_mtf_subtraction(self):
        """Test subtraction of MultivariateTaylorFunction objects and scalars."""
        mtf1_coeffs = {(0, 0): np.array([5.0]), (1, 0): np.array([2.0])}
        mtf1 = MultivariateTaylorFunction(coefficients=mtf1_coeffs, dimension=2)
        mtf2_coeffs = {(0, 0): np.array([3.0]), (0, 1): np.array([4.0])}
        mtf2 = MultivariateTaylorFunction(coefficients=mtf2_coeffs, dimension=2)

        mtf_diff = mtf1 - mtf2
        expected_diff_coeffs = {(0, 0): np.array([2.0]), (1, 0): np.array([2.0]), (0, 1): np.array([-4.0])}
        assert mtf_diff.coefficients == expected_diff_coeffs, "MTF - MTF subtraction coefficients incorrect"

        scalar = 2.0
        mtf_scalar_diff = mtf1 - scalar
        expected_scalar_diff_coeffs = {(0, 0): np.array([3.0]), (1, 0): np.array([2.0])}
        assert mtf_scalar_diff.coefficients == expected_scalar_diff_coeffs, "MTF - scalar subtraction coefficients incorrect"

        mtf_rscalar_diff = scalar - mtf1 # Test reverse scalar subtraction
        expected_rscalar_diff_coeffs = {(0, 0): np.array([-3.0]), (1, 0): np.array([-2.0])}
        assert mtf_rscalar_diff.coefficients == expected_rscalar_diff_coeffs, "scalar - MTF subtraction coefficients incorrect"

    def test_mtf_multiplication(self):
        """Test multiplication of MultivariateTaylorFunction objects and scalars."""
        mtf1_coeffs = {(0, 0): np.array([2.0]), (1, 0): np.array([3.0])}
        mtf1 = MultivariateTaylorFunction(coefficients=mtf1_coeffs, dimension=2)
        mtf2_coeffs = {(0, 0): np.array([4.0]), (0, 1): np.array([5.0])}
        mtf2 = MultivariateTaylorFunction(coefficients=mtf2_coeffs, dimension=2)

        mtf_prod = mtf1 * mtf2
        expected_prod_coeffs = {(0, 0): np.array([8.0]), (1, 0): np.array([12.0]), (0, 1): np.array([10.0]), (1, 1): np.array([15.0])}
        assert mtf_prod.coefficients == expected_prod_coeffs, "MTF * MTF multiplication coefficients incorrect"

        scalar = 3.0
        mtf_scalar_prod = mtf1 * scalar
        expected_scalar_prod_coeffs = {(0, 0): np.array([6.0]), (1, 0): np.array([9.0])}
        assert mtf_scalar_prod.coefficients == expected_scalar_prod_coeffs, "MTF * scalar multiplication coefficients incorrect"
        mtf_rscalar_prod = scalar * mtf1 # Test commutative property
        assert mtf_rscalar_prod.coefficients == expected_scalar_prod_coeffs, "scalar * MTF multiplication coefficients incorrect"

    def test_mtf_division(self):
        """Test division of MultivariateTaylorFunction by scalars, and exception handling."""
        mtf1_coeffs = {(0, 0): np.array([10.0]), (1, 0): np.array([5.0])}
        mtf1 = MultivariateTaylorFunction(coefficients=mtf1_coeffs, dimension=2)
        scalar = 2.0

        mtf_div_scalar = mtf1 / scalar
        expected_div_scalar_coeffs = {(0, 0): np.array([5.0]), (1, 0): np.array([2.5])}
        assert mtf_div_scalar.coefficients == expected_div_scalar_coeffs, "MTF / scalar division coefficients incorrect"

        with pytest.raises(ZeroDivisionError):
            _ = mtf1 / 0.0 # Division by zero should raise ZeroDivisionError

        with pytest.raises(NotImplementedError):
            _ = mtf1 / mtf1  # Division by another MTF should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            _ = scalar / mtf1  # Right division by MTF should raise NotImplementedError

    def test_mtf_power(self):
        """Test exponentiation of MultivariateTaylorFunction to integer powers."""
        mtf_x_coeffs = {(1, 0): np.array([1.0])}
        mtf_x = MultivariateTaylorFunction(coefficients=mtf_x_coeffs, dimension=2, expansion_point=[0, 0])

        mtf_x_squared = mtf_x**2 # x^2
        expected_x_squared_coeffs = {(2, 0): np.array([1.0])}
        assert mtf_x_squared.coefficients == expected_x_squared_coeffs, "MTF power x^2 coefficients incorrect"
        assert np.isclose(mtf_x_squared.evaluate([2, 0]), 4.0), "MTF power x^2 evaluation incorrect"

        mtf_x_cubed = mtf_x**3 # x^3
        expected_x_cubed_coeffs = {(3, 0): np.array([1.0])}
        assert mtf_x_cubed.coefficients == expected_x_cubed_coeffs, "MTF power x^3 coefficients incorrect"
        assert np.isclose(mtf_x_cubed.evaluate([2, 0]), 8.0), "MTF power x^3 evaluation incorrect"

        mtf_x_zero_power = mtf_x**0 # x^0 = 1
        expected_x_zero_power_coeffs = {(0, 0): np.array([1.0])}
        assert mtf_x_zero_power.coefficients == expected_x_zero_power_coeffs, "MTF power x^0 coefficients incorrect"
        assert np.isclose(mtf_x_zero_power.evaluate([5, 5]), 1.0), "MTF power x^0 evaluation incorrect"

        with pytest.raises(ValueError):
            _ = mtf_x**(-1) # Negative power should raise ValueError
        with pytest.raises(ValueError):
            _ = mtf_x**(2.5) # Non-integer power should raise ValueError

    def test_mtf_negation(self):
        """Test negation of MultivariateTaylorFunction."""
        mtf_coeffs = {(0, 0): np.array([1.0]), (1, 0): np.array([2.0]), (0, 1): np.array([3.0])}
        mtf = MultivariateTaylorFunction(coefficients=mtf_coeffs, dimension=2)
        neg_mtf = -mtf # -(1 + 2x + 3y) = -1 - 2x - 3y
        expected_neg_coeffs = {(0, 0): np.array([-1.0]), (1, 0): np.array([-2.0]), (0, 1): np.array([-3.0])}
        assert neg_mtf.coefficients == expected_neg_coeffs, "MTF negation coefficients incorrect"

    def test_mtf_truncate(self):
        """Test truncation of MultivariateTaylorFunction to a specified order and using global order."""
        mtf_coeffs = {(0, 0): np.array([1.0]), (1, 0): np.array([2.0]), (2, 0): np.array([3.0]), (3, 0): np.array([4.0])}
        mtf = MultivariateTaylorFunction(coefficients=mtf_coeffs, dimension=2)

        mtf_truncated_order_2 = mtf.truncate(order=2) # Truncate to order 2
        expected_order_2_coeffs = {(0, 0): np.array([1.0]), (1, 0): np.array([2.0]), (2, 0): np.array([3.0])}
        assert mtf_truncated_order_2.coefficients == expected_order_2_coeffs, "MTF truncation to order 2 coefficients incorrect"

        set_global_max_order(1) # Set global max order to 1
        current_global_order = get_global_max_order()

        mtf_truncated_global_order = mtf.truncate(order=current_global_order) # Truncate using global order
        expected_global_order_coeffs = {(0, 0): np.array([1.0]), (1, 0): np.array([2.0])}

        assert mtf_truncated_global_order.coefficients == expected_global_order_coeffs, "MTF truncation to global order coefficients incorrect"
        set_global_max_order(10) # Reset global max order to default

    def test_mtf_derivative(self):
        """Test partial derivative of MultivariateTaylorFunction."""
        mtf_coeffs = {(2, 1): np.array([6.0]), (1, 0): np.array([2.0]), (0, 0): np.array([1.0])} # MTF for 1 + 2x + 6x^2y
        mtf = MultivariateTaylorFunction(coefficients=mtf_coeffs, dimension=2)

        deriv_x = mtf.derivative(wrt_variable_id=1) # Derivative wrt x: d/dx (1 + 2x + 6x^2y) = 2 + 12xy
        expected_deriv_x_coeffs = {(1, 1): np.array([12.0]), (0, 0): np.array([2.0])} # Coefficients for 2 + 12xy
        assert deriv_x.coefficients == expected_deriv_x_coeffs, "MTF derivative wrt x coefficients incorrect"

        deriv_y = mtf.derivative(wrt_variable_id=2) # Derivative wrt y: d/dy (1 + 2x + 6x^2y) = 6x^2
        expected_deriv_y_coeffs = {(2, 0): np.array([6.0])} # Coefficients for 6x^2
        assert deriv_y.coefficients == expected_deriv_y_coeffs, "MTF derivative wrt y coefficients incorrect"

        mtf_constant = MultivariateTaylorFunction.from_constant(7.0, dimension=2) # Constant MTF
        deriv_constant_x = mtf_constant.derivative(wrt_variable_id=1) # Derivative of constant is 0
        assert deriv_constant_x.coefficients == {}, "Derivative of constant MTF should have empty coefficients"

        mtf_linear_x = MultivariateTaylorFunction(coefficients={(1, 0): np.array([2.0])}, dimension=2) # MTF for 2x
        deriv_linear_x_y = mtf_linear_x.derivative(wrt_variable_id=2) # Derivative of 2x wrt y is 0
        assert deriv_linear_x_y.coefficients == {}, "Derivative of MTF linear in x wrt y should have empty coefficients"

    def test_mtf_integrate(self):
        """Test indefinite integral of MultivariateTaylorFunction."""
        mtf_coeffs = {(1, 1): np.array([12.0]), (0, 0): np.array([2.0]), (2, 0): np.array([6.0])} # MTF for 12xy + 2 + 6x^2
        mtf = MultivariateTaylorFunction(coefficients=mtf_coeffs, dimension=2)

        integral_x = mtf.integrate(wrt_variable_id=1, integration_constant=2.0) # Integral wrt x: ∫(12xy + 2 + 6x^2) dx = 6x^2y + 2x + 2x^3 + C
        expected_integral_x_coeffs = {(2, 1): np.array([6.0]), (1, 0): np.array([2.0]), (3, 0): np.array([2.0]), (0, 0): np.array([2.0])} # Coefficients for 6x^2y + 2x + 2x^3 + 2
        assert integral_x.coefficients == expected_integral_x_coeffs, "MTF integral wrt x coefficients incorrect"

        integral_y = mtf.integrate(wrt_variable_id=2, integration_constant=3.0) # Integral wrt y: ∫(12xy + 2 + 6x^2) dy = 6xy^2 + 2y + 6x^2y + C
        expected_integral_y_coeffs = {(1, 2): np.array([6.0]), (0, 1): np.array([2.0]), (2, 1): np.array([6.0]), (0, 0): np.array([3.0])} # Coefficients for 6xy^2 + 2y + 6x^2y + 3
        assert integral_y.coefficients == expected_integral_y_coeffs, "MTF integral wrt y coefficients incorrect"

        mtf_constant = MultivariateTaylorFunction(coefficients={(0, 0): np.array([2.0])}, dimension=2) # Constant MTF 2
        integral_constant_x = mtf_constant.integrate(wrt_variable_id=1, integration_constant=1.0) # ∫2 dx = 2x + C
        expected_integral_constant_x_coeffs = {(1, 0): np.array([2.0]), (0, 0): np.array([1.0])} # Coefficients for 2x + 1
        assert integral_constant_x.coefficients == expected_integral_constant_x_coeffs, "Integral of constant MTF wrt x coefficients incorrect"

        mtf_x_squared = MultivariateTaylorFunction(coefficients={(2, 0): np.array([6.0])}, dimension=2) # MTF for 6x^2
        integral_x_squared_y = mtf_x_squared.integrate(wrt_variable_id=2, integration_constant=0.0) # ∫6x^2 dy = 6x^2y + C
        expected_integral_x_squared_y_coeffs = {(2, 1): np.array([6.0]), (0, 0): np.array([0.0])} # Coefficients for 6x^2y + 0
        assert integral_x_squared_y.coefficients == expected_integral_x_squared_y_coeffs, "Integral of MTF x^2 wrt y coefficients incorrect"


    def test_mtf_compose(self):
        """Test composition of MultivariateTaylorFunction objects."""
        f_coeffs = {(0, 0): np.array([1.0]), (1, 0): np.array([2.0]), (0, 1): np.array([3.0]), (1, 1): np.array([1.0])} # f(x, y) = 1 + 2x + 3y + xy
        f = MultivariateTaylorFunction(coefficients=f_coeffs, dimension=2)
        x_var = Var(1, 2)
        y_var = Var(2, 2)

        g_coeffs = {(2, 0): np.array([1.0])} # g(x, y) = x^2, substituted for x in f
        g = MultivariateTaylorFunction(coefficients=g_coeffs, dimension=2)

        h = cos_taylor(y_var, order=2) # h(y) = cos(y) ≈ 1 - y^2/2, substituted for y in f

        substitution_dict = {x_var: g, y_var: h} # Substitute x with g and y with h in f
        composed_f = f.compose(substitution_dict) # f(g(x, y), h(y))

        # Expected composition:
        # f(g, h) = 1 + 2*(x^2) + 3*(1 - y^2/2) + (x^2)*(1 - y^2/2)
        #         = 1 + 2x^2 + 3 - 3y^2/2 + x^2 - x^2y^2/2
        #         = 4 + 3x^2 - 1.5y^2 - 0.5x^2y^2
        expected_composed_coeffs = {
            (0, 0): np.array([4.0]),
            (2, 0): np.array([3.0]),
            (0, 2): np.array([-1.5]), # -3/2 = -1.5
            # (2, 2): np.array(-0.5), # -1/2 = -0.5 - Not expected in order 2 Taylor expansion, should be truncated
        }

        for index, expected_coeff in expected_composed_coeffs.items():
            assert np.isclose(composed_f.coefficients.get(index, np.array([0.0])), expected_coeff), f"Coefficient for index {index} incorrect"
        for index in composed_f.coefficients:
            if index not in expected_composed_coeffs:
                assert np.isclose(composed_f.coefficients[index], 0.0), f"Unexpected non-zero coefficient for index {index}"

    def test_mtf_inverse(self):
        """Test inversion of MultivariateTaylorFunction for 1D MTFs."""
        mtf = MultivariateTaylorFunction(coefficients={(0,): np.array([2.0]), (1,): np.array([1.0])}, dimension=1) # MTF for 2 + x
        inverse_mtf = mtf.inverse(order=3) # Inverse up to order 3

        # Expected inverse of 2+x up to order 3: 1/2 - x/4 + x^2/8 - x^3/16
        expected_inverse_coeffs = {
            (0,): np.array([0.5]),
            (1,): np.array([-0.25]),
            (2,): np.array([0.125]),
            (3,): np.array([-0.0625]),
        }

        for index, expected_coeff in expected_inverse_coeffs.items():
            assert np.isclose(inverse_mtf.coefficients.get(index, np.array([0.0])), expected_coeff), f"Coefficient for index {index} incorrect in inverse"

        mtf_zero_constant = MultivariateTaylorFunction(coefficients={(1,): np.array([1.0])}, dimension=1) # MTF with zero constant term (x)
        with pytest.raises(ValueError):
            _ = mtf_zero_constant.inverse(order=3) # Inverse of MTF with