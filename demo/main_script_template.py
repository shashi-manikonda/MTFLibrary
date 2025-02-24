# demo/main_script_template.py
from MTFLibrary.taylor_function import initialize_mtf_globals, set_global_etol, get_tabular_string
from MTFLibrary.taylor_function import MultivariateTaylorFunction, ComplexMultivariateTaylorFunction
from MTFLibrary.elementary_functions import exp_taylor, sin_taylor, cos_taylor
from MTFLibrary.MTFExtended import Var
import numpy as np

if __name__ == "__main__":
    # Initialize global settings
    initialize_mtf_globals(max_order=10, max_dimension=3)
    set_global_etol(1e-9)

    # Define variables
    x = Var(1)
    y = Var(2)

    # Example 1: Create a simple Multivariate Taylor Function
    coefficients_simple = {
        (0, 0): np.array([1.0]),
        (1, 0): np.array([2.0]),
        (0, 1): np.array([3.0]),
        (1, 1): np.array([4.0])
    }
    mtf_simple = MultivariateTaylorFunction(coefficients_simple, dimension=2)
    
    print("Example 1: Simple MTF")
    mtf_simple.print_tabular()
    print(f"MTF at [0.1, 0.2]: {mtf_simple(np.array([0.1, 0.2]))}\n")

    # Example 2: Arithmetic operations
    mtf_const = MultivariateTaylorFunction.from_constant(5.0, dimension=2)
    mtf_sum = mtf_simple + mtf_const
    mtf_product = mtf_simple * mtf_const

    print("Example 2: Arithmetic Operations")
    print("mtf_sum:")
    mtf_sum.print_tabular()
    print("mtf_product:")
    mtf_product.print_tabular()
    print("\n")

    # Example 3: Elementary functions
    mtf_exp = exp_taylor(mtf_simple)
    mtf_sin = sin_taylor(x)
    mtf_cos = cos_taylor(y)

    print("Example 3: Elementary Functions")
    print("exp_taylor(mtf_simple):")
    mtf_exp.truncate(2).print_tabular() # Truncate to order 2 for display
    print("sin_taylor(x):")
    sin_taylor(x).truncate(2).print_tabular()
    print("cos_taylor(y):")
    cos_taylor(y).truncate(2).print_tabular()
    print("\n")

    # Example 4: Complex MTF
    cmtf_const = ComplexMultivariateTaylorFunction.from_constant(1+1j, dimension=2)
    cmtf_real_to_complex = mtf_simple.to_complex_mtf()
    cmtf_sum = cmtf_real_to_complex + cmtf_const

    print("Example 4: Complex MTF")
    print("cmtf_sum:")
    cmtf_sum.truncate(1).print_tabular() # Truncate to order 1 for display
    print("\n")


    # Example Usage (assuming you have MTF and CMTF classes and instances)
    # Dummy get_global_max_order and get_global_etol for example
    def get_global_max_order():
        return 2
    def get_global_etol():
        return 1e-10

    class MockMTF: # Mock MultivariateTaylorFunction class for example
        def __init__(self, coefficients, dimension):
            self.coefficients = coefficients
            self.dimension = dimension
        def get_global_max_order(self): # Mock get_global_max_order if needed as instance method
            return get_global_max_order()

    class MockCMTF: # Mock ComplexMultivariateTaylorFunction class for example
        def __init__(self, coefficients, dimension):
            self.coefficients = coefficients
            self.dimension = dimension
        def get_global_max_order(self): # Mock get_global_max_order if needed as instance method
            return get_global_max_order()


    # Example coefficients for COMPLEX MTF (CMTF)
    complex_coefficients = {
        (0, 0): np.array([1.0 + 0.5j]),
        (1, 0): np.array([2.0 - 1.0j]),
        (0, 1): np.array([-0.5 + 2.0j]),
        (2, 0): np.array([0.5]),
        (1, 1): np.array([0.25j]),
        (0, 2): np.array([0.1]),
    }
    complex_dimension = 2
    cmtf_instance = MockCMTF(coefficients=complex_coefficients, dimension=complex_dimension) # Use MockCMTF

    table_string_complex = get_tabular_string(cmtf_instance, order=2) # Pass CMTF instance
    print("--- Complex MTF Table (Instance Input) ---")
    print(table_string_complex)


    # Example coefficients for REAL MTF (MTF)
    real_coefficients = {
        (0, 0): np.array([2.0]),
        (1, 0): np.array([3.0]),
        (0, 1): np.array([-1.0]),
        (2, 0): np.array([0.8]),
    }
    real_dimension = 2
    mtf_instance = MockMTF(coefficients=real_coefficients, dimension=real_dimension) # Use MockMTF

    table_string_real = get_tabular_string(mtf_instance, order=2) # Pass MTF instance
    print("\n--- Real MTF Table (Instance Input) ---")
    print(table_string_real)


    print("Demo script completed. Explore more functionalities by checking unit tests and library documentation.")