import subprocess
import os
import numpy as np
import pandas as pd
import time
from MTFLibrary import *
import cProfile
import pstats

# Mapping of MTFLibrary elementary function names to COSY function names
MTF_TO_COSY_FUNCTIONS = {
    "sin_taylor": "SIN",
    "cos_taylor": "COS",
    "exp_taylor": "EXP",
    "sqrt_taylor": "SQRT",
    "log_taylor": "LOG",
    "arctan_taylor": "ATAN",
    "tan_taylor": "TAN",
    "arcsin_taylor": "ASIN",
    "arccos_taylor": "ACOS",
    "sinh_taylor": "SINH",
    "cosh_taylor": "COSH",
    "tanh_taylor": "TANH",
    # "arctanh_taylor": "ATANH",
}

def convert_mtf_expression_to_cosy(expression_string):
    """Converts MTF expression string to COSY expression string."""
    expression_string = expression_string.strip()
    if expression_string.startswith('(') and expression_string.endswith(')'):
        balance = 0
        top_level_parentheses = True
        for i, char in enumerate(expression_string):
            if char == '(': balance += 1
            elif char == ')': balance -= 1
            if balance == 0 and i < len(expression_string) - 1: top_level_parentheses = False; break
        if balance == 0 and top_level_parentheses:
            inner_expression = expression_string[1:-1]
            cosy_inner = convert_mtf_expression_to_cosy(inner_expression)
            return f"({cosy_inner})"

    if expression_string in ["x", "y", "z", "u"]: return f"DA({['x', 'y', 'z', 'u'].index(expression_string) + 1})"
    try: return str(float(expression_string))
    except ValueError: pass

    if '**' in expression_string:
        split_expr = split_expression_by_operator(expression_string, '**')
        if split_expr:
            base_str, exponent_str = split_expr
            return f"POW({convert_mtf_expression_to_cosy(base_str)}, {convert_mtf_expression_to_cosy(exponent_str)})"

    for operator in ['*', '/']:
        if split_expression_by_operator(expression_string, operator):
            left_arg_str, right_arg_str = split_expression_by_operator(expression_string, operator)
            cosy_left = convert_mtf_expression_to_cosy(left_arg_str)
            cosy_right = convert_mtf_expression_to_cosy(right_arg_str)
            return f"({cosy_left} {operator} {cosy_right})" if operator == '*' else f"({cosy_left} / ({cosy_right}))"

    for operator in ['+', '-']:
        if split_expression_by_operator(expression_string, operator):
            left_arg_str, right_arg_str = split_expression_by_operator(expression_string, operator)
            cosy_left = convert_mtf_expression_to_cosy(left_arg_str)
            cosy_right = convert_mtf_expression_to_cosy(right_arg_str)
            return f"({cosy_left} {operator} {cosy_right})"

    if '(' in expression_string and ')' in expression_string:
        func_name = expression_string[:expression_string.find('(')].strip()
        if func_name in MTF_TO_COSY_FUNCTIONS:
            cosy_func_name = MTF_TO_COSY_FUNCTIONS[func_name]
            args_str = expression_string[expression_string.find('(')+1:expression_string.rfind(')')]
            cosy_args = ", ".join([convert_mtf_expression_to_cosy(arg.strip()) for arg in split_arguments(args_str)])
            return f"{cosy_func_name}({cosy_args})"
        else: raise ValueError(f"COSY conversion not implemented for function: {func_name}")

    try: float(expression_string); return expression_string
    except ValueError: pass
    if expression_string in ["x", "y", "z", "u"]: return f"DA({['x', 'y', 'z', 'u'].index(expression_string) + 1})"
    raise TypeError(f"Unsupported expression type: '{expression_string}'")

def split_expression_by_operator(expression_string, operator):
    """Splits expression by operator respecting parentheses."""
    balance = 0; split_index = -1
    if operator in ['+', '-']:
        for i, char in enumerate(expression_string):
            if char == '(': balance += 1
            elif char == ')': balance -= 1
            elif char == operator and balance == 0: split_index = i; break
    elif operator in ['**', '*', '/']:
        for i in range(len(expression_string) - 1, -1, -1):
            char = expression_string[i]
            if char == ')': balance += 1
            elif char == '(': balance -= 1
            elif operator == '**' and i + 1 < len(expression_string) and expression_string[i:i+2] == '**' and balance == 0: split_index = i; break
            elif operator != '**' and char == operator and balance == 0: split_index = i; break

    if split_index != -1:
        left_arg_str = expression_string[:split_index].strip()
        right_arg_str = expression_string[split_index+ len(operator):].strip()
        return left_arg_str, right_arg_str
    return None

def split_arguments(arguments_string):
    """Splits function arguments by commas, respecting parentheses."""
    args = []; arg_level = 0; current_arg = ''
    if not arguments_string: return args
    for char in arguments_string:
        if char == ',' and arg_level == 0: args.append(current_arg.strip()); current_arg = ''
        else: current_arg += char;
        if char == '(': arg_level += 1
        elif char == ')': arg_level -= 1
    args.append(current_arg.strip())
    return args

def extract_cosy_coefficients(cosy_output, dimension):
    """
    Parses COSY output and extracts Taylor expansion coefficients.
    Returns a dictionary where keys are exponent tuples and values are coefficients.
    """
    coefficients = {}
    lines = cosy_output.strip().split('\n')
    start_extracting = False
    for line in lines:
        line = line.strip()
        if line.startswith("I  COEFFICIENT"):
            start_extracting = True
            continue
        if line.startswith("----") and start_extracting: # End of coefficient table
            break
        if start_extracting and line:
            parts = line.split()
            try:
                coefficient = float(parts[1])
                order = int(parts[2]) # Order is not directly used now, exponents are more direct
                exponents = tuple(int(exp) for exp in parts[3:]) # Extract exponent tuple
                coefficients[exponents] = coefficient
            except (IndexError, ValueError):
                continue # Skip lines that don't fit expected coefficient format
    return coefficients



def get_mtf_coefficients(mtf_expression):
    """Evaluates MTF expression and returns coefficients."""
    mtf_function = convert_to_mtf(mtf_expression); coefficients_mtf = {}
    for exponents, coefficient_value in mtf_function.coefficients.items(): # Changed variable name
        coefficients_mtf[exponents] = coefficient_value # Direct assignment, no indexing
    return coefficients_mtf

def extract_cosy_timing(cosy_output):
    """Parses COSY output and extracts timing information."""
    for line in cosy_output.strip().split('\n'):
        if "CPU_TIME=" in line:
            time_str = line.split('=')[1].strip()
            try:
                return float(time_str)
            except ValueError:
                print(f"Warning: Could not parse CPU_TIME value: {time_str}")
                return np.nan
    print("Warning: CPU_TIME not found in COSY output.")
    return np.nan

BENCHMARK_COSY_SCRIPT_FILENAME = "benchmark_cosy_script.fox" # Define constant filename

def create_cosy_script(function_name, variable_names, cosy_expression_str, order=None, dimension=None, filename=BENCHMARK_COSY_SCRIPT_FILENAME, num_iterations=10000):
    """Generates and runs COSY script, returns output and time, line length < 80."""
    if order is None: order=get_global_max_order();
    if dimension is None: dimension=get_global_max_dimension()
    if dimension != len(variable_names[:dimension]):
        raise ValueError("Dimension must match variables count.")

    cosy_script = f"""
INCLUDE 'COSY';

PROCEDURE RUN;
VARIABLE ORDER 1;
VARIABLE DIM 1;
VARIABLE NM1 1;
PROCEDURE TESTEXP NM1;
    VARIABLE TEMP NM1;
    VARIABLE I 1;

    FUNCTION POW R N; VARIABLE I 2; VARIABLE TEMP NM1; TEMP:=1;
        IF N#0; LOOP I 1 N 1; TEMP:=TEMP*R;ENDLOOP; ENDIF;
        IF N=0; TEMP:=1; ENDIF; POW:=TEMP; ENDFUNCTION;

    WRITE 6 '{function_name}';
    LOOP I 1 {num_iterations};
        TEMP:={cosy_expression_str};
    ENDLOOP;
    WRITE 6 TEMP;
ENDPROCEDURE;

ORDER := {order}; {{ORDER OF COMPUTATION}}
DIM := {dimension}; {{number of variables}}

DAINI ORDER DIM 0 NM1;
WRITE 6 NM1; {{print number of monomials for given order and dimension}}

TESTEXP NM1;

ENDPROCEDURE;

RUN;
END;
"""

    # Basic line length check
    lines = cosy_script.splitlines()
    for line in lines:
        if len(line) > 80:
            print(f"Warning: COSY script line exceeds 80 chars:\n{line}")

    fox_filepath = filename; dat_filepath = "foxyinp.dat"
    with open(fox_filepath, "w") as f: f.write(cosy_script)
    with open(dat_filepath, "w") as f_dat: f_dat.write(os.path.splitext(filename)[0])

    cosy_exe_path = r"C:\Users\manik\Work\MTFLibrary\demo\cosy\cosy.exe" # Your COSY path

    start_time_cosy_python = time.perf_counter() # Start Python timer for COSY execution
    try:
        with open(dat_filepath, 'r') as dat_file:
            process = subprocess.run([cosy_exe_path], stdin=dat_file, capture_output=True, text=True, check=True)
        cosy_output = process.stdout
        cosy_execution_time_python = time.perf_counter() - start_time_cosy_python # End Python timer
        print(f"COSY executed successfully for {function_name}.")
        # cosy_timing_from_output = extract_cosy_timing(cosy_output) # Keep extracting COSY time from output for comparison
        return cosy_output, cosy_execution_time_python # Return Python-measured time

    except subprocess.CalledProcessError as e: print(f"COSY execution failed for {function_name}! Error:\n{e}"); print(f"COSY stderr:\n{e.stderr}"); return None, None
    except FileNotFoundError: print(f"Error: COSY executable not found at: {cosy_exe_path}"); return None, None

def compare_coefficients(mtf_coefficients, cosy_coefficients, function_name, show_coefficient_comparison=False):
    """Compares coefficients, calculates RMSE, and optionally shows detail."""
    all_exponents = set(cosy_coefficients.keys()) | set(mtf_coefficients.keys()); squared_errors = []; comparison_data = []
    for exponents in sorted(list(all_exponents)):
        cosy_coeff = cosy_coefficients.get(exponents, 0.0); mtf_coeff = mtf_coefficients.get(exponents, 0.0)
        difference = cosy_coeff - mtf_coeff; squared_errors.append(difference**2)
        comparison_data.append({'Exponents': str(exponents), 'COSY Coeff': cosy_coeff, 'MTFLibrary Coeff': mtf_coeff, 'Difference': difference})

    rmse_value = np.sqrt(np.mean(squared_errors))
    comparison_df = pd.DataFrame(comparison_data) if show_coefficient_comparison else None
    if show_coefficient_comparison: print(f"\n--- Coefficient Comparison for {function_name} ---\n{comparison_df.to_string(index=False)}")
    return rmse_value, comparison_df

def benchmark_functions(function_expressions, taylor_order, dimensions, variable_names, show_coefficient_comparison=False, num_iterations=10000):
    """Benchmarks MTF functions against COSY, including timing ratio."""
    rmse_results = []
    for function_name, mtf_expression_str, cosy_expression_str in function_expressions:
        print(f"\n--- Benchmarking {function_name}: {mtf_expression_str} ---")

        start_time_mtf = time.perf_counter() # Use perf_counter for MTF timing
        for _ in range(num_iterations): # Loop MTF calculation for timing
            mtf_temp = eval(mtf_expression_str)
        mtf_execution_time = (time.perf_counter() - start_time_mtf)
        mtf_coefficients = get_mtf_coefficients(mtf_temp)

        cosy_output_text, cosy_execution_time_python = create_cosy_script( # Get Python measured COSY time
            function_name=function_name, variable_names=variable_names,
            cosy_expression_str=cosy_expression_str, order=taylor_order, dimension=dimensions,
            filename=BENCHMARK_COSY_SCRIPT_FILENAME, num_iterations=num_iterations # Use constant filename
        )

        rmse_value = np.nan
        timing_ratio = np.nan # Initialize timing_ratio to NaN
        if cosy_output_text:
            cosy_coefficients = extract_cosy_coefficients(cosy_output_text, dimensions)
            rmse_value, comparison_df = compare_coefficients(mtf_coefficients, cosy_coefficients, function_name, show_coefficient_comparison)
            if not np.isnan(cosy_execution_time_python) and cosy_execution_time_python != 0: # Use Python COSY time
                timing_ratio = mtf_execution_time / cosy_execution_time_python
            else:
                timing_ratio = np.nan # Ratio is NaN if cosy_timing is NaN or 0


        rmse_results.append({'Function': function_name, 'RMSE': rmse_value,
                            'MTF Time (s)': mtf_execution_time, 'COSY Time (s)': cosy_execution_time_python, # Store Python COSY time
                            'Timing Ratio (MTF/COSY)': timing_ratio}) # Added Timing Ratio

        print(f"\n--- RMSE for {function_name} ---\nRMSE: {rmse_value:.16e}")
        print(f"MTFLibrary Time ({num_iterations} iterations): {mtf_execution_time:.4f} seconds")
        if not np.isnan(cosy_execution_time_python): # Use Python COSY time for output
            print(f"COSY Time (Python Measured, {num_iterations} iterations): {cosy_execution_time_python:.4f} seconds") # Indicate Python measured
            print(f"Timing Ratio (MTF/COSY): {timing_ratio:.2f}") # Print the ratio
        else:
            print("COSY Timing: COSY execution failed, timing not available.")
            print("Timing Ratio (MTF/COSY): Not Available (COSY failed)") # Indicate ratio NA due to COSY fail


    rmse_summary_df = pd.DataFrame(rmse_results)
    return rmse_summary_df

if __name__ == "__main__":
    taylor_order = 8; dimensions = 6
    initialize_mtf_globals(max_order=taylor_order, max_dimension=dimensions); set_global_etol(1e-16)
    x = Var(1); y = Var(2); z = Var(3); 
    u = Var(4); v = Var(5); w = Var(6); 
    variable_names = ["x", "y", "z", "u", "v", "w"]
    num_iterations = int(1)


    benchmark_expressions = [
        ################################################################################
        # Multiplication Intensive Tests
        ################################################################################
        ("mul_intensive_1", "(x + y) * (x - y)", "(DA(1)+DA(2))*(DA(1)-DA(2))"),
        ("mul_intensive_2", "(x + 1)**3", "pow((DA(1)+1),3)"), # Power involves repeated multiplication
        ("mul_intensive_3", "(x * y * z)**2", "POW((DA(1)*DA(2)*DA(3)),2)"),
        ("mul_intensive_4", "x * y * x * y", "DA(1)*DA(2)*DA(1)*DA(2)"), # Repeated variable multiplication
        ("mul_intensive_5", "(x + y + z + u )**2", "POW( (DA(1)+DA(2)+DA(3)+DA(4)),2)"), # Higher dimension multiplication

        ################################################################################
        # Basic Operation Tests (Verification mostly in test_taylor_operations.py)
        ################################################################################
        ("op_add", "x + y", "DA(1)+DA(2)"),
        ("op_sub", "x - y", "DA(1)-DA(2)"),
        ("op_mul", "2*x * y", "2*DA(1)*DA(2)"),
        ("op_div", "x / (y+2)", "DA(1)/(DA(2)+2)"),
        ("op_neg", "-x", "-DA(1)"),
        ("op_const_add", "x + 5", "DA(1)+5"),
        ("op_const_mul", "3*x", "3*DA(1)"),
        ("op_power", "x**2", "DA(1)*DA(1)"),

        ################################################################################
        # Elementary Function Tests (Focus on accuracy and correctness)
        ################################################################################

        ##### Univariate Functions #####
        # Sine Taylor
        ("sin_taylor", "sin_taylor(0.5+x+y)", "sin(0.5+DA(1)+DA(2))"), # Already has constant
        ("sin_taylor", "sin_taylor(0.3+x)", "sin(0.3+DA(1))"), # Modified to include constant
        ("sin_taylor", "sin_taylor(0.2*x - 0.1*y)", "sin(0.2*DA(1)-0.1*DA(2))"),

        # Cosine Taylor
        ("cos_taylor", "cos_taylor(0.5+x+y)", "cos(0.5+DA(1)+DA(2))"), # Already has constant
        ("cos_taylor", "cos_taylor(0.4+x)", "cos(0.4+DA(1))"), # Modified to include constant
        ("cos_taylor", "cos_taylor(0.2*x - 0.1*y)", "cos(0.2*DA(1)-0.1*DA(2))"),

        # Tangent Taylor
        ("tan_taylor", "tan_taylor(0.2+0.1*x)", "tan(0.2+0.1*DA(1))"), # Modified to include constant
        ("tan_taylor", "tan_taylor(0.5*x + 0.2*y)", "tan(0.5*DA(1)+0.2*DA(2))"),
        ("tan_taylor", "tan_taylor(-0.1*x)", "tan(-0.1*DA(1))"),

        # Exponential Taylor
        ("exp_taylor", "exp_taylor(0.1+0.1*x+0.05*y)", "exp(0.1+0.1*DA(1)+0.05*DA(2))"), # Modified to include constant
        ("exp_taylor", "exp_taylor(x-0.5)", "exp(DA(1)-0.5)"),
        ("exp_taylor", "exp_taylor(2*x)", "exp(2*DA(1))"),

        # Square Root Taylor
        ("sqrt_taylor", "sqrt_taylor(1.5+0.1*x)", "sqrt(1.5+0.1*DA(1))"), # Modified to include constant
        ("sqrt_taylor", "sqrt_taylor(4+0.2*x)", "sqrt(4+0.2*DA(1))"),
        ("sqrt_taylor", "sqrt_taylor(1-0.05*x)", "sqrt(1-0.05*DA(1))"),

        # Logarithm Taylor
        ("log_taylor", "log_taylor(3.5+0.1*x)", "log(3.5+0.1*DA(1))"), # Modified to include constant
        ("log_taylor", "log_taylor(1+0.2*x + 0.1*y)", "log(1+0.2*DA(1)+0.1*DA(2))"),
        ("log_taylor", "log_taylor(2-0.1*x)", "log(2-0.1*DA(1))"),

        # ArcTangent Taylor
        ("arctan_taylor", "arctan_taylor(0.2+x+0.1*y)", "atan(0.2+DA(1)+0.1*DA(2))"), # Modified to include constant
        ("arctan_taylor", "arctan_taylor(0.5*x - 0.3*z)", "atan(0.5*DA(1)-0.3*DA(3))"),
        ("arctan_taylor", "arctan_taylor(-0.2*x)", "atan(-0.2*DA(1))"),

        # Hyperbolic Sine Taylor
        ("sinh_taylor", "sinh_taylor(0.3+0.5*x)", "sinh(0.3+0.5*DA(1))"), # Modified to include constant
        ("sinh_taylor", "sinh_taylor(x+0.2*y)", "sinh(DA(1)+0.2*DA(2))"),
        ("sinh_taylor", "sinh_taylor(-0.3*x)", "sinh(-0.3*DA(1))"),

        # Hyperbolic Cosine Taylor
        ("cosh_taylor", "cosh_taylor(0.6+x)", "cosh(0.6+DA(1))"), # Modified to include constant
        ("cosh_taylor", "cosh_taylor(x)", "cosh(DA(1))"),
        ("cosh_taylor", "cosh_taylor(0.3*x - 0.1)", "cosh(0.3*DA(1)-0.1)"),

        # Hyperbolic Tangent Taylor
        ("tanh_taylor", "tanh_taylor(0.15+0.1*x)", "tanh(0.15+0.1*DA(1))"), # Modified to include constant
        ("tanh_taylor", "tanh_taylor(0.4*x + 0.1*y)", "tanh(0.4*DA(1)+0.1*DA(2))"),
        ("tanh_taylor", "tanh_taylor(-0.05*x)", "tanh(-0.05*DA(1))"),

        # ArcSine Taylor
        ("arcsin_taylor", "arcsin_taylor(0.05+0.1*x)", "asin(0.05+0.1*DA(1))"), # Modified to include constant
        ("arcsin_taylor", "arcsin_taylor(0.5*x - 0.2*y)", "asin(0.5*DA(1)-0.2*DA(2))"),
        ("arcsin_taylor", "arcsin_taylor(0.8*x)", "asin(0.8*DA(1))"), # Closer to domain boundary

        # ArcCosine Taylor
        ("arccos_taylor", "arccos_taylor(0.05+0.1*x)", "acos(0.05+0.1*DA(1))"), # Modified to include constant
        ("arccos_taylor", "arccos_taylor(0.5*x - 0.2*y)", "acos(0.5*DA(1)-0.2*DA(2))"),
        ("arccos_taylor", "arccos_taylor(0.8*x)", "acos(0.8*DA(1))"),
    ]

    show_coefficient_details = False

    rmse_summary_table = benchmark_functions(
        benchmark_expressions, taylor_order, dimensions, variable_names, show_coefficient_details, num_iterations=num_iterations
    )

    print("\n--- RMSE and Timing Summary Table ---")
    print(rmse_summary_table.to_string(index=False))