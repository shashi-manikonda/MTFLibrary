# MTFLibrary/__init__.py

from . import taylor_function
from . import elementary_functions
from . import MTFExtended

from .taylor_function import initialize_mtf_globals, set_global_etol, get_global_max_order, get_global_max_dimension
from .MTFExtended import Var, MTFExtended #Import Var and MTFExtended from MTFExtended.py module
from .taylor_function import MultivariateTaylorFunction, convert_to_mtf, ComplexMultivariateTaylorFunction #Import core classes from taylor_function module
from .elementary_functions import * # Import all elementary functions

from . import EMLibrary # Add this line to expose EMLibrary

__all__ = [
    'taylor_function',
    'elementary_functions',
    'MTFExtended',
    'initialize_mtf_globals',
    'set_global_etol',
    'get_global_max_order',
    'get_global_max_dimension',
    'Var',
    'MTFExtended',
    'MultivariateTaylorFunction',
    'convert_to_mtf',
    'ComplexMultivariateTaylorFunction'
    # ... (you can add elementary functions to __all__ if you want them directly accessible from MTFLibrary) ...
]