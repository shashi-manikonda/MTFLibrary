import pytest
import numpy as np
import mtflib
from mtflib import TaylorMap, MTF

def test_map_inversion_from_demo():
    """
    This test replicates the map inversion demo from taylor_map_demo.ipynb
    to verify if it's broken after recent changes.
    """
    # 1. Initialize mtflib
    try:
        mtflib.initialize_mtf_globals(max_order=4, max_dimension=2)
    except RuntimeError:
        pass # Already initialized

    # 2. Create the invertible map
    x = MTF.from_variable(1, 2)
    y = MTF.from_variable(2, 2)
    f1 = x + 0.1 * y**2
    f2 = y - 0.1 * x**2
    map_to_invert = TaylorMap([f1, f2])

    # 3. Invert the map
    inverted_map = map_to_invert.invert()

    # 4. Verify by composing F and F_inv
    composition = inverted_map.compose(map_to_invert)

    # 5. The result should be the identity map [x, y]
    identity_map = TaylorMap([x, y])

    # 6. Check that the components are equal
    # We need to truncate both to the working order for a fair comparison
    max_order = mtflib.get_global_max_order()
    composition_trunc = composition.truncate(max_order)
    identity_trunc = identity_map.truncate(max_order)

    assert composition_trunc.get_component(0) == identity_trunc.get_component(0)
    assert composition_trunc.get_component(1) == identity_trunc.get_component(1)
