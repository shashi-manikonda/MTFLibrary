import pytest
import numpy as np
import mtflib
from mtflib import TaylorMap, MTF

@pytest.fixture(scope="module", autouse=True)
def setup_mtf_module():
    """Initializes mtflib globals for the test module."""
    if not mtflib.get_mtf_initialized_status():
        mtflib.initialize_mtf_globals(max_order=5, max_dimension=3)

@pytest.fixture
def sample_maps():
    """Creates some sample TaylorMap objects for testing."""
    # Map 1: R^2 -> R^2, F(x,y) = [1+x, 2+y]
    f1 = MTF.from_variable(1, 2) + 1
    f2 = MTF.from_variable(2, 2) + 2
    map1 = TaylorMap([f1, f2])

    # Map 2: R^2 -> R^2, G(x,y) = [x*y, x+y]
    x = MTF.from_variable(1, 2)
    y = MTF.from_variable(2, 2)
    g1 = x * y
    g2 = x + y
    map2 = TaylorMap([g1, g2])

    # Map 3: R^2 -> R^3, H(x,y) = [x, y, x+y]
    h1 = MTF.from_variable(1, 2)
    h2 = MTF.from_variable(2, 2)
    h3 = h1 + h2
    map3 = TaylorMap([h1, h2, h3])

    return map1, map2, map3

def test_constructor(sample_maps):
    map1, _, _ = sample_maps
    assert isinstance(map1, TaylorMap)
    assert map1.map_dim == 2
    assert len(map1.components) == 2
    assert isinstance(map1.get_component(0), MTF)

def test_addition(sample_maps):
    map1, map2, _ = sample_maps
    result = map1 + map2

    # Expected: [1+x+xy, 2+y+x+y] = [1+x+xy, 2+x+2y]
    x = MTF.from_variable(1, 2)
    y = MTF.from_variable(2, 2)

    expected_c1 = 1 + x + x*y
    expected_c2 = 2 + y + x + y

    assert result.map_dim == 2
    assert result.get_component(0) == expected_c1
    assert result.get_component(1) == expected_c2

def test_addition_dim_mismatch(sample_maps):
    map1, _, map3 = sample_maps
    with pytest.raises(ValueError, match="TaylorMap dimensions must match for addition."):
        map1 + map3

def test_composition():
    # F: R^2 -> R^2, F(x,y) = [x+y, x-y]
    x = MTF.from_variable(1, 2)
    y = MTF.from_variable(2, 2)
    f1 = x + y
    f2 = x - y
    mapF = TaylorMap([f1, f2])

    # G: R^1 -> R^2, G(t) = [t^2, 2t]
    t = MTF.from_variable(1, 1)
    g1 = t**2
    g2 = 2*t
    mapG = TaylorMap([g1, g2])

    # Compose F(G(t))
    # F(g1, g2) = [g1+g2, g1-g2] = [t^2 + 2t, t^2 - 2t]
    result = mapF.compose(mapG)

    expected_c1 = t**2 + 2*t
    expected_c2 = t**2 - 2*t

    assert result.map_dim == 2
    assert result.get_component(0) == expected_c1
    assert result.get_component(1) == expected_c2
    # The new map should have input dimension 1
    assert result.get_component(0).dimension == 1

def test_composition_dim_mismatch(sample_maps):
    map1, _, map3 = sample_maps
    # map1 input dim (2) != map3 output dim (3)
    with pytest.raises(ValueError, match="Cannot compose maps"):
        map1.compose(map3)

def test_trace(sample_maps):
    map1, _, _ = sample_maps
    # F(x,y) = [1+x, 2+y]
    # Jacobian = [[1, 0], [0, 1]]
    # Trace of linear part = 1 + 1 = 2
    assert map1.trace() == 2.0

    # Test map2 separately to avoid fixture pollution issues
    x = MTF.from_variable(1, 2)
    y = MTF.from_variable(2, 2)
    map2 = TaylorMap([x * y, x + y])

    # G(x,y) = [xy, x+y]
    # Jacobian = [[y, x], [1, 1]]
    # At (0,0), Jacobian is [[0,0],[1,1]]
    # Trace of the linear part is d(xy)/dx|0 + d(x+y)/dy|0 = 0 + 1 = 1
    assert map2.trace() == 1.0

def test_get_set_coefficient(sample_maps):
    map1_orig, _, _ = sample_maps
    # Create a deep copy for modification to avoid test pollution
    map1 = TaylorMap([c.copy() for c in map1_orig.components])

    # F(x,y) = [1+x, 2+y]
    # Component 0 is 1+x. Coeff of x is 1. x is exp (1,0)
    assert map1.get_coefficient(0, np.array([1, 0])) == 1.0

    map1.set_coefficient(0, np.array([1, 0]), 5.0)
    assert map1.get_coefficient(0, np.array([1, 0])) == 5.0

    # Reset for other tests
    map1.set_coefficient(0, np.array([1, 0]), 1.0)
    assert map1.get_coefficient(0, np.array([1,0])) == 1.0

def test_substitute_partial(sample_maps):
    map1, _, _ = sample_maps
    # map1 is F(x,y) = [1+x, 2+y]
    # Substitute x=3 (var_index=1)
    result_map = map1.substitute({1: 3})

    # Expected: F(3,y) = [4, 2+y]
    y = MTF.from_variable(2, 2)
    expected_c1 = MTF.from_constant(4.0, dimension=2)
    expected_c2 = 2 + y

    assert isinstance(result_map, TaylorMap)
    assert result_map.get_component(0) == expected_c1
    assert result_map.get_component(1) == expected_c2

def test_substitute_full(sample_maps):
    map1, _, _ = sample_maps
    # map1 is F(x,y) = [1+x, 2+y]
    # Substitute x=3, y=5
    result = map1.substitute({1: 3, 2: 5})

    # Expected: F(3,5) = [4, 7]
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, [4.0, 7.0])

def test_variable_creation_bug():
    """
    This test is to isolate a suspected bug where creating a variable
    might be incorrectly adding a constant term.
    """
    y = MTF.from_variable(2, 2)

    # A variable should not have a constant term.
    # The constant term corresponds to an exponent tuple of all zeros.
    constant_term = y.extract_coefficient(tuple([0, 0])).item()
    assert constant_term == 0.0, "A variable created with from_variable should not have a constant term."

    x = MTF.from_variable(1, 2)
    prod = x * y

    # The product of x and y should be xy. It should not contain a linear x term.
    # A linear x term would have an exponent of (1, 0).
    x_term_in_prod = prod.extract_coefficient(tuple([1, 0])).item()
    assert x_term_in_prod == 0.0, "The product x*y should not contain a linear x term."

def test_trace_standalone():
    """
    A standalone test for the trace of a specific map to isolate
    potential fixture-related bugs.
    """
    x = MTF.from_variable(1, 2)
    y = MTF.from_variable(2, 2)

    g1 = x * y
    g2 = x + y

    map_to_test = TaylorMap([g1, g2])

    # Trace of [x*y, x+y] should be d(xy)/dx + d(x+y)/dy at origin, which is 0 + 1 = 1.
    assert map_to_test.trace() == 1.0
