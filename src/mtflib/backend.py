import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

class NumpyBackend:
    @staticmethod
    def power(base, exp):
        return np.power(base, exp)

    @staticmethod
    def prod(a, axis=None):
        return np.prod(a, axis=axis)

    @staticmethod
    def dot(a, b):
        return np.dot(a, b)

    @staticmethod
    def zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def atleast_2d(a):
        return np.atleast_2d(a)

    @staticmethod
    def from_numpy(a):
        return np.array(a)

    @staticmethod
    def to_numpy(a):
        return np.array(a)

if _TORCH_AVAILABLE:
    class TorchBackend:
        @staticmethod
        def power(base, exp):
            return torch.pow(base, exp)

        @staticmethod
        def prod(a, axis=None):
            return torch.prod(a, dim=axis)

        @staticmethod
        def dot(a, b):
            return torch.matmul(a, b)

        @staticmethod
        def zeros(shape, dtype=None):
            return torch.zeros(shape, dtype=dtype)

        @staticmethod
        def atleast_2d(a):
            if a.dim() >= 2:
                return a
            return a.unsqueeze(0)

        @staticmethod
        def from_numpy(a):
            return torch.from_numpy(a)

        @staticmethod
        def to_numpy(a):
            return a.numpy()

_backends = {
    np.ndarray: NumpyBackend,
}
if _TORCH_AVAILABLE:
    _backends[torch.Tensor] = TorchBackend

def get_backend(array):
    """
    Returns the backend module for a given array type.
    """
    array_type = type(array)
    if array_type in _backends:
        return _backends[array_type]()

    # Fallback for subclasses
    if isinstance(array, np.ndarray):
        return NumpyBackend()
    if _TORCH_AVAILABLE and isinstance(array, torch.Tensor):
        return TorchBackend()

    raise TypeError(f"Unsupported array type: {array_type}")
