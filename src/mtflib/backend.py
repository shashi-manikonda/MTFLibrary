import numpy as np
import torch

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
    torch.Tensor: TorchBackend,
}

def get_backend(array):
    """
    Returns the backend module for a given array type.
    """
    backend_class = _backends.get(type(array))
    if backend_class is None:
        # Fallback for numpy and torch subclasses
        if isinstance(array, np.ndarray):
            return NumpyBackend()
        if isinstance(array, torch.Tensor):
            return TorchBackend()
        raise TypeError(f"Unsupported array type: {type(array)}")
    return backend_class()
