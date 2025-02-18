# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:04:49 2025

@author: manik
"""
# taylor_operations.py
_GLOBAL_MAX_ORDER = 10  # Default global max order

def set_global_max_order(order):
    """
    Set the global maximum order for Taylor series truncation.

    Parameters:
        order (int): The maximum order to set globally.
    """
    global _GLOBAL_MAX_ORDER
    if not isinstance(order, int) or order < 0:
        raise ValueError("Global max order must be a non-negative integer.")
    _GLOBAL_MAX_ORDER = order

def get_global_max_order():
    """
    Get the current global maximum order for Taylor series truncation.

    Returns:
        int: The current global maximum order.
    """
    return _GLOBAL_MAX_ORDER