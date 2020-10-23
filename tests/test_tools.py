import pytest
import numpy as np
from tools import MSE, optimal_values


x = np.linspace(0, 2 * np.pi, 1000)
target = np.cos(x)
approximate = 1 - x**2 / 2


def test_MSE_1():
    '''
    Case 1: mse returns float
    '''
    assert(isinstance(MSE(approximate, target), np.float64))


def test_optimal_values_1():
    """
    Case 1: optimal_values returns the correct set.
    """

    x = np.array([-1,0,0])
    y = np.array([1,0,0])
    b = np.outer(x,y)
    min_set = (0,0)

    opt_set = optimal_values(b, x, y)
    assert(min_set==opt_set)
