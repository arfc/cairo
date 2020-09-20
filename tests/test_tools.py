import pytest
import numpy as np
from tools import MSE


x = np.linspace(0,2*np.pi,1000)
target = np.cos(x)
approximate = 1 - x**2/2

def test_MSE_1():
    '''
    Case 1: mse returns float
    '''
    assert(type(MSE(approximate, target)) is np.float64)
