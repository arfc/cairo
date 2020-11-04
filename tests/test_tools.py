import pytest
import numpy as np
import numpy.random as rd
from tools import MSE, optimal_values, esn_prediction, param_string

# =========================================================
# Set up code
# =========================================================
N = 1000
t = np.linspace(0, 6 * np.pi, N)
noisy_cos = np.cos(t) + (2 * rd.random(N) - 1) / 10
smooth_cos = np.cos(t)
X_in = np.concatenate([[noisy_cos, smooth_cos]], axis=1)

params = {'n_reservoir': 600,
          'sparsity': 0.1,
          'rand_seed': 85,
          'rho': 0.7,
          'noise': 0.001,
          'future': 20,
          'window': 3,
          'trainlen': 500}
# =========================================================
# =========================================================


def test_MSE_1():
    '''
    Case 1: mse returns float
    '''
    assert(isinstance(MSE(smooth_cos, noisy_cos), np.float64))


def test_optimal_values_1():
    """
    Case 1: optimal_values returns the correct set.
    """

    x = np.array([-1, 0, 0])
    y = np.array([1, 0, 0])
    b = np.outer(x, y)
    min_set = (-1, 1)

    opt_set = optimal_values(b, x, y)
    assert(min_set == opt_set)

    return


def test_esn_prediction_1():
    """
    Case 1: The ESN does not train because of mismatched
    input shapes.
    """

    with pytest.raises(IndexError):
        pred = esn_prediction(X_in, params)

    return


def test_esn_prediction_1():
    """
    Case 2: The window size is not a multiple of the total future.
    """

    with pytest.raises(AssertionError):
        pred = esn_prediction(X_in, params)

    return


def test_param_string():
    """
    Verifies that param_string returns string.
    """
    pstring = param_string(params)
    assert(isinstance(pstring, str))
