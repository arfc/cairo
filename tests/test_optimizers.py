import pytest
import numpy as np
import numpy.random as rd
from tools import MSE, optimal_values, esn_prediction
from optimizers import grid_optimizer

# =========================================================
# Set up code
# =========================================================
N = 1000
t = np.linspace(0, 6 * np.pi, N)
noisy_cos = np.cos(t) + (2 * rd.random(N) - 1) / 10
smooth_cos = np.cos(t)
X_in = np.concatenate([[noisy_cos, smooth_cos]], axis=1)

reservoir_set = [600, 800, 1000]
sparsity_set = [0.005, 0.01, 0.03]
trainingLengths = np.arange(100, 800, 100)

params = {'n_reservoir': 600,
          'sparsity': 0.1,
          'rand_seed': 85,
          'rho': 0.7,
          'noise': 0.001,
          'future': 20,
          'window': 4,
          'trainlen': 500}
# =========================================================
# =========================================================


def test_grid_optimize_1():
    """
    Case 1: Too many variables given
    """
    with pytest.raises(AssertionError):
        loss = grid_optimizer(X_in.T,
                              params,
                              args=['trainlen', 'rho', 'noise'],
                              xset=trainingLengths,
                              verbose=False,
                              visualize=False)
    return


def test_grid_optimize_2():
    """
    Case 2: Variable not in params
    """
    with pytest.raises(AssertionError):
        loss = grid_optimizer(X_in.T,
                              params,
                              args=['jimmy'],
                              xset=trainingLengths,
                              verbose=False,
                              visualize=False)
    return


def test_grid_optimize_3():
    """
    Case 3: Two variables specified, only one set given.
    """
    with pytest.raises(AssertionError):
        loss = grid_optimizer(X_in.T,
                              params,
                              args=['n_reservoir', 'sparsity'],
                              xset=reservoir_set,
                              verbose=False,
                              visualize=False)
    return


def test_grid_optimize_4():
    """
    Case 4: One variable specified, two sets given.
    """
    with pytest.raises(AssertionError):
        loss = grid_optimizer(X_in.T,
                              params,
                              args=['n_reservoir'],
                              xset=reservoir_set,
                              yset=sparsity_set,
                              verbose=False,
                              visualize=False)
    return


def test_grid_optimize_5():
    """
    Case 5: ESN does not train because of mismatched input shapes.
    """
    with pytest.raises(IndexError):
        loss = grid_optimizer(X_in,
                              params,
                              args=['n_reservoir'],
                              xset=reservoir_set,
                              yset=None,
                              verbose=False,
                              visualize=False)
    return


def test_grid_optimize_6():
    """
    Case 6: No xset is given to test. Should raise TypeError.
    """
    with pytest.raises(TypeError):
        loss = grid_optimizer(X_in,
                              params,
                              args=['n_reservoir'])
    return


def test_grid_optimize_7():
    """
    Case 7: yset is given, but not xset.
    """
    with pytest.raises(TypeError):
        loss = grid_optimizer(X_in,
                              params,
                              args=['n_reservoir'],
                              yset=reservoir_set)
    return
