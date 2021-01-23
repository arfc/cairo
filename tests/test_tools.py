import pytest
import numpy as np
import numpy.random as rd
from tools import *
from pytest import approx

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
    Case 1: MSE returns float
    '''
    assert(isinstance(MSE(smooth_cos, noisy_cos), np.float64))


def test_MSE_equal():
    '''
    MSE is zero for two equally sized arrays
    '''
    yhat = np.random.randint(100, size=(5))
    y = yhat

    obs_i = MSE(yhat, y)
    exp_i = 0
    assert obs_i == exp_i


def test_MSE_sldiff():
    '''
    MSE is a known float when all but one
    entry are the same.
    '''
    y = np.array([[1,2,3]])
    yhat = np.array([[1,1,3]])

    obs = MSE(yhat,y)
    exp = 0.5773502691896257
    assert obs == approx(exp, 0.01)


def test_MSE_comdiff():
    '''
    The MSE should be one when every entry
    is one unit off of its predicted value.
    '''
    y = np.array([[1,2,3]])
    yhat = np.array([[0,1,2]])

    obs = MSE(yhat,y)
    exp = 1
    assert obs == exp


@pytest.mark.xfail(reason="The two arrays should be the same size")
def test_MSE_diffsize():
    '''
    Different sized arrays should not work for
    MSE.
    '''
    yhat = np.array([[1,2,3]])
    y = np.array([[1,2]])

    MSE(yhat,y)

    return


def test_MAE_equal():
    '''
    MAE is zero for two equally sized arrays
    '''
    yhat = np.random.randint(100, size=(5))
    y = yhat

    obs_i = MAE(yhat, y)
    exp_i = 0
    assert obs_i == exp_i


def test_MAE_sldiff():
    '''
    MAE is 1/3 for two arrays that are the
    same but for one entry which differs by one
    unit
    '''
    yhat = np.array([[1,2,3]])
    y = np.array([[1,1,3]])

    obs_i = MAE(yhat, y)
    exp_i = 1/3
    assert approx(obs_i, 0.01) == exp_i


def test_MAE_comdiff():
    '''
    MAE is a float for two completely different arrays
    where the target vector is the same size as the
    predicted vector but all of the entries are
    three units apart from the corresponding one.
    '''
    yhat = np.array([[1,2,3]])
    y = np.array([[4,5,6]])

    obs_i = MAE(yhat, y)
    exp_i = 3.0
    assert obs_i == exp_i


@pytest.mark.xfail(reason="The two arrays should be the same size")
def test_MAE_diffsize():
    '''
    Different sized arrays should not work for
    MAE.
    '''
    yhat = np.array([[1,2,3]])
    y = np.array([[1,2]])

    MAE(yhat,y)

    return


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
