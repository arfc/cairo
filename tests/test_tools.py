import pytest
from lorenz import generate_L96
import numpy as np
import numpy.random as rd
from tools import *
from pytest import approx
import os

# =========================================================
# Set up code
# =========================================================
N = 1000
t = np.linspace(0, 6 * np.pi, N)
noisy_cos = np.cos(t) + (2 * rd.random(N) - 1) / 10
smooth_cos = np.cos(t)
X_in = np.concatenate([[noisy_cos, smooth_cos]], axis=1)
params_broke = {
    'n_reservoir': 600,
    'sparsity': 0.1,
    'rand_seed': 85,
    'rho': 0.7,
    'noise': 0.001,
    'future': 20,
    'window': 3,
    'trainlen': 500
}

q = np.arange(0, 30.0, 0.01)
x = generate_L96(q)
params_work = {
    'n_reservoir': 600,
    'sparsity': 0.03,
    'rand_seed': 85,
    'rho': 1.5,
    'noise': 0.01,
    'future': 72,
    'window': 72,
    'trainlen': 1200
}
# =========================================================
# =========================================================


def test_MSE_float():
    """
    MSE returns float
    """
    assert(isinstance(MSE(smooth_cos, noisy_cos), np.float64))

    return


def test_MSE_equal():
    """
    MSE is zero for two equally sized arrays
    """
    yhat = np.random.randint(100, size=(5))
    y = yhat

    obs_i = MSE(yhat, y)
    exp_i = 0
    assert obs_i == exp_i

    return


def test_MSE_sldiff():
    """
    MSE is a known float when all but one
    entry are the same.
    """
    y = np.array([[1, 2, 3]])
    yhat = np.array([[1, 1, 3]])

    obs = MSE(yhat, y)
    exp = 0.5773502691896257
    assert obs == approx(exp, 0.01)

    return


def test_MSE_comdiff():
    """
    The MSE should be one when every entry
    is one unit off of its predicted value.
    """
    y = np.array([[1, 2, 3]])
    yhat = np.array([[0, 1, 2]])

    obs = MSE(yhat, y)
    exp = 1
    assert obs == exp

    return


def test_MSE_diffsize():
    """
    Different sized arrays should not work for
    MSE.
    """
    yhat = np.array([[1, 2, 3]])
    y = np.array([[1, 2]])
    with pytest.raises(ValueError):
        MSE(yhat, y)

    return


def test_MAE_float():
    """
    MAE returns float
    """
    assert(isinstance(MAE(smooth_cos, noisy_cos),  np.float64))

    return


def test_MAE_equal():
    """
    MAE is zero for two equally sized arrays.
    """
    yhat = np.random.randint(100, size=(5))
    y = yhat

    obs_i = MAE(yhat, y)
    exp_i = 0
    assert obs_i == exp_i

    return


def test_MAE_sldiff():
    """
    MAE is 1/3 for two arrays that are the
    same but for one entry which differs by one
    unit.
    """
    yhat = np.array([[1, 2, 3]])
    y = np.array([[1, 1, 3]])

    obs_i = MAE(yhat, y)
    exp_i = 1/3
    assert approx(obs_i, 0.01) == exp_i

    return


def test_MAE_comdiff():
    """
    MAE is a float for two completely different arrays
    where the target vector is the same size as the
    predicted vector but all of the entries are
    three units apart from the corresponding one.
    """
    yhat = np.array([[1, 2, 3]])
    y = np.array([[4, 5, 6]])

    obs_i = MAE(yhat, y)
    exp_i = 3.0
    assert obs_i == exp_i

    return


def test_MAE_diffsize():
    """
    Different sized arrays should not work for
    MAE.
    """
    yhat = np.array([[1, 2, 3]])
    y = np.array([[1, 2]])
    with pytest.raises(ValueError):
        MAE(yhat, y)

    return


def test_param_string():
    """
    Verifies that param_string returns string.
    """
    pstring = param_string(params_broke)
    assert(isinstance(pstring, str))

    return


def test_optimal_values_pmone():
    """
    Optimal_values returns the correct set
    when the output should clearly be plus
    or minus one.
    """
    x = np.array([-1, 0, 0])
    y = np.array([1, 0, 0])
    b = np.outer(3, 3)
    min_set = (-1, 1)

    opt_set = optimal_values(b, x, y)
    assert(min_set == opt_set)

    return


def test_optimal_values_equal_arrays():
    """
    Optimal_values returns the correct
    set for two arrays with the same
    values.
    """
    x = np.array([1, 0, 0])
    y = np.array([1, 0, 0])
    b = np.array([
        [0.80, 0.28, 0.46],
        [0.12, 0.49, 0.93],
        [0.38, 0.50, 0.66]
    ])
    min_set = (0, 1)

    opt_set = optimal_values(b, x, y)
    assert (min_set == opt_set)

    return


def test_esn_prediction_diffsize():
    """
    The ESN does not train because of
    mismatched input shapes.
    """
    with pytest.raises(IndexError):
        pred = esn_prediction(X_in, params_work)

    return


def test_esn_prediction_multiple():
    """
    The window size is not a multiple of the
    total future.
    """
    with pytest.raises(AssertionError):
        pred = esn_prediction(X_in, params_broke)

    return


def test_esn_save():
    """
    The esn_prediction function has the
    ability to save predictions to the
    data folder. This test generates a
    sample data set based off of the
    params_save values, with the domain
    of q values and x function defined by
    generate_L96 from the Lorenz module. It
    makes a test file and then removes it.
    As such, there should not be a test data
    file after the test has been completed.
    """
    esn_prediction(x, params_work, 'test_save')
    assert os.path.exists('./data/test_save_prediction.npy')
    if os.path.exists('./data/test_save_prediction.npy'):
        os.remove('./data/test_save_prediction.npy')
    else:
        pass

    return


def test_esn_scenario_output_size():
    """
    The output size of esn_scenario
    should contain an array of the same
    size as the data parameter that
    is input.
    """
    zeros = np.zeros([3, 3])
    params_work['future'] = 1
    params_work['window'] = 1
    output = esn_scenario(zeros, params_work)
    outlen = len(output[0][0])
    exp = 3
    assert outlen == exp

    return


def test_esn_scenario_output_type():
    """
    The output of esn_scenario
    should contain a numpy array.
    """
    output = esn_scenario(x, params_work)
    assert type(output[0][0]) is np.ndarray

    return
