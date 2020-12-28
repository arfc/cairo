from lorenz import *
import numpy as np
from pytest import approx
from scipy.integrate import odeint


def test_lorenz96_constant():
    """
    For the case of a horizontal distribution,
    so that the slope should be a zero over
    time.
    
    Not finished
    """
    obs_i = 0
    exp_i = lorenz96(x, t, 4, 0)
    assert obs_i == exp_i


def test_lorenz96_constant():
    """
    For the case of a horizontal distribution,
    so that the slope should be a zero over
    time.
    
    Not finished
    """
    obs_i = 0
    exp_i = lorenz96(x, t, 5, 8)
    assert obs_i == exp_i


def test_generate_L96_zero():
    """
    For a time that contains one unit and
    describes a data set of zero, perturbation
    of zero, four variables, and a forcing
    constant of zero
    """
    obs_i = np.array([[0., 0., 0., 0.],
    [0., 0., 0., 0.],[0., 0., 0., 0.],
    [0., 0., 0., 0.]])
    exp_i = generate_L96(np.zeros(4), 0, 4, 0)
    assert obs_i == exp_i


def test_generate_L96_constant():
    """
    For a time that contains multiple units
    and describes a data set of constants
    with a perturbation of 0, five variables
    and a forcing constant of 8
    """
    obs_i = np.array([[8.,8.,8.,8.,8.],
    [8.,8.,8.,8.,8.],[8.,8.,8.,8.,8.],
    [8.,8.,8.,8.,8.],[8.,8.,8.,8.,8.]])
    exp_i = generate_L96(np.ones(5),0,5,8)
    assert obs_i == exp_i


def test_generate_L96_perturbed():
    """
    For a time that contains multiple units
    and describes a data set of constants
    with a perturbation of 1, five variables
    and a forcing constant of 8
    """
    obs_i = np.array([[9.,8.,8.,8.,8.],
    [9.,8.,8.,8.,8.],[9.,8.,8.,8.,8.],
    [9.,8.,8.,8.,8.],[9.,8.,8.,8.,8.]])
    exp_i = generate_L96(np.ones(5),1,5,8)
    assert obs_i == exp_i


def test_lorenz63():
    """
    
    """
    obs_i = 0
    exp_i = 0
    assert obs_i == exp_i


def test_generate_L63():
    """
    
    """
    obs_i = 0
    exp_i = 0
    assert obs_i == exp_i
