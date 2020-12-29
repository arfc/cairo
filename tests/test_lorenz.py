from lorenz import *
import numpy as np
from pytest import approx
from scipy.integrate import odeint


def test_lorenz96_constant():
    """
    For the case of a horizontal distribution,
    so that the slope should be a zero over
    time.
    """
    obs_i = np.array([8., 8., 8., 8., 8.])
    exp_i = lorenz96(np.array([0, 0, 0, 0, 0]), np.arange(0, 4, 1))
    assert (obs_i == exp_i).all()


def test_lorenz96_linear_distribution():
    """
    For the case of a linear input.
    """
    obs_i = np.array([0., 7., 9., 11., -2.])
    exp_i = lorenz96(np.array([0, 1, 2, 3, 4]), np.arange(0, 4, 0.1))
    assert (obs_i == exp_i).all()


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
    assert (obs_i == exp_i).all()


def test_generate_L96_constant():
    """
    For a time that contains multiple units
    and describes a data set of constants
    with a perturbation of 0, five variables
    and a forcing constant of 8
    """
    obs_i = np.array([[8., 8., 8., 8., 8.],
        [8., 8., 8., 8., 8.], [8., 8., 8., 8., 8.],
        [8., 8., 8., 8., 8.], [8., 8., 8., 8., 8.]])
    exp_i = generate_L96(np.ones(5), 0, 5, 8)
    assert (obs_i == exp_i).all()


def test_generate_L96_perturbed():
    """
    For a time that contains multiple units
    and describes a data set of constants
    with a perturbation of 1, five variables
    and a forcing constant of 8
    """
    obs_i = np.array([[9., 8., 8., 8., 8.],
        [9., 8., 8., 8., 8.], [9., 8., 8., 8., 8.],
        [9., 8., 8., 8., 8.], [9., 8., 8., 8., 8.]])
    exp_i = generate_L96(np.ones(5), 1, 5, 8)
    assert (obs_i == exp_i).all()


def test_lorenz63_constant():
    """
    For the case of a constant x input over
    time.
    """
    obs_i = np.array([0., 0., 0.])
    exp_i = lorenz63(np.array([0, 0, 0, 0, 0]), np.arange(0, 4, 0.1))
    assert (obs_i == exp_i).all()


def test_lorenz63_linear_distribution():
    """
    For the case of a linearly increasing
    x input over time
    """
    obs_i = np.array([10., -1., -16/3])
    exp_i = lorenz63(np.array([0, 1, 2, 3, 4]), np.arange(0, 4, 0.1))
    assert (obs_i == exp_i).all()


def test_generate_L63_rho_leq1():
    """
    Testing a time range with a rho value
    less than one.
    """
    obs_i = np.array([[1., 1., 1.],
       [0.40779223, 0.37990175, 0.16719831],
       [0.22938198, 0.21751446, 0.03659244],
       [0.13924149, 0.13244845, 0.01146649]])
    exp_i = generate_L63(np.arange(0,4,1),0.5)
    assert obs_i == approx(exp_i)


def test_generate_L63_rho_geq1():
    """
    Testing a time range with a rho value
    greater than one.
    """
    obs_i = np.array([[ 1., 1., 1.],
       [-9.37856995, -8.35703373, 29.36232527],
       [-8.17349956, -9.56202269, 24.62070256],
       [-7.45666031, -6.19099807, 27.44180888]])
    exp_i = generate_L63(np.arange(0,4,1))
    assert obs_i == approx(exp_i)


def test_generate_L63_rho_eq1():
    """
    Testing a time range with a rho value
    equal to one.
    """
    obs_i = np.array([[1., 1., 1.],
       [0.63113547, 0.6153045 , 0.23992849],
       [0.53483921, 0.5285069 , 0.12377577],
       [0.4835722 , 0.47931319, 0.09419961]])
    exp_i = generate_L63(np.arange(0,4,1),1)
    assert obs_i == approx(exp_i)


def test_generate_L63_zeros():
    """
    Generate a set with an input of zeros.
    """
    obs_i = np.array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])
    exp_i = generate_L63(np.zeros(4))
    assert (obs_i == exp_i).all()
