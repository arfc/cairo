import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def lorenz96(x, t, N=5, F=8):
    """
    This is the Lorenz 96 model with constant forcing.
    Code snippet adapted from a Wikipedia example.
    Here the differential equation is described for
    use with the ``odeint`` function.

    Parameters:
    -----------
    x : The independent variable
    t : The time of the simulation
    N : integer
        The number of variables in the system
        N >=4
    F : integer or float
        The forcing constant

    Returns:
    --------
    dxdt : The function
    """

    dxdt = np.ones(N)

    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F

    return dxdt


def generate_L96(t, P=0.01, N=5, F=8):
    """
    This function generates data for the Lorenz-96
    model.

    Parameters:
    -----------
    t : numpy array
        The array of time steps.
    P : integer or float
        The initial perturbation on the system.
    N : integer
        The number of variables in the system.
        N >= 4.
    F : integer or float
        The forcing constant.

    Returns:
    --------
    data : The time series data for Lorenz-96.
    """

    # set initial conditions
    x0 = F * np.ones(N)
    x0[0] += P

    data = odeint(lorenz96, x0, t)

    return data


def lorenz63(x, t, rho=28.0, sigma=10.0, beta=(8.0 / 3.0)):
    """
    This is the Lorenz-63 model used in paper by
    Pathak et. al. 2017. DOI: 10.1063/1.5010300

    This function describes the differential equation
    for use with the ``odeint`` function from scipy.
    """

    N = 3

    dxdt = np.ones(N)

    for i in range(N):
        if i == 0:
            dxdt[i] = sigma * (x[i + 1] - x[i])
        elif i == 1:
            dxdt[i] = x[i - 1] * (rho - x[i + 1]) - x[i]
        elif i == 2:
            dxdt[i] = x[i - 2] * x[i - 1] - beta * x[i]

    return dxdt


def generate_L63(t, rho=28.0, sigma=10.0, beta=(8.0 / 3.0)):
    """
    This function generates data for the Lorenz-63
    model.
    """
    N = 3
    x0 = np.ones(N)

    data = odeint(lorenz63, x0, t, args=(rho, sigma, beta))

    return data


if __name__ == "__main__":

#==================================================================
# Lorenz 96 : Three Dimensional Plot
#==================================================================

    # t = np.arange(0, 40.0, 0.01)
    # x = generate_L96(t)
    #
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # ax.plot(x[:, 0], x[:, 1], x[:, 2])
    # ax.set_xlabel("$x_1$")
    # ax.set_ylabel("$x_2$")
    # ax.set_zlabel("$x_3$")
    # plt.show()

#==================================================================
# Lorenz 63 : Three Dimensional Plot
#==================================================================
    # t = np.arange(0, 40.0, 0.01)
    # x = generate_L63(t, rho=1.2, sigma=0.1, beta=0)

    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # ax.plot(x[:, 0], x[:, 1], x[:, 2])
    # ax.set_xlabel("$x_1$")
    # ax.set_ylabel("$x_2$")
    # ax.set_zlabel("$x_3$")
    # plt.show()

#==================================================================
# Lorenz 96 : One Dimensional Plots
#==================================================================
    t = np.arange(0, 40.0, 0.01)
    x = generate_L63(t, rho=1.2, sigma=0.1, beta=0)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t, x[:, 0], label=r'$X_1$')
    ax[0].legend()
    ax[1].plot(t, x[:, 1], label=r'$X_2$')
    ax[1].legend()
    ax[2].plot(t, x[:, 2], label=r'$X_3$')
    ax[2].legend(loc='lower right')

    ax[0].set_xlabel("time")
    ax[1].set_xlabel("time")
    ax[2].set_xlabel("time")

    fig.tight_layout()
    plt.show()
