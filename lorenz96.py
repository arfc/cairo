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
    F : integer or float
        The forcing constant

    Returns:
    --------
    dxdt : The function
    """

    dxdt = np.ones(N)

    for i in range(N):
        dxdt[i] = (x[(i+1) % N] - x[i-2]) * x[i-1] - x[i] + F

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
    x0 = F*np.ones(N)
    x0[0] += P

    data = odeint(lorenz96, x0, t)

    return data


if __name__ == "__main__":
    # t = np.linspace(0,30,10000)
    t = np.arange(0,30.0, 0.01)
    x = generate_L96(t)

    print(x)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(x[:, 0], x[:, 1], x[:, 2])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.show()
