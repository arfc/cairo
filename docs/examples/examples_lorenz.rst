***************
Lorenz Examples
***************

The Lorenz module has several functions who serve to
create numpy arrays of data from arrays of time and
perturbations.


:py:func:`lorenz.lorenz96`
--------------------------

Example:
This example makes use of the function
:py:func:`lorenz.generate_L96`, which
is intended to aid in the creation of
the first argument of `lorenz96`

>>> N = 4
>>> F = 8
>>> P = 0.
>>> t = np.arange(0, 4, 0.1)
>>> x = generate_L96(t, P, N, F)
>>> lorenz96(x, t, N, F)
    """Doesn't Work!"""

.. image:: ../examples/plots/lorenz_plots/lorenz96.png
    :align: center


:py:func:`lorenz.generate_L96`
------------------------------

Example:
This is an example over a 40 unit time-scale
that produces a numpy array of floats

>>> F = 8
>>> P = 0.1
>>> N = 5
>>> t = np.arange(0, 4, 1)
>>> x = generate_L96(t, P, N, F)
    array([[ 8.01   ,  8.    ,  8.   ,  8.   ,  8. ],
       [11.26924566, 12.84391153, -0.72459535, -0.98975901,  3.37443344],
       [ 0.32729446, 11.42177354,  1.87815127,  2.61059131, -0.07544472],
       [-2.63491575,  3.05427465,  2.5680456 ,  3.20375244,  6.83414517]])
