**************
Tools Examples
**************


The Tools module contains a variety of
functions that aide in the error
assessment.

:py:func:`tools.MSE`
--------------------

Example:

>>> yhat = np.arange(0,2,1)
>>> y = np.arange(3,4,1)
>>> MSE(yhat,y)
    2.5495097567963922

:py:func:`tools.param_string`
-----------------------------

Example:

>>> params = {
   'n_reservoir': 600,
   'sparsity': 0.1,
   'rand_seed': 85,
   'rho': 0.7,
   'noise': 0.001,
   'future': 20,
   'window': 3,
   'trainlen': 500
}
>>> param_string(params)
    'Reservoir Size:600, Sparsity: 0.1, Spectral Radius: 0.7, Noise: 0.001,
    Training Length: 500, Prediction Window: 3'

:py:func:`optimal_values`
-------------------------

Example:

>>> xset = np.matrix([[1,2],[3,4]])
>>> yset = np.matrix([[1,2],[3,4]])
>>> loss = np.matrix([[1,2],[0,1]])
>>> optimal_values(loss, xset, yset)
    (matrix([[3, 4]]), matrix([[1, 2]]))

:py:func:`esn_prediction`
-------------------------

Example:

>>> data = np.arange(0,200,1)
>>> params = {'n_reservoir': 600,
              'sparsity': 0.1,
              'rand_seed': 85,
              'rho': 0.7,
              'noise': 0.0001,
              'future': 20,
              'window': 4,
              'trainlen': 120}
>>> esn_prediction(data,params)
    array([
       [179.96847224],
       [180.91618768],
       [181.86422127],
       [182.77469821],
       [183.92124944],
       [184.84607557],
       [185.74887841],
       [186.6408091 ],
       [187.92495103],
       [188.81949313],
       [189.71348691],
       [190.60141575],
       [191.9528515 ],
       [192.91274887],
       [193.83421269],
       [194.7837955 ],
       [195.9295301 ],
       [196.820348  ],
       [197.72993036],
       [198.63995168]
    ])
