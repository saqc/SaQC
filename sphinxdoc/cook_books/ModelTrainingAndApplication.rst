.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. testsetup:: exampleML

   data_path = './resources/temp/tutorialModels'

Model Training And Application
==============================

The tutorial introduces examples of implementation and calibration of basic timeseries based machine learning tasks.

AutoML
------

The :py:meth:`saqc.SaQC.trainModel` method defaultly wraps the `mljar-supervised <https://supervised.mljar.com>`
``AutoML`` class, which means, that hyperparameter tuning, k-fold validation splitting and model selection/ensambling
is performed automatically upon model fitting. This autotuning can be performed at a decent and exhaustive level,
depending on the ``mode``, selected. In this tutorial, for computational efficiency and for mere syntactical demonstration,
we will only fit in `"Explain"` mode - which basically produces unvalidated models with default parameters, that wont generalize well.
Consider switching and playing around with the ``AutoML`` options, by passing a dictionary of options to
the ``train_kwargs`` parameter.
Get an overview over the parameters here: `Here <https://supervised.mljar.com/api/>`.

Regression
----------

Lets generate an :py:class:`saqc.SaQC` instance from some generic toy data.

.. doctest:: exampleML

   >>> import os
   >>> data_path = os.path.abspath('./resources/data/tutorialModels')
   >>> import pandas as pd
   >>> import numpy as np
   >>> import saqc
   >>> rng = np.random.RandomState(1)
   >>> variable1 = rng.random(366)*10
   >>> variable3 = variable1 > 5
   >>> variable1 += np.linspace(0, 20, 366)
   >>> data = pd.DataFrame({'variable1': variable1, 'variable2': variable1*.5, 'variable3': variable3}, index=pd.date_range('2000', freq='1D', periods=366))
   >>> data # doctest:+NORMALIZE_WHITESPACE
               variable1  variable2  variable3
   2000-01-01   4.170220   2.085110      False
   2000-01-02   7.258039   3.629020       True
   2000-01-03   0.110733   0.055366      False
   2000-01-04   3.187709   1.593855      False
   2000-01-05   1.686737   0.843368      False
   ...               ...        ...        ...
   2000-12-27  21.210880  10.605440      False
   2000-12-28  28.848701  14.424350       True
   2000-12-29  25.306005  12.653002       True
   2000-12-30  29.692609  14.846305       True
   2000-12-31  26.366044  13.183022       True
   <BLANKLINE>
   [366 rows x 3 columns]
   >>> qc = saqc.SaQC(data)

First, we will set up a basic uni-variat regression task with a neural network regressor. We set aside some
test data by setting a train-test split point at `"2000-12-01"`. So Training (and validation) is only performed
on the data collected prior to the date ``test_split``. The data collected subsequently will be used for calculating
the report test scores.

.. doctest:: exampleML

   >>> model_path = os.path.join(data_path, 'tutorialModel1VarRegressor_model')
      >>> qc = qc.trainModel('variable1', target='variable1', window='3D', target_idx='center', mode='regressor', path=model_path, ini_kwargs={'mode':'Explain', "algorithms": ["Neural Network"]}, override=True, test_split='2000-12-01')
      AutoML directory: ...
      The task is regression with evaluation metric rmse
      AutoML will use algorithms: ['Neural Network']
      AutoML will ensemble available models
      ...

   Just pass the
   >>> qc = qc.trainModel('variable1', target='variable1', window='3D', target_idx='center', mode='regressor', path=model_path, train_kwargs={'mode':'Explain', "algorithms": ["Neural Network"]}, override=True, test_split='2000-12-01')
   AutoML directory: ...
   The task is regression with evaluation metric rmse
   AutoML will use algorithms: ['Neural Network']
   AutoML will ensemble available models
   ...

Just pass the `modelPredict` method the fitted models location and assign the field, the prediction should be written
to:

.. doctest:: exampleML

   >>> model_path = os.path.join(data_path, 'tutorialModel1VarRegressor_model')
   >>> qc = qc.modelPredict('variable1', target='variable1_1VarPrediction', path=model_path)

Check out the newly generated model folder `'tutorialModel1VarRegressor'` in the ``data_path`` directory. It will contain
the ``AutoML`` fitting report, along with the default fitting report `saqc` generates, containing
scores from common performance measures and some regression plots.

Of course The model performs quite poorly, since most the data variance cant be explained with
`variable1`, since its mostly random. The model performance will improve significantly, if we
add `variable2`, to the predictors set:

.. doctest:: exampleML

   >>> model_path = os.path.join(data_path, 'tutorialModel2VarRegressor_model')
      >>> qc = qc.trainModel(['variable1','variable2'], target='variable1', window='3D', target_idx='center', mode='regressor', path=model_path, ini_kwargs={'mode':'Explain', "algorithms": ["Neural Network"]}, override=True, test_split='2000-12-01')
      AutoML directory: ...
      The task is regression with evaluation metric rmse
      AutoML will use algorithms: ['Neural Network']
      AutoML will ensemble available models
      ...
   >>> qc = qc.trainModel(['variable1','variable2'], target='variable1', window='3D', target_idx='center', mode='regressor', path=model_path, train_kwargs={'mode':'Explain', "algorithms": ["Neural Network"]}, override=True, test_split='2000-12-01')
   AutoML directory: ...
   The task is regression with evaluation metric rmse
   AutoML will use algorithms: ['Neural Network']
   AutoML will ensemble available models
   ...


Classification
--------------

There are two supported methods for the training of Classifier models.
First, we can train a model directly on a timeseries of classes. We can change the target
of training to `variable3`, which is a boolean series:

.. doctest:: exampleML

   >>> model_path = os.path.join(data_path, 'tutorialModel2VarClassifier_model')
      >>> qc = qc.trainModel(['variable1','variable2'], target='variable3', window='3D', target_idx='center', mode='classifier', path=model_path, ini_kwargs={'mode':'Explain', "algorithms": ["Neural Network"]}, override=True, test_split='2000-12-01')
      AutoML directory: ...
      The task is binary_classification with evaluation metric logloss
      AutoML will use algorithms: ['Neural Network']
      AutoML will ensemble available models
      ...

   We can also train a model on the flags of a timeseries:
   >>> qc = qc.trainModel(['variable1','variable2'], target='variable3', window='3D', target_idx='center', mode='classifier', path=model_path, train_kwargs={'mode':'Explain', "algorithms": ["Neural Network"]}, override=True, test_split='2000-12-01')
   AutoML directory: ...
   The task is binary_classification with evaluation metric logloss
   AutoML will use algorithms: ['Neural Network']
   AutoML will ensemble available models
   ...

We can also train a model on the flags of a timeseries:

.. doctest:: exampleML

   >>> flag_model_path = os.path.join(data_path, 'tutorialModel2VarClassifierOnFlags_model')
      >>> qc = qc.flagGeneric('variable3', target='variable2', func=lambda x: x==1)
      >>> qc = qc.trainModel(['variable1','variable2'], target='variable2', window='3D', target_idx='center', mode='flagger', path=flag_model_path, ini_kwargs={'mode':'Explain', "algorithms": ["Neural Network"]}, override=True, test_split='2000-12-01', dfilter=np.inf)
      AutoML directory: ...
      The task is binary_classification with evaluation metric logloss
      AutoML will use algorithms: ['Neural Network']
      AutoML will ensemble available models
      ...
   >>> qc = qc.flagGeneric('variable3', target='variable2', func=lambda x: x==1)
   >>> qc = qc.trainModel(['variable1','variable2'], target='variable2', window='3D', target_idx='center', mode='flagger', path=flag_model_path, train_kwargs={'mode':'Explain', "algorithms": ["Neural Network"]}, override=True, test_split='2000-12-01', dfilter=np.inf)
   AutoML directory: ...
   The task is binary_classification with evaluation metric logloss
   AutoML will use algorithms: ['Neural Network']
   AutoML will ensemble available models
   ...

Flagging with a Classifier
--------------------------

A trained binary classifier can be used to flag variables:

.. doctest:: exampleML

   >>> qc = qc.modelFlag('variable1',  path=flag_model_path)
