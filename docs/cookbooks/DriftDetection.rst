.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later


Drift Detection
===============



Overview
--------

The guide briefly introduces into the usage of the :py:meth:`~saqc.SaQC.flagDriftFromNorm` method.
The method detects sections in timeseries that deviate from the majority in a group of variables


Example Data Import
-------------------

.. plot::
   :context: reset
   :include-source: False

   import matplotlib
   import saqc
   import pandas as pd
   data = pd.read_csv('../resources/data/tempSensorGroup.csv', index_col=0)
   data.index = pd.DatetimeIndex(data.index)
   qc = saqc.SaQC(data)

We load the example `data set <https://git.ufz.de/rdm-software/saqc/-/blob/develop/docs/resources/data/tempsenorGroup.csv>`_
from the *saqc* repository using the `pandas <https://pandas.pydata.org/>`_ csv
file reader. Subsequently, we cast the index of the imported data to `DatetimeIndex <https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html>`
and use the `plot` method built into the dataframe object to have a look at the imported variables:

.. doctest:: flagDriftFromNorm

   >>> import saqc
   >>> data = pd.read_csv('./resources/data/tempSensorGroup.csv')
   >>> data = data.set_index('Timestamp')
   >>> data.plot() # doctest: +SKIP


.. plot::
   :context: close-figs
   :include-source: False
   :class: center

    data.plot()

Parameters
----------



Although there seems to be a lot of user input to parametrize, most of it is easy to be interpreted and can be selected
defaultly.

frac
^^^^

The percentage of data, needed to define the "normal" group expressed in a number out of :math:`[0,1]`.
This, of course must be something over 50  percent (math:`0.5`), and can be
selected according to the number of drifting variables one expects the data to have at max.

method
^^^^^^

The linkage method can have some impact on the clustering, but sticking to the default value `single` might be
sufficient for most the tasks.

spread
^^^^^^

The main parameter giving control over the algorithms behavior and having to be selected carefully, is the spreading
norm parameter `spread`.
It determines the maximum spread of a normal group by limiting the costs, a cluster agglomeration must not exceed in
every linkage step.

For singleton clusters, that costs just equal half the distance, the timeseries in the clusters have to
each other. So, no timeseries` can be clustered together, that are more then two times the spreading norm distanted
from each other.

When timeseries get clustered together, this new clusters distance to all the other timeseries/clusters is calculated
according to the linkage method specified. By default, it is the minimum distance, the members of the clusters have to
each other.

Having that in mind, it is advisable to choose a distance function as metric, that can be well interpreted in the units
dimension of the measurement, and where the interpretation is invariant over the length of the timeseries`.

Metric
^^^^^^

The *averaged manhatten metric* is set as the metric default, since it more or less represents the
averaged value distance, two timeseries have (as opposed by *euclidean*, for example, which scales non linear with the
compared timeseries length). For the selection of the `spread` parameter the default metric is helpful, since it
allows interpreting the spreading in the dimension of the meassurments.


Method
------

The aim of the algorithm is to sections in timeseries, that significantly deviate from a normal group of parallel timeseries.

"Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed.
In addition, only a group is considered "normal", if it contains more then a certain percentage of the variables that are to be clustered in "normal" ones and "not normal" ones.

The steps of the algorithm are the following:

* Calculate the distances :math:`d(x_i,x_j)` for all timeseries :math:`x_i` that are to be clustered with a metric specified by the user
* Calculate a dendogram with a hierarchical linkage algorithm, specified by the user.
* Flatten the dendogram at the level, the agglomeration costs exceed the value given by a spreading norm, specified by the user
* check if there is a cluster containing more than a certain percentage of variables (percentage specified by the user).
   * if yes: flag all the variables that are not in that cluster
   * if no: flag nothing


Example
-------


