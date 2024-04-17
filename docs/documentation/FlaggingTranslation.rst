.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _FlagsHistoryTranslations:

Flags, History and Translations
===============================


The tutorial aims to introduce into a simple to use, jet powerful method for clearing :ref:`uniformly sampled <cookbooks/DataRegularisation:Data Regularization>`, *univariate*
data, from global und local outliers as well as outlier clusters.
Therefor, we will introduce into the usage of the :py:meth:`~saqc.SaQC.flagUniLOF` method, which represents a
modification of the established `Local Outlier Factor <https://de.wikipedia.org/wiki/Local_Outlier_Factor>`_ (LOF)
algorithm and is applicable without prior modelling of the data to flag.

* :ref:`Example Data Import <cookbooks/OutlierDetection:Example Data Import>`
* :ref:`Initial Flagging <cookbooks/OutlierDetection:Initial Flagging>`
* :ref:`Tuning Threshold Parameter <cookbooks/OutlierDetection:Tuning Threshold Parameter>`
* :ref:`Tuning Locality Parameter <cookbooks/OutlierDetection:Tuning Locality Parameter>`


Example Data Import
-------------------

.. plot::
   :context: reset
   :include-source: False

   import matplotlib
   import saqc
   import pandas as pd
   data = pd.read_csv('../resources/data/hydro_data.csv')
   data = data.set_index('Timestamp')
   data.index = pd.DatetimeIndex(data.index)
   qc = saqc.SaQC(data)

We load the example `data set <https://git.ufz.de/rdm-software/saqc/-/blob/develop/docs/resources/data/hydro_data.csv>`_
from the *saqc* repository using the `pandas <https://pandas.pydata.org/>`_ csv
file reader.
Subsequently, we cast the index of the imported data to `DatetimeIndex <https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html>`, then initialize
a :py:class:`~saqc.SaQC` instance using the imported data and finally we plot
it via the built-in :py:meth:`~saqc.SaQC.plot` method.

.. doctest:: flagUniLOFExample

   >>> import saqc
   >>> data = pd.read_csv('./resources/data/hydro_data.csv')
   >>> data = data.set_index('Timestamp')
   >>> data.index = pd.DatetimeIndex(data.index)
   >>> qc = saqc.SaQC(data)
   >>> qc.plot('sac254_raw') # doctest: +SKIP

.. plot::
   :context:
   :include-source: False
   :class: center

    qc.plot('sac254_raw')

Initial Flagging
----------------

We start by applying the algorithm :py:meth:`~saqc.SaQC.flagUniLOF` with
default arguments, so the main calibration
parameters :py:attr:`n` and :py:attr:`thresh` are set to `20` and `1.5`
respectively.

For an detailed overview over all the parameters, as well as an introduction
into the working of the algorithm, see the documentation of :py:meth:`~saqc.SaQC.flagUniLOF`
itself.

.. doctest:: flagUniLOFExample

   >>> import saqc
   >>> qc = qc.flagUniLOF('sac254_raw')
   >>> qc.plot('sac254_raw') # doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: Flagging result with default parameter configuration.

   qc = qc.flagUniLOF('sac254_raw')
   qc.plot('sac254_raw')

The results from that initial shot seem to look not too bad.
Most instances of obvious outliers seem to have been flagged right
away and there seem to be no instances of inliers having been falsely labeled.
Zooming in onto a 3 months strip on *2016*, gives the impression of
some not so extreme outliers having passed :py:meth:`~saqc.SaQC.flagUniLOF`
undetected:

.. plot::
   :context: close-figs
   :include-source: False
   :class: centers
   :caption: Assuming the flickering values in late september also qualify as outliers, we will see how to tune the algorithm to detect those in the next section.

   qc.plot('sac254_raw', xscope=slice('2016-09','2016-11'))
