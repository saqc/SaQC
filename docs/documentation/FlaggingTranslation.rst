.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _FlagsHistoryTranslations:

Flags, History and Translations
===============================


.. doctest:: FlagsDemo

   >>> import saqc
   >>> data = pd.read_csv('./resources/data/hydro_data.csv')
   >>> data = data.set_index('Timestamp')
   >>> data.index = pd.DatetimeIndex(data.index)
   >>> qc = saqc.SaQC(data[['sac254_raw', 'level_raw']])
   >>> qc = qc.align('sac254_raw', method='time', freq='10min')
   >>> qc = qc.flagRange('sac254_raw', max=25, label='too high')
   >>> qc = qc.flagRange('sac254_raw', min=25, label='too low')

We can now check out the result

.. doctest:: FlagsDemo

   >>> qc.flags  # doctest: +SKIP
                   sac254_raw |                level_raw |
   ========================== | ======================== |
   2016-01-01 00:00:00   -inf | 2016-01-01 00:02:00 -inf |
   2016-01-01 00:10:00  255.0 | 2016-01-01 00:17:00 -inf |
   2016-01-01 00:20:00   -inf | 2016-01-01 00:32:00 -inf |
   2016-01-01 00:30:00   -inf | 2016-01-01 00:47:00 -inf |
   2016-01-01 00:40:00  255.0 | 2016-01-01 01:02:00 -inf |
   ...                    ... | ...                  ... |
   2017-12-31 23:10:00  255.0 | 2017-12-31 22:47:00 -inf |
   2017-12-31 23:20:00   -inf | 2017-12-31 23:02:00 -inf |
   2017-12-31 23:30:00   -inf | 2017-12-31 23:17:00 -inf |
   2017-12-31 23:40:00  255.0 | 2017-12-31 23:32:00 -inf |
   2017-12-31 23:50:00   -inf | 2017-12-31 23:47:00 -inf |

Merging all the results to Dataframe

.. doctest:: FlagsDemo

   >>> qc.flags.to_pandas()
                        sac254_raw  level_raw
   2016-01-01 00:00:00        -inf        NaN
   2016-01-01 00:02:00         NaN       -inf
   2016-01-01 00:10:00       255.0        NaN
   2016-01-01 00:17:00         NaN       -inf
   2016-01-01 00:20:00        -inf        NaN
   ...                         ...        ...
   2017-12-31 23:30:00        -inf        NaN
   2017-12-31 23:32:00         NaN       -inf
   2017-12-31 23:40:00       255.0        NaN
   2017-12-31 23:47:00         NaN       -inf
   2017-12-31 23:50:00        -inf        NaN
   <BLANKLINE>
   [175427 rows x 2 columns]

Getting effective flags of specific Variable:

.. doctest:: FlagsDemo

   >>> qc.flags['sac254_raw']
   2016-01-01 00:00:00     -inf
   2016-01-01 00:10:00    255.0
   2016-01-01 00:20:00     -inf
   2016-01-01 00:30:00     -inf
   2016-01-01 00:40:00    255.0
                          ...
   2017-12-31 23:10:00    255.0
   2017-12-31 23:20:00     -inf
   2017-12-31 23:30:00     -inf
   2017-12-31 23:40:00    255.0
   2017-12-31 23:50:00     -inf
   Freq: 10min, Length: 105264, dtype: float64

The History Accessor:

.. doctest:: FlagsDemo

   >>> qc._history['sac254_raw']  #doctest: +SKIP

Access a variables history as a Dataframe:

.. doctest:: FlagsDemo

   >>> qc._history['sac254_raw'].hist
                         0      1      2
   2016-01-01 00:00:00 NaN    NaN    NaN
   2016-01-01 00:10:00 NaN    NaN  255.0
   2016-01-01 00:20:00 NaN    NaN    NaN
   2016-01-01 00:30:00 NaN    NaN    NaN
   2016-01-01 00:40:00 NaN    NaN  255.0
   ...                  ..    ...    ...
   2017-12-31 23:10:00 NaN  255.0    NaN
   2017-12-31 23:20:00 NaN    NaN    NaN
   2017-12-31 23:30:00 NaN    NaN    NaN
   2017-12-31 23:40:00 NaN  255.0    NaN
   2017-12-31 23:50:00 NaN    NaN    NaN
   <BLANKLINE>
   [105264 rows x 3 columns]


Accessing the flags origin annotations:

.. doctest:: FlagsDemo

   >>> qc._history['sac254_raw'].meta[1]
   {'func': 'flagRange', 'args': (), 'kwargs': {'max': 25, 'label': 'too high', 'dfilter': -inf, 'field': 'sac254_raw'}}

work with flags Translation schemes:

.. doctest:: SchemeDemo

   >>> import saqc
   >>> data = pd.read_csv('./resources/data/hydro_data.csv')
   >>> data = data.set_index('Timestamp')
   >>> data.index = pd.DatetimeIndex(data.index)
   >>> qc = saqc.SaQC(data[['sac254_raw', 'level_raw']], scheme=saqc.SimpleScheme())
   >>> qc = qc.align('sac254_raw', method='time', freq='10min')
   >>> qc = qc.flagRange('sac254_raw', max=25, label='too high')
   >>> qc = qc.flagRange('sac254_raw', min=25, label='too low')

Now flags look different:

.. doctest:: SchemeDemo

   >>> qc.flags # doctest: +SKIP
                       sac254_raw |                      level_raw |
   ============================== | ============================== |
   2016-01-01 00:00:00  UNFLAGGED | 2016-01-01 00:02:00  UNFLAGGED |
   2016-01-01 00:10:00         OK | 2016-01-01 00:17:00  UNFLAGGED |
   2016-01-01 00:20:00  UNFLAGGED | 2016-01-01 00:32:00  UNFLAGGED |
   2016-01-01 00:30:00  UNFLAGGED | 2016-01-01 00:47:00  UNFLAGGED |
   2016-01-01 00:40:00         OK | 2016-01-01 01:02:00  UNFLAGGED |
   ...                        ... | ...                        ... |
   2017-12-31 23:10:00        BAD | 2017-12-31 22:47:00  UNFLAGGED |
   2017-12-31 23:20:00  UNFLAGGED | 2017-12-31 23:02:00  UNFLAGGED |
   2017-12-31 23:30:00  UNFLAGGED | 2017-12-31 23:17:00  UNFLAGGED |
   2017-12-31 23:40:00        BAD | 2017-12-31 23:32:00  UNFLAGGED |
   2017-12-31 23:50:00  UNFLAGGED | 2017-12-31 23:47:00  UNFLAGGED |

Getting columns of effective flags works the same:

.. doctest:: SchemeDemo

   >>> qc.flags['sac254_raw'] # doctest: +NORMALIZE_WHITESPACE
   2016-01-01 00:00:00    UNFLAGGED
   2016-01-01 00:10:00          BAD
   2016-01-01 00:20:00    UNFLAGGED
   2016-01-01 00:30:00    UNFLAGGED
   2016-01-01 00:40:00          BAD
                                ...
   2017-12-31 23:10:00          BAD
   2017-12-31 23:20:00    UNFLAGGED
   2017-12-31 23:30:00    UNFLAGGED
   2017-12-31 23:40:00          BAD
   2017-12-31 23:50:00    UNFLAGGED
   Freq: 10min, Length: 105264, dtype: object



History unfortunately shows only internal values:

.. doctest:: SchemeDemo

   >>> qc._history['sac254_raw'].hist # doctest: +NORMALIZE_WHITESPACE
                        0      1      2
   2016-01-01 00:00:00 NaN    NaN    NaN
   2016-01-01 00:10:00 NaN    NaN  255.0
   2016-01-01 00:20:00 NaN    NaN    NaN
   2016-01-01 00:30:00 NaN    NaN    NaN
   2016-01-01 00:40:00 NaN    NaN  255.0
   ...                  ..    ...    ...
   2017-12-31 23:10:00 NaN  255.0    NaN
   2017-12-31 23:20:00 NaN    NaN    NaN
   2017-12-31 23:30:00 NaN    NaN    NaN
   2017-12-31 23:40:00 NaN  255.0    NaN
   2017-12-31 23:50:00 NaN    NaN    NaN
   <BLANKLINE>
   [105264 rows x 3 columns]


We can use the Schemes Value translation dictionary to get a proper representation:

.. doctest:: SchemeDemo

   >>> qc._history['sac254_raw'].hist.replace(saqc.SimpleScheme._BACKWARD) # doctest: +NORMALIZE_WHITESPACE
                                0          1          2
   2016-01-01 00:00:00  UNFLAGGED  UNFLAGGED  UNFLAGGED
   2016-01-01 00:10:00  UNFLAGGED  UNFLAGGED        BAD
   2016-01-01 00:20:00  UNFLAGGED  UNFLAGGED  UNFLAGGED
   2016-01-01 00:30:00  UNFLAGGED  UNFLAGGED  UNFLAGGED
   2016-01-01 00:40:00  UNFLAGGED  UNFLAGGED        BAD
                           ...        ...        ...
   2017-12-31 23:10:00  UNFLAGGED        BAD  UNFLAGGED
   2017-12-31 23:20:00  UNFLAGGED  UNFLAGGED  UNFLAGGED
   2017-12-31 23:30:00  UNFLAGGED  UNFLAGGED  UNFLAGGED
   2017-12-31 23:40:00  UNFLAGGED        BAD  UNFLAGGED
   2017-12-31 23:50:00  UNFLAGGED  UNFLAGGED  UNFLAGGED
   [105264 rows x 3 columns]

Schemes can be changed by simple assignment:

.. doctest:: SchemeDemo

   >>> qc.scheme = saqc.PositionalScheme()
   >>> qc.flags['sac254_raw'] # doctest: +NORMALIZE_WHITESPACE
   2016-01-01 00:00:00    9000
   2016-01-01 00:10:00    9002
   2016-01-01 00:20:00    9000
   2016-01-01 00:30:00    9000
   2016-01-01 00:40:00    9002
                          ...
   2017-12-31 23:10:00    9020
   2017-12-31 23:20:00    9000
   2017-12-31 23:30:00    9000
   2017-12-31 23:40:00    9020
   2017-12-31 23:50:00    9000
   Freq: 10min, Length: 105264, dtype: int64

The positional scheme is a custom scheme that generates effective flags with digits referring to tests ran over the variable.
A test that didnt flag a value is represented by a `0`, a flag is represented by a `2`. We can checkout the Value Translations dictionary
to learn about a flags values internal *Flag intensity*:

.. doctest:: SchemeDemo

   >>> saqc.PositionalScheme()._BACKWARD # doctest: +NORMALIZE_WHITESPACE
   {nan: 0, -inf: 0, 0: 0, 25.0: 1, 255.0: 2}

So, values not checked by any tests and values not flagged by any tests are both represented by `0`, where
`1` represents flag intensity `25.0` and `2` is associated with the worst possible flag (`255.0`). We can
again look at the translated history:

.. doctest:: SchemeDemo

   >>> qc._history['sac254_raw'].hist.replace(saqc.PositionalScheme()._BACKWARD) # doctest: +NORMALIZE_WHITESPACE
                          0    1    2
   2016-01-01 00:00:00  0.0  0.0  0.0
   2016-01-01 00:10:00  0.0  0.0  2.0
   2016-01-01 00:20:00  0.0  0.0  0.0
   2016-01-01 00:30:00  0.0  0.0  0.0
   2016-01-01 00:40:00  0.0  0.0  2.0
   ...                  ...  ...  ...
   2017-12-31 23:10:00  0.0  2.0  0.0
   2017-12-31 23:20:00  0.0  0.0  0.0
   2017-12-31 23:30:00  0.0  0.0  0.0
   2017-12-31 23:40:00  0.0  2.0  0.0
   2017-12-31 23:50:00  0.0  0.0  0.0
   <BLANKLINE>
   [105264 rows x 3 columns]

`SaQC` provides a simple scheme that readily makes available a flags origin in the effective flags series, so we dont have to investigate the history.
The annotated float Scheme:

.. doctest:: SchemeDemo

   >>> qc.scheme = saqc.core.translation.AnnotatedFloatScheme()
   >>> qc.flags['sac254_raw'] # doctest: +NORMALIZE_WHITESPACE
                         flag       func                                         parameters
   2016-01-01 00:00:00   -inf
   2016-01-01 00:10:00  255.0  flagRange  {'min': 25, 'label': 'too low', 'dfilter': -in...
   2016-01-01 00:20:00   -inf
   2016-01-01 00:30:00   -inf
   2016-01-01 00:40:00  255.0  flagRange  {'min': 25, 'label': 'too low', 'dfilter': -in...
   ...                    ...        ...                                                ...
   2017-12-31 23:10:00  255.0  flagRange  {'max': 25, 'label': 'too high', 'dfilter': -i...
   2017-12-31 23:20:00   -inf
   2017-12-31 23:30:00   -inf
   2017-12-31 23:40:00  255.0  flagRange  {'max': 25, 'label': 'too high', 'dfilter': -i...
   2017-12-31 23:50:00   -inf
   <BLANKLINE>
   [105264 rows x 3 columns]

Every effective flag in this scheme consists of three components (instead of just one).

1. The flags value itself:

.. doctest:: SchemeDemo

   >>> qc.flags['sac254_raw']['flag'] # doctest: +NORMALIZE_WHITESPACE
   2016-01-01 00:00:00     -inf
   2016-01-01 00:10:00    255.0
   2016-01-01 00:20:00     -inf
   2016-01-01 00:30:00     -inf
   2016-01-01 00:40:00    255.0
                          ...
   2017-12-31 23:10:00    255.0
   2017-12-31 23:20:00     -inf
   2017-12-31 23:30:00     -inf
   2017-12-31 23:40:00    255.0
   2017-12-31 23:50:00     -inf
   Freq: 10min, Name: flag, Length: 105264, dtype: float64

2. The function every flag originated from:

.. doctest:: SchemeDemo

   >>> qc.flags['sac254_raw']['func'] # doctest: +NORMALIZE_WHITESPACE
   2016-01-01 00:00:00
   2016-01-01 00:10:00    flagRange
   2016-01-01 00:20:00
   2016-01-01 00:30:00
   2016-01-01 00:40:00    flagRange
                            ...
   2017-12-31 23:10:00    flagRange
   2017-12-31 23:20:00
   2017-12-31 23:30:00
   2017-12-31 23:40:00    flagRange
   2017-12-31 23:50:00
   Freq: 10min, Name: func, Length: 105264, dtype: object

3. And the parameters the flag generating function was called with:

.. doctest:: SchemeDemo

   >>> qc.flags['sac254_raw']['parameters'] # doctest: +NORMALIZE_WHITESPACE
   2016-01-01 00:00:00
   2016-01-01 00:10:00    {'min': 25, 'label': 'too low', 'dfilter': -in...
   2016-01-01 00:20:00
   2016-01-01 00:30:00
   2016-01-01 00:40:00    {'min': 25, 'label': 'too low', 'dfilter': -in...
                                                ...
   2017-12-31 23:10:00    {'max': 25, 'label': 'too high', 'dfilter': -i...
   2017-12-31 23:20:00
   2017-12-31 23:30:00
   2017-12-31 23:40:00    {'max': 25, 'label': 'too high', 'dfilter': -i...
   2017-12-31 23:50:00
   Freq: 10min, Name: parameters, Length: 105264, dtype: object



