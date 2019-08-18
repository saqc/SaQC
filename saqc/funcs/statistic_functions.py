#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def _is_valid(data, max_nan_total, max_nan_consec):
    if max_nan_total is None:
        return True

    nan_mask = data.isna()

    if nan_mask.sum() <= max_nan_total:
        if max_nan_consec is None:
            return True
        elif ((1-(~nan_mask)).groupby((~nan_mask).cumsum()).transform(pd.Series.cumsum)).max() <= max_nan_consec:
            return True
        else:
            return False
    else:
        return False


def std_qc(data, max_nan_total=None, max_nan_consec=None):
    """Pandas built in function for statistical moments have quite poor nan- control, so here comes a wrapper that
    will return the standart deviation for a given series input, if the total number of nans in the series does
    not exceed "max_nan_total" and the number of consecutive nans does not exceed max_nan_consec.

    :param data             Pandas Series. The data series, the standart deviation shall be calculated of.
    :param max_nan_total    Integer. Number of np.nan entries allowed to be contained in the series
    :param max_nan_consec   Integer. Maximal number of consecutive nan entries allowed to occure in data.
    """
    if _is_valid(data, max_nan_total, max_nan_consec):
        return data.std()
    return np.nan


def var_qc(data, max_nan_total=None, max_nan_consec=None):
    """Pandas built in function for statistical moments have quite poor nan- control, so here comes a wrapper that
    will return the variance for a given series input, if the total number of nans in the series does
    not exceed "max_nan_total" and the number of consecutive nans does not exceed max_nan_consec.

    :param data             Pandas Series. The data series, the variance shall be calculated of.
    :param max_nan_total    Integer. Number of np.nan entries allowed to be contained in the series
    :param max_nan_consec   Integer. Maximal number of consecutive nan entries allowed to occure in data.
    """
    if _is_valid(data, max_nan_total, max_nan_consec):
        return data.var()
    return np.nan


def mean_qc(data, max_nan_total=None, max_nan_consec=None):
    """Pandas built in function for statistical moments have quite poor nan- control, so here comes a wrapper that
    will return the mean for a given series input, if the total number of nans in the series does
    not exceed "max_nan_total" and the number of consecutive nans does not exceed max_nan_consec.

    :param data             Pandas Series. The data series, the mean shall be calculated of.
    :param max_nan_total    Integer. Number of np.nan entries allowed to be contained in the series
    :param max_nan_consec   Integer. Maximal number of consecutive nan entries allowed to occure in data.
    """
    if _is_valid(data, max_nan_total, max_nan_consec):
        return data.mean()
    return np.nan