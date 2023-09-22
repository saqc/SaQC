#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as st

import saqc.lib.ts_operators as ts_ops
from saqc import BAD, DOUBTFUL, FILTER_ALL, FILTER_NONE, GOOD, UNFLAGGED


def clip(series, lower=None, upper=None):
    return series.clip(lower=lower, upper=upper)


def zscore(obj):
    return st.zscore(obj, nan_policy="omit")


def cv(series: pd.Series) -> pd.Series:
    """
    calculates the coefficient of variation on a min-max scaled time series
    """
    series_ = (series - series.min()) / (series.max() - series.min())
    return series_.std() / series_.mean()

# operators dict (mapping array-likes to scalars)
ENV_OPERATORS = {
    # value sum. ignores NaN.
    "sum": np.nansum,
    # data containers length (including NaN.)
    "len": len,
    # Mean. Omits NaN values.
    "mean": np.nanmean,
    # Sample maximum.
    "max": np.nanmax,
    # Sample minimum.
    "min": np.nanmin,
    # Sample standard deviation. Omits NaN values.
    "std": np.nanstd,
    # Sample Variance Omits NaN values.
    "var": np.nanvar,
    # Median absolute deviation. Omits NaN values
    "mad": ts_ops.mad,
    # Sample coefficient of variation. Omits NaN values.
    "cv": cv,
    # Sample median. Omits NaN values
    "median": np.nanmedian,
    # Count number of values. Omits NaN values.
    "count": ts_ops.count,
    # evaluate datachunks with regard to total and consecutive number of invalid values
    "isValid": ts_ops.isValid
}

# transformations dict (mapping array likes to array likes of same size)
ENV_TRAFOS = {
    # Returns a series` diff.
    "diff": ts_ops.difference,
    # Scales data to [0,1] interval.
    "scale": ts_ops.normScale,
    # Standardize with standard deviation.
    "zScore": ts_ops.standardizeByMean,
    # Standardize with median and MAD.
    "madScore": ts_ops.standardizeByMedian,
    # Standardize with median and inter quantile range.
    "iqsScore": ts_ops.standardizeByIQR,
    # Identity.
    "id": ts_ops.identity,
    # Absolute value function.
    "abs": np.abs,
    # Exponential value Function.
    "exp": np.exp,
    # Logarithm.
    "log": np.log,
    # Logarithm, returning NaN for zero input, instead of -inf.
    "nanLog": ts_ops.zeroLog,
    # clip
    "clip": clip,
    # evaluate datachunks with regard to total and consecutive number of invalid values
    "evaluate": ts_ops.validationTrafo
}

# Constants Dictionary
ENV_CONSTANTS = {
    "GOOD": GOOD,
    "BAD": BAD,
    "UNFLAGGED": UNFLAGGED,
    "DOUBTFUL": DOUBTFUL,
    "FILTER_ALL": FILTER_ALL,
    "FILTER_NONE": FILTER_NONE,
}
# environment
ENVIRONMENT = {**ENV_TRAFOS, **ENV_OPERATORS, **ENV_CONSTANTS}
