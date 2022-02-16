#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import uuid
from typing import Optional, Union, Tuple, Sequence, Callable
from typing_extensions import Literal

import numba
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import warnings

from dios import DictOfSeries
from outliers import smirnov_grubbs
from scipy.optimize import curve_fit

from saqc.constants import BAD, UNFLAGGED
from saqc.core import register, Flags
from saqc.core.register import flagging
from saqc.lib.tools import customRoller, findIndex, getFreqDelta, toSequence
from saqc.funcs.scores import assignKNNScore
from saqc.funcs.tools import copyField, dropField
from saqc.funcs.transformation import transform
import saqc.lib.ts_operators as ts_ops


def _generateSamples(X: str, Y: str, sub_len: int, data: pd.DataFrame,
                     target: Union[list, int],
                     mode: Literal['flat', 'ts']='flat'):

    x_cols = len(X)
    y_cols = len(Y)

    x_data = data[X].values
    y_data = data[Y].values

    x_split = np.lib.stride_tricks.sliding_window_view(x_data, (sub_len, x_cols))
    x_samples = x_split.reshape(x_split.shape[0], x_split.shape[2], x_split.shape[3])
    # flatten mode (results in [row0, row1, row2, ..., rowSubLen]
    x_samples = x_samples.reshape(x_samples.shape[0], x_samples.shape[1]*x_samples.shape[2])

    y_split = np.lib.stride_tricks.sliding_window_view(y_data, (sub_len, y_cols))
    y_samples = y_split.reshape(x_split.shape[0], y_split.shape[2], y_split.shape[3])

    target = toSequence(target)
    y_mask = [y for y in Y if y in X]
    y_mask = [X.index(y) for y in y_mask]

    selector = list(range(x_samples.shape[1]))
    # indices to drop from selector = allCombinations(t in target,y in y_mask, t*y)
    drop = [x*y for x in target for y in y_mask]
