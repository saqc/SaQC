#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations
import uuid
from typing import Optional, Union, Tuple, Sequence, Callable
from typing_extensions import Literal

import numba
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from dios import DictOfSeries
from saqc.lib.tools import toSequence, getFreqDelta
from xgboost import XGBClassifier
from saqc.core import register, Flags


def _generateSamples(
    X: str,
    Y: str,
    sub_len: int,
    data: pd.DataFrame,
    target_i: Union[list, int],
    x_mask: str = [],
):

    X = toSequence(X)
    Y = toSequence(Y)
    x_mask = toSequence(x_mask)

    x_cols = len(X)
    y_cols = len(Y)

    x_data = data[X].values
    y_data = data[Y].values

    x_split = np.lib.stride_tricks.sliding_window_view(x_data, (sub_len, x_cols))
    x_samples = x_split.reshape(x_split.shape[0], x_split.shape[2], x_split.shape[3])
    # flatten mode (results in [row0, row1, row2, ..., rowSubLen]
    x_samples = x_samples.reshape(
        x_samples.shape[0], x_samples.shape[1] * x_samples.shape[2]
    )

    y_split = np.lib.stride_tricks.sliding_window_view(y_data, (sub_len, y_cols))
    y_samples = y_split.reshape(x_split.shape[0], y_split.shape[2], y_split.shape[3])

    target_i = toSequence(target_i)
    y_mask = [y for y in x_mask if y in X]
    y_mask = [X.index(y) for y in y_mask]

    selector = list(range(x_samples.shape[1]))
    # indices to drop from selector = allCombinations(t in target_i,y in y_mask, t*y)
    drop = [y * x_cols + x for x in target_i for y in y_mask]
    selector = [s for s in selector if s not in drop]

    x_samples = x_samples[:, selector]
    y_samples = y_samples[:, target_i, :]

    return x_samples, y_samples


@register(mask=[], demask=[], squeeze=[], handles_target=True)
def trainXGB(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: str,
    window: Union[str, int],
    target_i: Union[int, list, Literal["center", "forward"]],
    predict: Union[Literal["flag", "value"], str],
    mask_target: bool = True,
    train_kwargs: Optional[dict] = None,
    **kwargs
):
    """
    Function
    """
    x_data = data[field]

    if isinstance(window, str):
        freq = getFreqDelta(x_data.index)
        if not freq:
            raise IndexError("XGB training with irregularly sampled data not supported")
        window = int(pd.Timedelta(window) / freq)

    if target_i in ["center", "forward"]:
        target_i = window // 2 if target_i == "center" else window - 1

    if mask_target:
        x_mask = target

    if predict == "value":
        data_in = pd.concat([x_data, data[target]], axis=1)
    elif predict == "flag":
        y_data = pd.concat([flags[t] for t in target], axis=1)
        target = [t + "_flag" for t in toSequence(target)]
        y_data.columns = target
        data_in = pd.concat([x_data, y_data], axis=1)
    else:
        y_data = pd.DataFrame([], index=x_data.index)
        for y in toSequence(target):
            hist_col = [ix for ix,m in enumerate(flags.history[target].meta) if m['kwargs'].get('label', None) == predict ]
            flags_col = flags.history[y].hist[hist_col[0]]
            flags_col = flags_col.notna() & (flags_col != -np.inf)
            y_data = pd.concat([y_data, flags_col], axis=1)
        target = [t + predict for t in toSequence(target)]
        y_data.columns = target
        data_in = pd.concat([x_data, y_data], axis=1)


    samples = _generateSamples(
        X=field, Y=target, sub_len=window, data=data_in, target_i=target_i, x_mask=x_mask
    )

    if predict != 'value':
        train_kwargs.update(train_kwargs.pop('objective', {'objective':"binary:logistic"}))
        model = XGBClassifier(**train_kwargs)

    return data, flags
