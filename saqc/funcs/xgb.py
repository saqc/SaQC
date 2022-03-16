#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations
import uuid
from typing import Optional, Union, Tuple, Sequence, Callable

import xgboost
from typing_extensions import Literal

import numba
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import os
import pickle
from dios import DictOfSeries
from saqc.lib.tools import toSequence, getFreqDelta
from xgboost import XGBClassifier, XGBRegressor
from saqc.core import register, Flags

# TODO: k-fold CV
# TODO: meta CV
# TODO: early-stopping (?)
# TODO: best-model-selection
# TODO: geo-frame (?)
# TODO: opt nTrees (nTreeLimit) (?)
# TODO: auto-ML (?)
# TODO: multi-var-prediction (?)
# TODO: target must not contain NaN (!) / i-Filter
# TODO: Train/Validation and Test split


def _getSamplerParams(
    data: DictOfSeries,
    flags: Flags,
    predictors: str,
    target: str,
    window: Union[str, int],
    target_i: Union[int, list, Literal["center", "forward"]],
    predict: Union[Literal["flag", "value"], str],
    mask_target: bool = True,
    **kwargs
):
    x_data = data[predictors].to_df()

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
        data_in = pd.concat([x_data, data[target].to_df()], axis=1)
        data_in = data_in.loc[:, ~data_in.columns.duplicated()]
    elif predict == "flag":
        y_data = pd.concat([flags[t] for t in target], axis=1)
        target = [t + "_flag" for t in toSequence(target)]
        y_data.columns = target
        data_in = pd.concat([x_data, y_data], axis=1)
    else:
        y_data = pd.DataFrame([], index=x_data.index)
        for y in toSequence(target):
            hist_col = [
                ix
                for ix, m in enumerate(flags.history[target].meta)
                if m["kwargs"].get("label", None) == predict
            ]
            flags_col = flags.history[y].hist[hist_col[0]]
            flags_col = flags_col.notna() & (flags_col != -np.inf)
            y_data = pd.concat([y_data, flags_col], axis=1)
        target = [t + predict for t in toSequence(target)]
        y_data.columns = target
        data_in = pd.concat([x_data, y_data], axis=1)

    return window, data_in, x_mask, target, target_i


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
    # currently only support for 1-d y (i guess)
    y_samples = np.squeeze(y_samples, axis=2)
    return x_samples, y_samples


@register(mask=[], demask=[], squeeze=[], multivariate=True, handles_target=True)
def trainXGB(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: str,
    window: Union[str, int],
    target_i: Union[int, list, Literal["center", "forward"]],
    predict: Union[Literal["flag", "value"], str],
    model_dir: str,
    id: Optional[str] = None,
    mask_target: bool = True,
    train_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Dummy Strings.
    """

    id = id or ''
    train_kwargs = train_kwargs or {}

    sampler_config = {'predictors': field,
              'window': window,
              'predict': predict,
              'target_i': target_i,
              'mask_target': mask_target,
              'target': target}

    window, data_in, x_mask, target, target_i = _getSamplerParams(
        data, flags, **sampler_config
    )

    samples = _generateSamples(
        X=field,
        Y=target,
        sub_len=window,
        data=data_in,
        target_i=target_i,
        x_mask=x_mask,
    )

    if predict != "value":
        # TODO: scale_pos_weight
        scale_pos_weight = 1
        train_kwargs.update(
            train_kwargs.pop("objective", {"objective": "binary:logistic"})
        )
        model = XGBClassifier(**train_kwargs)
    else:
        model = XGBRegressor(**train_kwargs)

    fitted = model.fit(samples[0], samples[1])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    var_dir = os.path.join(model_dir, target[0])

    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

    fitted.save_model(os.path.join(var_dir, "model" + id + ".txt"))
    
    config_ser = pd.Series(sampler_config.values(), index=sampler_config.keys(), name=target[0])
    with open(os.path.join(var_dir, "config" + id + ".pkl"), 'wb') as f:
        pickle.dump(sampler_config, f)
    #config_ser.to_csv(os.path.join(var_dir, "config" + id + ".csv"))

    return data, flags


@register(mask=[], demask=[], squeeze=[], multivariate=True)
def predictXGB(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    model_dir: str,
    id: Optional[str] = None,
    model_var: Optional[str] = None,
    **kwargs,
    ):
    """
    Dummy Strings.
    """

    model_var = model_var or field[0]
    id = id or ''

    model_folder = os.path.join(model_dir, model_var)

    with open(os.path.join(model_folder, 'config' + id + '.pkl'), 'rb') as f:
        sampler_config = pickle.load(f)
    if sampler_config['predict'] == 'value':
        model = xgboost.XGBRegressor()
    else:
        model = xgboost.XGBClassifier()

    model.load_model(os.path.join(model_folder, 'model' + id + '.txt'))

    window, data_in, x_mask, target, target_i = _getSamplerParams(
        data, flags, **sampler_config
    )

    samples = _generateSamples(
        X=sampler_config['predictors'],
        Y=target,
        sub_len=window,
        data=data_in,
        target_i=target_i,
        x_mask=x_mask,
    )

    return data, flags