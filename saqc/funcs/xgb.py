#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations
import uuid
from typing import Optional, Union, Tuple, Sequence, Callable

import sklearn.multioutput
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
from sklearn.multioutput import RegressorChain

# TODO: k-fold CV
# TODO: meta CV
# TODO: early-stopping (?)
# TODO: best-model-selection
# TODO: geo-frame (?)
# TODO: opt nTrees (nTreeLimit) (?)
# TODO: auto-ML (?)
# TODO: flag Filter (value prediction vs flag prediction)
# TODO: Train/Validation and Test split
# TODO: Imputation Wrap/Fill Wrap
# TODO: Include Predictor isFlagged (np.nan, False True)? (!)
# TODO: Chain Regression/Classification Order

MULTI_TARGET_MODELS = {'chain_reg': sklearn.multioutput.RegressorChain,
                       'multi_reg': sklearn.multioutput.MultiOutputRegressor,
                       'chain_class': sklearn.multioutput.ClassifierChain,
                       'multi_class': sklearn.multioutput.MultiOutputRegressor}

def _getSamplerParams(
    data: DictOfSeries,
    flags: Flags,
    predictors: str,
    target: str,
    window: Union[str, int],
    target_i: Union[int, list, Literal["center", "forward"]],
    predict: Union[Literal["flag", "value"], str],
    mask_target: bool = True,
    filter_predictors: Optional[bool] = None,
    **kwargs,
):
    x_data = data[predictors].to_df()

    if isinstance(window, str):
        freq = getFreqDelta(x_data.index)
        if not freq:
            raise IndexError("XGB training with irregularly sampled data not supported")
        window = int(pd.Timedelta(window) / freq)

    if target_i in ["center", "forward"]:
        target_i = window // 2 if target_i == "center" else window - 1

    target_i = toSequence(target_i)
    target_i.sort()
    x_mask = []
    if mask_target:
        x_mask = target

    if predict == "value":
        data_in = pd.concat([x_data, data[target].to_df()], axis=1)
        data_in = data_in.loc[:, ~data_in.columns.duplicated()]
    elif predict == "flag":
        y_data = pd.concat([flags[t] > -np.inf for t in target], axis=1)
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

    if len(target_i) > 1:
        na_filter_x = True
    else:
        na_filter_x = filter_predictors or False

    return window, data_in, x_mask, target, target_i, na_filter_x


def _generateSamples(
    X: str,
    Y: str,
    sub_len: int,
    data: pd.DataFrame,
    target_i: Union[list, int],
    x_mask: str = [],
    na_filter_x: bool = True,
    na_filter_y: bool = True
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
    y_samples = y_split.reshape(y_split.shape[0], y_split.shape[2], y_split.shape[3])

    map_split = np.lib.stride_tricks.sliding_window_view(
        np.arange(len(y_data)), sub_len
    )
    map_samples = map_split.reshape(map_split.shape[0], map_split.shape[1])

    y_mask = [y for y in x_mask if y in X]
    y_mask = [X.index(y) for y in y_mask]

    selector = list(range(x_samples.shape[1]))
    # indices to drop from selector = allCombinations(t in target_i,y in y_mask, t*y)
    drop = [y * x_cols + x for x in target_i for y in y_mask]
    selector = [s for s in selector if s not in drop]

    x_samples = x_samples[:, selector]
    y_samples = y_samples[:, target_i, :]
    map_samples = map_samples[:, target_i]
    # currently only support for 1-d y (i guess)
    y_samples = np.squeeze(y_samples, axis=2)

    na_samples = np.full(y_samples.shape[0], False)
    if na_filter_y:
        na_samples = np.any(np.isnan(y_samples), axis=1)

    if na_filter_x:
        na_s = np.any(np.isnan(x_samples), axis=1)
        na_samples |= na_s

    return x_samples[~na_samples], y_samples[~na_samples], map_samples[~na_samples]

def _mergePredictions(prediction_index, target_length, predictions, prediction_map, pred_agg):
    # generate array that holds in any row, the predictions for the associated data index row from all prediction
    # windows and apply prediction aggregation function on that window
    win_arr = np.empty((prediction_index.shape[0], target_length))
    win_arr[:] = np.nan
    win_arr[prediction_map[:, 0], :] = predictions
    for k in range(target_length):
        win_arr[:, k] = np.roll(win_arr[:, k], shift=k)
        win_arr[:k, k] = np.nan
    y_pred = np.apply_along_axis(pred_agg, 1, win_arr)
    pred_ser = pd.Series(y_pred, index=prediction_index)
    return pred_ser


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
    mask_target: Optional[bool] = None,
    filter_predictors: Optional[bool] = None,
    train_kwargs: Optional[dict] = None,
    multi_target_model: Optional[Literal['Chain', 'Multi']] = None,
    **kwargs,
):
    """
    Dummy Strings.
    * [field target] has to be harmed (or field > target)
    * MultiVarRegressionOnly works with no-Na input
    """

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    var_dir = os.path.join(model_dir, target[0])

    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

    id = id or ""
    train_kwargs = train_kwargs or {}
    if not mask_target:
        mask_target = True if predict == 'value' else False

    sampler_config = {
        "predictors": field,
        "window": window,
        "predict": predict,
        "target_i": target_i,
        "mask_target": mask_target,
        "target": target,
    }

    window, data_in, x_mask, target, target_i, na_filter_x = _getSamplerParams(
        data, flags, filter_predictors=filter_predictors, **sampler_config
    )

    samples = _generateSamples(
        X=field,
        Y=target,
        sub_len=window,
        data=data_in,
        target_i=target_i,
        x_mask=x_mask,
        na_filter_x=na_filter_x,
        na_filter_y=True
    )

    if predict != "value":
        # TODO: scale_pos_weight
        # scale_pos_weight = 1
        train_kwargs.update({'use_label_encoder': False})
        model = XGBClassifier(**train_kwargs)
        if len(target_i) > 1:
            model = MULTI_TARGET_MODELS[multi_target_model + '_class'](model)
        fitted = model.fit(samples[0], samples[1].astype(int))
    else:
        model = XGBRegressor(**train_kwargs)
        if len(target_i) > 1:
            model = MULTI_TARGET_MODELS[multi_target_model + '_reg'](model)
        fitted = model.fit(samples[0], samples[1])

    with open(os.path.join(var_dir, "config" + id + ".pkl"), "wb") as f:
        pickle.dump(sampler_config, f)

    with open(os.path.join(var_dir, "model" + id + ".pkl"), "wb") as f:
        pickle.dump(fitted, f)

    return data, flags


@register(mask=[], demask=[], squeeze=[], multivariate=True)
def predictXGB(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    model_dir: str,
    pred_agg: callable = np.nanmean,
    id: Optional[str] = None,
    model_var: Optional[str] = None,
    filter_predictors: Optional[bool] = None,
    **kwargs,
):
    """
    Dummy Strings.
    """

    model_var = model_var or field[0]
    id = id or ""

    model_folder = os.path.join(model_dir, model_var)

    with open(os.path.join(model_folder, "config" + id + ".pkl"), "rb") as f:
        sampler_config = pickle.load(f)

    with open(os.path.join(model_folder, "model" + id + ".pkl"), "rb") as f:
        model = pickle.load(f)

    window, data_in, x_mask, target, target_i, na_filter_x = _getSamplerParams(
        data, flags, filter_predictors=filter_predictors, **sampler_config
    )

    samples = _generateSamples(
        X=sampler_config["predictors"],
        Y=target,
        sub_len=window,
        data=data_in,
        target_i=target_i,
        x_mask=x_mask,
        na_filter_x=na_filter_x,
        na_filter_y=False
    )

    y_pred = model.predict(samples[0])
    if len(target_i) > 1:
        pred_ser = _mergePredictions(data_in.index, len(target_i), y_pred, samples[2], pred_agg)
        # generate array that holds in any row, the predictions for the associated data index row from all prediction
        # windows and apply prediction aggregation function on that window
        win_arr = np.empty((data_in.shape[0], len(target_i)))
        win_arr[:] = np.nan
        win_arr[samples[2][:, 0], :] = y_pred
        for k in range(len(target_i)):
            win_arr[:, k] = np.roll(win_arr[:, k], shift=k)
            win_arr[:k, k] = np.nan
        y_pred = np.apply_along_axis(pred_agg, 1, win_arr)
        pred_ser = pd.Series(y_pred, index=data_in.index)
    else:
        pred_ser = pd.Series(np.nan, index=data_in.index)
        pred_ser.iloc[samples[2][:, 0]] = y_pred

    data[field] = pred_ser.to_frame()
    return data, flags


