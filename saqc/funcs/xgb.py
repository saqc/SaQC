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
from supervised.automl import AutoML

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
from saqc.constants import BAD, UNFLAGGED, FILTER_NONE
from sklearn.model_selection import train_test_split
from sklearn import metrics

# TODO: train-test split
# TODO: geo-frame (?)
# TODO: flag Filter (value prediction vs flag prediction)
# TODO: Imputation Wrap/Fill Wrap
# TODO: Include Predictor isFlagged (np.nan, False True)? (!)
# TODO: dfilter - like in plot
# TODO: chain regression/report
# TODO: sample filter

MULTI_TARGET_MODELS = {
    "chain_reg": sklearn.multioutput.RegressorChain,
    "multi_reg": sklearn.multioutput.MultiOutputRegressor,
    "chain_class": sklearn.multioutput.ClassifierChain,
    "multi_class": sklearn.multioutput.MultiOutputRegressor,
}

AUTO_ML_DEFAULT = {"algorithms": ["Xgboost"],
                   "mode": "Perform"}


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
            raise IndexError("Training with irregularly sampled data not supported")
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
                for ix, m in enumerate(flags.history[y].meta)
                if m["kwargs"].get("label", None) == predict
            ]
            flags_col = flags.history[y].hist[hist_col[0]]
            flags_col = flags_col.notna() & (flags_col != -np.inf)
            y_data = pd.concat([y_data, flags_col], axis=1)
        target = [t + "_" + predict for t in toSequence(target)]
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
    na_filter_y: bool = True,
):

    X = toSequence(X)
    Y = toSequence(Y)
    x_mask = toSequence(x_mask)

    x_cols = len(X)
    y_cols = len(Y)

    x_data = data[X].values
    y_data = data[Y].values

    x_map = np.empty((sub_len, len(X)), dtype='object')
    for var in enumerate(X):
        x_map[:, var[0]] = [var[1] + f'_{k}' for k in range(sub_len)]

    y_map = np.empty((sub_len, len(Y)), dtype='object')
    for var in enumerate(Y):
        y_map[:, var[0]] = [var[1] + f'_{k}' for k in range(sub_len)]

    x_split = np.lib.stride_tricks.sliding_window_view(x_data, (sub_len, x_cols))
    x_samples = x_split.reshape(x_split.shape[0], x_split.shape[2], x_split.shape[3])
    # flatten mode (results in [row0, row1, row2, ..., rowSubLen]
    x_samples = x_samples.reshape(
        x_samples.shape[0], x_samples.shape[1] * x_samples.shape[2]
    )

    x_map_split = np.lib.stride_tricks.sliding_window_view(x_map, (sub_len, x_cols))
    x_map_samples = x_map_split.reshape(x_map_split.shape[0], x_map_split.shape[2], x_map_split.shape[3])
    x_map_samples = x_map_samples.reshape(
        x_map_samples.shape[0], x_map_samples.shape[1] * x_map_samples.shape[2]
    )

    y_split = np.lib.stride_tricks.sliding_window_view(y_data, (sub_len, y_cols))
    y_samples = y_split.reshape(y_split.shape[0], y_split.shape[2], y_split.shape[3])

    y_map_split = np.lib.stride_tricks.sliding_window_view(y_map, (sub_len, y_cols))
    y_map_samples = y_map_split.reshape(y_map_split.shape[0], y_map_split.shape[2], y_map_split.shape[3])


    i_map_split = np.lib.stride_tricks.sliding_window_view(
        np.arange(len(y_data)), sub_len
    )
    i_map_samples = i_map_split.reshape(i_map_split.shape[0], i_map_split.shape[1])

    y_mask = [y for y in x_mask if y in X]
    y_mask = [X.index(y) for y in y_mask]

    selector = list(range(x_samples.shape[1]))
    # indices to drop from selector = allCombinations(t in target_i,y in y_mask, t*y)
    drop = [y * x_cols + x for x in target_i for y in y_mask]
    selector = [s for s in selector if s not in drop]

    x_samples = x_samples[:, selector]
    x_map_samples = x_map_samples[:, selector]
    y_samples = y_samples[:, target_i, :]
    y_map_samples = y_map_samples[:, target_i, :]
    i_map_samples = i_map_samples[:, target_i]
    # currently only support for 1-d y (i guess)
    y_samples = np.squeeze(y_samples, axis=2)
    y_map_samples = np.squeeze(y_map_samples, axis=2)

    na_samples = np.full(y_samples.shape[0], False)
    if na_filter_y:
        na_samples = np.any(np.isnan(y_samples), axis=1)

    if na_filter_x:
        na_s = np.any(np.isnan(x_samples), axis=1)
        na_samples |= na_s

    return x_samples[~na_samples], y_samples[~na_samples], i_map_samples[~na_samples], x_map_samples, y_map_samples


def _mergePredictions(
    prediction_index,
    target_length: int,
    predictions: np.array,
    prediction_map: np.array,
    pred_agg: Callable,
):
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


def _predictionBody(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    model_dir: str,
    pred_agg: callable = np.nanmean,
    id: Optional[str] = None,
    model_var: Optional[str] = None,
    filter_predictors: Optional[bool] = None,
):
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
        na_filter_y=False,
    )

    y_pred = model.predict(samples[0])
    if len(target_i) > 1:
        pred_ser = _mergePredictions(
            data_in.index, len(target_i), y_pred, samples[2], pred_agg
        )
    else:
        pred_ser = pd.Series(np.nan, index=data_in.index)
        pred_ser.iloc[samples[2][:, 0]] = y_pred

    return pred_ser

def _tt_split(d_index, samples, tt_split):
    s_i = samples[2][:, 0]
    index_en = pd.Series(range(len(d_index)), d_index)
    if isinstance(tt_split, str):
        split_point = index_en[:tt_split].values[-1]
        split_i = np.searchsorted(s_i, split_point)
        x_train, x_test = samples[0][:split_i, :], samples[0][split_i:, :]
        y_train, y_test = samples[1][:split_i, :], samples[1][split_i:, :]
    elif isinstance(tt_split, float):
        x_train, x_test, y_train, y_test = train_test_split(samples[0], samples[1], test_size=tt_split, shuffle=True)
    return x_train, x_test, y_train, y_test



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
    tt_split: Union[slice, int, str] = None,
    id: Optional[str] = None,
    mask_target: Optional[bool] = None,
    filter_predictors: Optional[bool] = None,
    train_kwargs: Optional[dict] = None,
    multi_target_model: Optional[Literal["chain", "multi"]] = None,
    base_estimater: Optional[callable] = None,
    **kwargs,
):
    """
    Dummy Strings.
    * [field target] has to be harmed (or field > target)
    * MultiVarRegressionOnly works with no-Na input
    * auto mode only supports MultiOutputRegression (not classification)
    """

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    var_dir = os.path.join(model_dir, target[0])

    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

    id = id or ""
    train_kwargs = train_kwargs or {}
    multi_target_model = multi_target_model or 'chain'
    if not mask_target:
        mask_target = True if predict == "value" else False

    model_type = 'binary_classifier' if predict != 'value' else 'regressor'

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
        na_filter_y=True,
    )

    x_train, x_test, y_train, y_test = _tt_split(data_in.index, samples, tt_split)

    order = train_kwargs.pop('order', None)
    if not base_estimater:
        for k in AUTO_ML_DEFAULT:
            train_kwargs.setdefault(k, AUTO_ML_DEFAULT[k])
            train_kwargs.update({'results_path': var_dir})
        if len(target_i) > 1:
            train_kwargs.pop('results_path', None)
        model = AutoML(**train_kwargs)
    else:
        model = base_estimater(**train_kwargs)

    if len(target_i) > 1:
        if predict != "value":
            model = MULTI_TARGET_MODELS[multi_target_model + "_class"](model, order=order)
        else:
            model = MULTI_TARGET_MODELS[multi_target_model + "_reg"](model, order=order)

    if model_type == 'regressor':
        fitted = model.fit(x_train, y_train.squeeze())
    else:
        fitted = model.fit(x_train, y_train.squeeze().astype(int))

    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    score_book = {}
    classification_report = {}
    if model_type == 'regressor':
        score_book.update({
         'score': [model.score(x_train, y_train.squeeze()), model.score(x_test, y_test.squeeze())],
         'mse': [metrics.mean_squared_error(y_pred_train, y_train), metrics.mean_squared_error(y_pred_test, y_test)],
         'mae': [metrics.mean_absolute_error(y_pred_train, y_train), metrics.mean_absolute_error(y_pred_test, y_test)],
         'explained_var': [metrics.explained_variance_score(y_train, y_pred_train), metrics.explained_variance_score(y_test, y_pred_test)],
         'r2_score': [metrics.r2_score(y_train, y_pred_train), metrics.r2_score(y_test, y_pred_test)]})
    elif model_type == 'binary_classifier':
        score_book.update({
            'score': [model.score(x_train, y_train.squeeze()), model.score(x_test, y_test.squeeze())]})
        confusion_test = sklearn.metrics.confusion_matrix(y_pred_test,y_test)
        confusion_train = sklearn.metrics.confusion_matrix(y_pred_test, y_test)
        for i in range(confusion_train.shape[0]):
            for j in range(confusion_train.shape[1]):
                score_book.update({f'confusion_{i}_{j}': [confusion_train[i, j], confusion_test[i, j]]})
        classification_report.update(metrics.classification_report(y_train, y_pred_train, output_dict=True))

    with open(os.path.join(var_dir, "config" + id + ".pkl"), "wb") as f:
        pickle.dump(sampler_config, f)

    with open(os.path.join(var_dir, "model" + id + ".pkl"), "wb") as f:
        pickle.dump(fitted, f)

    pd.Series(samples[3].squeeze()).to_csv(os.path.join(var_dir, f"x_feature_map_{id}.csv"))
    pd.Series(samples[4].squeeze()).to_csv(os.path.join(var_dir, f"y_feature_map_{id}.csv"))
    pd.DataFrame(score_book).to_csv(os.path.join(var_dir, f"scores_{id}.csv"))
    pd.DataFrame(classification_report).to_csv(os.path.join(var_dir, f"classification_report_{id}.csv"))

    return data, flags


@register(mask=[], demask=[], squeeze=[])
def xgbRegressor(
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

    pred_ser = _predictionBody(data, toSequence(field), flags, model_dir, pred_agg, id, model_var, filter_predictors)

    data[field] = pred_ser
    return data, flags


@register(mask=[], demask=[], squeeze=[])
def xgbClassifier(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    model_dir: str,
    pred_agg: callable = np.nanmean,
    id: Optional[str] = None,
    model_var: Optional[str] = None,
    filter_predictors: Optional[bool] = None,
    flag: float = BAD,
    **kwargs,
):
    """
    Dummy Strings.
    """

    pred_ser = _predictionBody(data, toSequence(field), flags, model_dir, pred_agg, id, model_var, filter_predictors)
    flags[(pred_ser>0).values, field] = flag
    return data, flags
