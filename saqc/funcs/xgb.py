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
from datetime import datetime

import numba
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import os
import pickle
from dios import DictOfSeries
from saqc.lib.tools import toSequence, getFreqDelta
from saqc.funcs.tools import copyField
from xgboost import XGBClassifier, XGBRegressor
from saqc.core import register, Flags
from saqc.constants import BAD, UNFLAGGED, FILTER_NONE, FILTER_ALL
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.base import BaseEstimator
import shutil
from saqc.funcs.outliers import flagRange
from saqc.funcs.flagtools import clearFlags, transferFlags
from saqc.funcs.tools import dropField
from saqc.funcs.generic import flagGeneric, processGeneric
from saqc.funcs.breaks import flagMissing

# TODO: geo-frame (?)
# TODO: flag Filter (value prediction vs flag prediction)
# TODO: Include Predictor isFlagged (np.nan, False True)? (!)
# TODO: sample filter
# TODO: reassign vals
# TODO: transparent mask-target control

MULTI_TARGET_MODELS = {
    "chain_reg": sklearn.multioutput.RegressorChain,
    "multi_reg": sklearn.multioutput.MultiOutputRegressor,
    "chain_class": sklearn.multioutput.ClassifierChain,
    "multi_class": sklearn.multioutput.MultiOutputClassifier,
}

AUTO_ML_DEFAULT = {"algorithms": ["Xgboost"], "mode": "Perform"}


def _getSamplerParams(
    data: DictOfSeries,
    flags: Flags,
    predictors: str,
    target: str,
    window: Union[str, int],
    target_i: Union[int, list, Literal["center", "forward"]],
    predict: Union[Literal["flag", "value"], str],
    predictors_mask: bool = True,
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

    mask_frame = pd.DataFrame(True, columns=predictors, index=range(window))
    if isinstance(predictors_mask, str):
        if predictors_mask == 'target':
            mask_frame.loc[target_i, target] = False
        else:
            raise ValueError(f'"{predictors_mask}" not a thing.')
    elif isinstance(predictors_mask, dict):
        for key in predictors_mask:
            mask_frame.loc[predictors_mask[key], key] = False
    elif isinstance(predictors_mask, pd.DataFrame):
            mask_frame[predictors_mask.columns] = predictors_mask
    elif isinstance(predictors_mask, np.array):
            mask_frame[:] = predictors_mask

    if predict in ["Regressor", "Classifier"]:
        data_in = pd.concat([x_data, data[target].to_df()], axis=1)
        data_in = data_in.loc[:, ~data_in.columns.duplicated()]
    elif predict == "Flagger":
        y_data = pd.concat([flags[t] > -np.inf for t in target], axis=1)
        target = [t + "_flag" for t in toSequence(target)]
        y_data.columns = target
        data_in = pd.concat([x_data, y_data], axis=1)
    else:
        y_data = pd.DataFrame([], index=x_data.index)
        for y in toSequence(target):
            try:
                hist_col = [
                    ix
                    for ix, m in enumerate(flags.history[y].meta)
                    if m["kwargs"].get("label", None) == predict
                ][0]
            except IndexError:
                raise IndexError(
                    f'Cant find nno data to train on: no flags labeled: "{predict}", for target variable {y}.'
                )
            flags_col = flags.history[y].hist[hist_col]
            flags_col = flags_col.notna() & (flags_col != -np.inf)
            y_data = pd.concat([y_data, flags_col], axis=1)
        target = [t + "_" + predict for t in toSequence(target)]
        y_data.columns = target
        data_in = pd.concat([x_data, y_data], axis=1)

    if len(target_i) > 1:
        na_filter_x = True
    else:
        na_filter_x = filter_predictors or True

    return window, data_in, mask_frame, target, target_i, na_filter_x


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
    x_mask = x_mask.values

    x_cols = len(X)
    y_cols = len(Y)

    x_data = data[X].values
    y_data = data[Y].values

    x_map = np.empty((sub_len, len(X)), dtype="object")
    for var in enumerate(X):
        x_map[:, var[0]] = [var[1] + f"_{k}" for k in range(sub_len)]

    y_map = np.empty((sub_len, len(Y)), dtype="object")
    for var in enumerate(Y):
        y_map[:, var[0]] = [var[1] + f"_{k}" for k in range(sub_len)]

    x_split = np.lib.stride_tricks.sliding_window_view(x_data, (sub_len, x_cols))
    x_samples = x_split.reshape(x_split.shape[0], x_split.shape[2], x_split.shape[3])
    # flatten mode (results in [row0, row1, row2, ..., rowSubLen]
    x_samples = x_samples.reshape(
        x_samples.shape[0], x_samples.shape[1] * x_samples.shape[2]
    )

    x_map_split = np.lib.stride_tricks.sliding_window_view(x_map, (sub_len, x_cols))
    x_map_samples = x_map_split.reshape(
        x_map_split.shape[0], x_map_split.shape[2], x_map_split.shape[3]
    )
    x_map_samples = x_map_samples.reshape(
        x_map_samples.shape[0], x_map_samples.shape[1] * x_map_samples.shape[2]
    )

    y_split = np.lib.stride_tricks.sliding_window_view(y_data, (sub_len, y_cols))
    y_samples = y_split.reshape(y_split.shape[0], y_split.shape[2], y_split.shape[3])

    y_map_split = np.lib.stride_tricks.sliding_window_view(y_map, (sub_len, y_cols))
    y_map_samples = y_map_split.reshape(
        y_map_split.shape[0], y_map_split.shape[2], y_map_split.shape[3]
    )

    i_map_split = np.lib.stride_tricks.sliding_window_view(
        np.arange(len(y_data)), sub_len
    )
    i_map_samples = i_map_split.reshape(i_map_split.shape[0], i_map_split.shape[1])

    x_mask_split = np.lib.stride_tricks.sliding_window_view(x_mask, (sub_len, x_cols))
    x_mask_samples = x_mask_split.reshape(
        x_mask_split.shape[0], x_mask_split.shape[2], x_mask_split.shape[3]
    )
    x_mask_samples = x_mask_samples.reshape(
        x_mask_samples.shape[0], x_mask_samples.shape[1] * x_mask_samples.shape[2]
    )

    selector = x_mask_samples.squeeze(axis=0)
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

    return (
        x_samples[~na_samples],
        y_samples[~na_samples],
        i_map_samples[~na_samples],
        x_map_samples,
        y_map_samples,
    )


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


def _tt_split(d_index, samples, tt_split):
    s_i = samples[2][:, 0]
    index_en = pd.Series(range(len(d_index)), d_index)
    if tt_split is None:
        x_train, y_train, x_test, y_test = (
            samples[0],
            samples[1],
            np.empty((0,) + samples[0].shape[1:]),
            np.empty((0,) + samples[1].shape[1:]),
        )
    if isinstance(tt_split, str):
        split_point = index_en[:tt_split].values[-1]
        split_i = np.searchsorted(s_i, split_point)
        x_train, x_test = samples[0][:split_i, :], samples[0][split_i:, :]
        y_train, y_test = samples[1][:split_i, :], samples[1][split_i:, :]
    elif isinstance(tt_split, float):
        x_train, x_test, y_train, y_test = train_test_split(
            samples[0], samples[1], test_size=tt_split, shuffle=True
        )
    return x_train, x_test, y_train, y_test


@register(mask=[], demask=[], squeeze=[], multivariate=True, handles_target=True)
def trainModel(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: str,
    window: Union[str, int],
    target_i: Union[int, list, Literal["center", "forward"]],
    mode: Union[Literal["Regressor", "Classifier", "Flagger"], str],
    results_path: str,
    model_folder: Optional[str] = None,
    tt_split: Optional[Union[float, str]] = None,
    predictors_mask: Optional[Union[str, np.array, pd.DataFrame, dict]] = None,
    filter_predictors: Optional[bool] = None,
    train_kwargs: Optional[dict] = None,
    multi_target_model: Optional[Literal["chain", "multi"]] = None,
    base_estimator: Optional[BaseEstimator] = None,
    dfilter: float = BAD,
    override: bool = False,
    **kwargs,
):
    """Fits a machine learning model to the target time series or its flags.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldnames of the column, holding the Predictor time series.
    flags : saqc.Flags
        Container to store flags of the data.
    target : str
        The fieldname of the column, holding the Target time series
    window : {int, str}
        Window size of predictor series.
    target_i : {List[int], "center", "Forward"}
        Index of the target values relatively to the window of predictors.
    mode : {"Regressor", "Classifier", "Flagger",  str}
        Type of model to be trained.
        * "Flagger" trains a binary classifier on the flags value of `target`.
        * If another string is passed, a binary classifier gets trained on the flags column labeled `mode`.
    results_path : str
        File path for the training results parent folder.
    model_folder : str, default None
        Folder to write the training results to. If None is passed, the model folder will be named `target`.
        The folder will contain:

        * the pickled model fit: ``model.pkl``
        * the pickled configuration dictionary: ``config.pkl``
        * a csv file listing model fit scores for training and test data: ``scores.csv``
        * mapping of timeseries indices to feature indices: ``x_feature_map.csv``, ``y_feature_map.csv``
        * If trained model is a mlyar.AutoML model, its report path will also point to `model_folder` and the
          report is written to it as well.

        If None is passed, the model folder will be named `target`.
    tt_split: {float, str}, default None
        Rule for splitting data up, into training and testing data.

        * If `None` is passed, no test data will be set aside.
        * If a float is passed, it will be interpreted as the proportion of randomly selected data, that is to be
          set aside for test score calculation (0 <= `tt_split <= 1)`.
        * If a string is passed, it will be interpreted as split date time point: Any data collected before tt_split
          will be the training data set, the rest will be used for testing.

        Test data scores are written to the `score.csv` file in the `model_folder` after model fit.

    mask_target: Optional[bool] = None,
        Wheather or not to include target values in the predictors. This only makes sence, if t

    filter_predictors: Optional[bool] = None,
    train_kwargs: Optional[dict] = None,
    multi_target_model: Optional[Literal["chain", "multi"]] = None,
    base_estimator: Optional[BaseEstimator] = None,
    dfilter: float = BAD,
    override: bool = False,


    * [field target] has to be harmed (or field > target)
    * MultiVarRegressionOnly works with no-Na input
    * auto mode only supports MultiOutputRegression (not classification)


    """

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if model_folder is None:
        model_folder = os.path.join(results_path, target[0])
    else:
        model_folder = os.path.join(results_path, model_folder)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    elif override:
        shutil.rmtree(model_folder)
        os.makedirs(model_folder)

    if not dfilter:
        dfilter = BAD if mode in ["Regressor", "Classifier"] else FILTER_NONE
    train_kwargs = train_kwargs or {}
    multi_target_model = multi_target_model or "chain"
    if predictors_mask is None:
        predictors_mask = 'target' if mode in ["Classifier", "Regressor"] else None

    sampler_config = {
        "predictors": field,
        "window": window,
        "predict": mode,
        "target_i": target_i,
        "predictors_mask": predictors_mask,
        "target": target,
    }

    mode = "Classifier" if mode != "Regressor" else "Regressor"

    window, data_in, predictors_mask, target, target_i, na_filter_x = _getSamplerParams(
        data, flags, filter_predictors=filter_predictors, **sampler_config
    )

    if dfilter < np.inf:
        for f in data_in.columns:
            if f in flags.columns:
                data_in.loc[flags[f] >= dfilter, field] = np.nan

    samples = _generateSamples(
        X=field,
        Y=target,
        sub_len=window,
        data=data_in,
        target_i=target_i,
        x_mask=predictors_mask,
        na_filter_x=na_filter_x,
        na_filter_y=True,
    )

    x_train, x_test, y_train, y_test = _tt_split(data_in.index, samples, tt_split)
    if x_train.shape[0] == 0:
        return data, flags
    if x_test.shape[0] == 0:
        x_test = np.zeros_like(samples[3])
        y_test = np.zeros_like(samples[4])

    if multi_target_model == "chain":
        multi_train_kwargs = {"order": train_kwargs.pop("order", None)}
    else:
        multi_train_kwargs = {}

    if not base_estimator:
        for k in AUTO_ML_DEFAULT:
            train_kwargs.setdefault(k, AUTO_ML_DEFAULT[k])
            train_kwargs.update({"results_path": model_folder})
        if len(target_i) > 1:
            train_kwargs.pop("results_path", None)
        model = AutoML(**train_kwargs)
    else:
        model = base_estimator(**train_kwargs)

    if y_train.max() == y_train.min():
        model = getattr(sklearn.dummy, f'Dummy{mode}')(strategy='constant', constant=y_train[0, 0])

    if len(target_i) > 1:
        if mode == "Regressor":
            model = MULTI_TARGET_MODELS[multi_target_model + "_reg"](
                model, **multi_train_kwargs
            )
        else:
            model = MULTI_TARGET_MODELS[multi_target_model + "_class"](
                model, **multi_train_kwargs
            )

    if mode == "Regressor":
        fitted = model.fit(x_train, y_train.squeeze())
    else:
        fitted = model.fit(x_train, y_train.squeeze().astype(int))

    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    score_book = {}
    classification_report = {}
    if mode == "Regressor":
        score_book.update(
            {
                "score": [
                    model.score(x_train, y_train.squeeze(axis=1)),
                    model.score(x_test, y_test.squeeze(axis=1)),
                ],
                "mse": [
                    metrics.mean_squared_error(y_pred_train, y_train),
                    metrics.mean_squared_error(y_pred_test, y_test),
                ],
                "mae": [
                    metrics.mean_absolute_error(y_pred_train, y_train),
                    metrics.mean_absolute_error(y_pred_test, y_test),
                ],
                "explained_var": [
                    metrics.explained_variance_score(y_train, y_pred_train),
                    metrics.explained_variance_score(y_test, y_pred_test),
                ],
                "r2_score": [
                    metrics.r2_score(y_train, y_pred_train),
                    metrics.r2_score(y_test, y_pred_test),
                ],
            }
        )
    elif mode == "Classifier":
        y_test, y_pred_test, y_train, y_pred_train = (
            y_test.astype(int),
            y_pred_test.astype(int),
            y_train.astype(int),
            y_pred_train.astype(int),
        )
        score_book.update(
            {
                "score": [
                    model.score(x_train, y_train.squeeze(axis=1)),
                    model.score(x_test, y_test.squeeze(axis=1)),
                ]
            }
        )
        confusion_test = sklearn.metrics.confusion_matrix(y_pred_test, y_test)
        confusion_train = sklearn.metrics.confusion_matrix(y_pred_train, y_train)
        c_selector = (
            lambda x, i, j: x[i, j] if (i < x.shape[0] and j < x.shape[1]) else np.nan
        )
        for i in range(confusion_train.shape[0]):
            for j in range(confusion_train.shape[1]):
                score_book.update(
                    {
                        f"confusion_{i}_{j}": [
                            confusion_train[i, j],
                            c_selector(confusion_test, i, j),
                        ]
                    }
                )
        classification_report.update(
            metrics.classification_report(y_train, y_pred_train, output_dict=True)
        )

    with open(os.path.join(model_folder, "config.pkl"), "wb") as f:
        pickle.dump(sampler_config, f)

    with open(os.path.join(model_folder, "model.pkl"), "wb") as f:
        pickle.dump(fitted, f)

    pd.Series().to_csv(os.path.join(model_folder, "saqc_model_dir.csv"))
    pd.Series(samples[3].squeeze()).to_csv(
        os.path.join(model_folder, f"x_feature_map.csv")
    )
    pd.Series(samples[4].squeeze()).to_csv(
        os.path.join(model_folder, f"y_feature_map.csv")
    )
    pd.DataFrame(score_book).to_csv(os.path.join(model_folder, f"scores.csv"))
    pd.DataFrame(classification_report).to_csv(
        os.path.join(model_folder, f"classification_report.csv")
    )

    return data, flags


@register(mask=[], demask=[], squeeze=[])
def modelPredict(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    results_path: str,
    pred_agg: callable = np.nanmean,
    model_folder: Optional[str] = None,
    filter_predictors: Optional[bool] = None,
    dfilter: float = FILTER_NONE,
    **kwargs,
):
    """
    Dummy Strings.
    """

    if model_folder is None:
        model_folder = os.path.join(results_path, field)
    else:
        model_folder = os.path.join(results_path, model_folder)

    with open(os.path.join(model_folder, "config.pkl"), "rb") as f:
        sampler_config = pickle.load(f)

    with open(os.path.join(model_folder, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    window, data_in, x_mask, target, target_i, na_filter_x = _getSamplerParams(
        data, flags, filter_predictors=filter_predictors, **sampler_config
    )

    if dfilter < np.inf:
        for f in sampler_config["predictors"]:
            data_in.loc[flags[f] >= dfilter, field] = np.nan

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

    data[field] = pred_ser
    return data, flags


@register(mask=[], demask=[], squeeze=["field"])
def modelFlag(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    results_path: str,
    pred_agg: callable = np.nanmean,
    model_folder: Optional[str] = None,
    filter_predictors: Optional[bool] = None,
    dfilter: float = BAD,
    **kwargs,
):
    """
    Dummy Strings.
    """

    temp_trg = (
        field
        + str(datetime.now()).replace(" ", "")
        + np.random.random(1)[0].astype(str)
    )
    data, flags = copyField(data, field, flags, target=temp_trg, **kwargs)
    data, flags = modelPredict(
        data,
        temp_trg,
        flags,
        results_path=results_path,
        pred_agg=pred_agg,
        model_folder=model_folder,
        filter_predictors=filter_predictors,
        dfilter=dfilter,
        **kwargs,
    )
    data, flags = clearFlags(data, temp_trg, flags, **kwargs)
    data, flags = flagRange(data, temp_trg, flags, max=0, **kwargs)
    data, flags = transferFlags(data, temp_trg, flags, target=field, **kwargs)
    data, flags = dropField(data, temp_trg, flags, **kwargs)
    return data, flags


@register(mask=[], demask=[], squeeze=[])
def modelImpute(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    results_path: str,
    pred_agg: callable = np.nanmean,
    model_folder: Optional[str] = None,
    filter_predictors: Optional[bool] = None,
    dfilter: float = BAD,
    flag: float = UNFLAGGED,
    **kwargs,
):
    """
    Dummy Strings.
    """

    temp_trg = (
        field
        + str(datetime.now()).replace(" ", "")
        + np.random.random(1)[0].astype(str)
    )
    data, flags = copyField(data, field, flags, target=temp_trg, **kwargs)
    data, flags = modelPredict(
        data,
        temp_trg,
        flags,
        results_path=results_path,
        pred_agg=pred_agg,
        model_folder=model_folder,
        filter_predictors=filter_predictors,
        dfilter=dfilter,
        **kwargs,
    )

    imputation_index = (data[field].isna() | (flags[field] > -np.inf)) & data[
        temp_trg
    ].notna()
    data.loc[imputation_index, field] = data[temp_trg][imputation_index]
    new_vals = data.loc[imputation_index, field].notna()
    new_flags = pd.Series(np.nan, index=flags[field].index)
    new_flags.loc[new_vals.index] = np.nan if flag is None else flag
    flags.history[field].append(
        new_flags, {"func": "modelImpute", "args": (), "kwargs": kwargs}
    )
    return data, flags
