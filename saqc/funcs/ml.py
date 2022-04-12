#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations
import uuid
from typing import Optional, Union, Tuple, Sequence, Callable

import sklearn.multioutput
from typing_extensions import Literal
from supervised.automl import AutoML
from datetime import datetime


import numpy as np
import pandas as pd
import os
import pickle
from dios import DictOfSeries
from saqc.lib.tools import toSequence, getFreqDelta
from saqc.funcs.tools import copyField
from saqc.core import register, Flags
from saqc.constants import BAD, UNFLAGGED, FILTER_NONE, FILTER_ALL
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.base import BaseEstimator
import shutil
from saqc.funcs.outliers import flagRange
from saqc.funcs.flagtools import clearFlags, transferFlags
from saqc.funcs.tools import dropField


MULTI_TARGET_MODELS = {
    "chain_reg": sklearn.multioutput.RegressorChain,
    "multi_reg": sklearn.multioutput.MultiOutputRegressor,
    "chain_class": sklearn.multioutput.ClassifierChain,
    "multi_class": sklearn.multioutput.MultiOutputClassifier,
}

AUTO_ML_DEFAULT = {"algorithms": ("Xgboost",), "mode": "Perform"}


def _getSamplerParams(
    data: DictOfSeries,
    flags: Flags,
    predictors: str,
    target: str,
    window: Union[str, int],
    target_idx: Union[int, list, Literal["center", "forward"]],
    predict: Union[Literal["flag", "value"], str],
    feature_mask: bool = True,
    drop_na_samples: bool = True,
    **kwargs,
):
    x_data = data[predictors].to_df()

    if isinstance(window, str):
        freq = getFreqDelta(x_data.index)
        if not freq:
            raise IndexError("Training with irregularly sampled data not supported")
        window = int(pd.Timedelta(window) / freq)

    if target_idx in ["center", "forward"]:
        target_idx = window // 2 if target_idx == "center" else window - 1

    target_idx = toSequence(target_idx)
    target_idx.sort()

    mask_frame = pd.DataFrame(True, columns=predictors, index=range(window))
    if isinstance(feature_mask, str):
        if feature_mask == "target":
            if toSequence(target)[0] in mask_frame.columns:
                mask_frame.loc[target_idx, target] = False
        else:
            raise ValueError(f'"{feature_mask}" not a thing.')
    elif isinstance(feature_mask, dict):
        for key in feature_mask:
            mask_frame.loc[feature_mask[key], key] = False
    elif isinstance(feature_mask, pd.DataFrame):
        mask_frame[feature_mask.columns] = feature_mask
    elif isinstance(feature_mask, np.ndarray):
        mask_frame[:] = feature_mask

    if predict in ["regressor", "classifier"]:
        data_in = pd.concat([x_data, data[target].to_df()], axis=1)
        data_in = data_in.loc[:, ~data_in.columns.duplicated()]
    elif predict == "flagger":
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
                    f'Cant find no data to train on: no flags labeled: "{predict}", for target variable {y}.'
                )
            flags_col = flags.history[y].hist[hist_col]
            flags_col = flags_col.notna() & (flags_col != -np.inf)
            y_data = pd.concat([y_data, flags_col], axis=1)
        target = [t + "_" + predict for t in toSequence(target)]
        y_data.columns = target
        data_in = pd.concat([x_data, y_data], axis=1)

    if len(target_idx) > 1:
        na_filter_x = True
    else:
        na_filter_x = drop_na_samples

    return window, data_in, mask_frame, target, target_idx, na_filter_x


def _generateSamples(
    X: str,
    Y: str,
    sub_len: int,
    data: pd.DataFrame,
    target_idx: Union[list, int],
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
    y_samples = y_samples[:, target_idx, :]
    y_map_samples = y_map_samples[:, target_idx, :]
    i_map_samples = i_map_samples[:, target_idx]
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


def _makeScoreReports(y_pred_train, y_pred_test, y_train, y_test, mode):
    score_book = {}
    classification_report = {}
    if mode == "regressor":
        score_book.update(
            {
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
    elif mode == "classifier":
        y_test, y_pred_test, y_train, y_pred_train = (
            y_test.astype(int),
            y_pred_test.astype(int),
            y_train.astype(int),
            y_pred_train.astype(int),
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
        return score_book, classification_report



def _samplesToSplits(data_in, samples, tt_split):
    x_train, x_test, y_train, y_test = _tt_split(data_in.index, samples, tt_split)
    if x_test.shape[0] == 0:
        x_test = np.zeros_like(samples[3])
        y_test = np.zeros_like(samples[4])

    return x_train, x_test, y_train, y_test

def _modelSelector(multi_target_model, base_estimator, target_idx, train_kwargs, y_train, model_folder, mode):
    if multi_target_model == "chain":
        multi_train_kwargs = {"order": train_kwargs.pop("order", None)}
    else:
        multi_train_kwargs = {}

    if not base_estimator:
        for k in AUTO_ML_DEFAULT:
            train_kwargs.setdefault(k, AUTO_ML_DEFAULT[k])
            train_kwargs.update({"results_path": model_folder})
        if len(target_idx) > 1:
            train_kwargs.pop("results_path", None)
        model = AutoML(**train_kwargs)
    else:
        model = base_estimator(**train_kwargs)

    if y_train.max() == y_train.min():
        model = getattr(sklearn.dummy, f"Dummy{mode.capitalize()}")(
            strategy="constant", constant=y_train[0, 0]
        )

    if len(target_idx) > 1:
        if mode == "regressor":
            model = MULTI_TARGET_MODELS[multi_target_model + "_reg"](
                model, **multi_train_kwargs
            )
        else:
            model = MULTI_TARGET_MODELS[multi_target_model + "_class"](
                model, **multi_train_kwargs
            )

    return model


def _modelFitting(x_train, y_train, model, mode):
    if mode == "regressor":
        fitted = model.fit(x_train, y_train.squeeze())
    else:
        fitted = model.fit(x_train, y_train.squeeze().astype(int))
    return fitted


@register(mask=[], demask=[], squeeze=[], multivariate=True, handles_target=True)
def trainModel(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: str,
    window: Union[str, int],
    target_idx: Union[int, list, Literal["center", "forward"]],
    mode: Union[Literal["regressor", "classifier", "flagger"], str],
    results_path: str,
    model_folder: Optional[str] = None,
    tt_split: Optional[Union[float, str]] = None,
    feature_mask: Optional[Union[str, np.array, pd.DataFrame, dict]] = None,
    drop_na_samples: bool = True,
    train_kwargs: Optional[dict] = None,
    multi_target_model: Optional[Literal["chain", "multi"]] = "chain",
    base_estimator: Optional[BaseEstimator] = None,
    dfilter: float = BAD,
    override: bool = False,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Fits a machine learning model to the target time series or its flags.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldnames of the column, holding the Predictor time series.

    flags : saqc.Flags
        Container to store flags of the data.

    target : str
        The fieldname of the column, holding the Target time series.

    window : {int, str}
        Window size of predictor series.

    target_idx : {List[int], "center", "Forward"}
        Index of the target values relatively to the window of predictors.

    mode : {"regressor", "classifier", "flagger",  str}
        Type of model to be trained.

        * "flagger" trains a binary classifier on the flags value of `target`.
        * If another string is passed, a binary classifier gets trained on the flags column labeled `mode`.

    results_path : str
        File path for the training results parent folder.

    model_folder : str, default None
        Folder to write the training results to. If ``None`` is passed, the model folder will be named `target`.
        The folder will contain:

        * the pickled model fit: ``model.pkl``
        * the pickled configuration dictionary: ``config.pkl``
        * a csv file listing model fit scores for training and test data: ``scores.csv``
        * mapping of timeseries indices to feature indices: ``x_feature_map.csv``, ``y_feature_map.csv``
        * If trained model is a `mlyar.AutoML` model, its report path will also point to `model_folder` and the
          report is written to it as well.

        If ``None`` is passed, the model folder will be named `target`.

    tt_split: {float, str}, default None
        Rule for splitting data up, into training and testing data.

        * If ``None`` is passed, no test data will be set aside.
        * If a float is passed, it will be interpreted as the proportion of randomly selected data, that is to be
          set aside for test score calculation (0 <= ``tt_split`` <= 1).
        * If a string is passed, it will be interpreted as split date time point: Any data sampled before ``tt_split``
          will be comprised in the training data set, the rest will be used for testing.

        Test data scores are written to the `score.csv` file in the `model_folder` after model fit.

    feature_mask: {"target", pd.DataFrame, dict, np.ndarray}, default None
        Controlls wich indices from the input variables are to be hidden (=dropped) while training.
        When ``None`` is passed (default), and a ``mode`` is either `"classifier"` or `"regressor"`, the target
        indices of the target variable are dropped, if the target variable is part of the predictors set. If mode is
        `"flagger"`, no features get hidden by the default ``feature_mask``.

        * "target" - Drop the target indices of the target variable
        * `dict`: A dictionary with variable names as keys and integer lists as items, denoting the indices to be
          dropped.
        * `pd.DataFrame`: A boolean Dataframe, with column named as the variables to be masked, and rows according to
          the number of indices in the feature window.

    drop_na_samples: bool, default True
        Drop samples that contain NaN values.
        In case of a multi target model, fitting with NaN containing samples is not supported.

    train_kwargs : dict, default None
        Keywords to be passed on to the base estimators instantiation method.
        If the base estimator is an ``AutoML`` model (default), the train kwargs default to training an `Xgboost` model
        in "perform" mode. (``Algorithms=["Xgboost"]``, ``mode="Perform"``)
        If multiple features get fitted, one can control the wrappers fitting order by passing the ``train_kwargs``
        an "order" keyword. (If the wrapper is a Chain Model (default))

    multi_target_model : {"chain", "multi"}, default "chain"
        Which multi target wrapper to use for fitting multiple features.
        To alter order in case of chain wrapper (default), add an "order" keyword to the ``train_kwargs``.
        The wrappers instantiated are the ``sklearn.multioutput`` models.

    base_estimator : BaseEstimator, default None
        The base estimator to be fitted to the data. If ``None`` (default), the base estimator
        is an ``AutoML`` (mljar-supervised) instance.

    dfilter : float, default BAD
        Filter Field and Target variables.

    override : bool, default False
        Override the ``results_path``/``model_folder`` directory, if it already exists.
        Fitting a ``mljar.AutoML`` model with not-empty target folder will fail, if ``override`` is ``False``.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    flags : saqc.Flags
        The quality flags of data

    Notes
    -----
    * ``field`` and ``target`` data has to be sampled at the same frequency for the training to
      work as expected.
    * Multi Output Models cant be trained on samples containing NaN values. So those will
      get filtered outomatically.
    * Currently Multi Output Classification is not supported for ``AutoML`` models.

    See Also
    --------
    * :py:meth:`saqc.SaQC.modelPredict`
    * :py:meth:`saqc.SaQC.modelImpute`
    * :py:meth:`saqc.SaQC.modelFlag`
    """

    in_freq = getFreqDelta(pd.concat([data[f] for f in toSequence(field)+ toSequence(target)], axis=1).index)
    if in_freq is None:
        raise IndexError('Input data empty, or not sampled at (multiples) of the same frequency')

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

    train_kwargs = train_kwargs or {}

    if feature_mask is None:
        feature_mask = "target" if mode in ["classifier", "regressor"] else None

    sampler_config = {
        "predictors": field,
        "window": window,
        "predict": mode,
        "target_idx": target_idx,
        "feature_mask": feature_mask,
        "target": target,
        "drop_na_samples": drop_na_samples,
    }

    mode = "classifier" if mode != "regressor" else "regressor"

    window, data_in, feature_mask, target, target_idx, na_filter_x = _getSamplerParams(
        data, flags, **sampler_config
    )

    sampler_config["freq"] = getFreqDelta(data_in.index)

    if dfilter < np.inf:
        for f in data_in.columns:
            if f in flags.columns:
                data_in.loc[flags[f] >= dfilter, field] = np.nan

    samples = _generateSamples(
        X=field,
        Y=target,
        sub_len=window,
        data=data_in,
        target_idx=target_idx,
        x_mask=feature_mask,
        na_filter_x=na_filter_x,
        na_filter_y=True,
    )

    x_train, x_test, y_train, y_test = _samplesToSplits(data_in, samples, tt_split)

    if x_train.shape[0] == 0:
        return data, flags

    model = _modelSelector(multi_target_model, base_estimator, target_idx, train_kwargs, y_train, model_folder, mode)

    fitted = _modelFitting(x_train, y_train, model, mode)

    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    score_book, classification_report = _makeScoreReports(y_pred_train, y_pred_test, y_train, y_test, mode)

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
    drop_na_samples: Optional[bool] = None,
    assign_features: Optional[dict] = None,
    dfilter: float = FILTER_NONE,
    **kwargs,
):
    """
    Use a trained model for predictions.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldnames of the variable to be predicted. Will get overriden with the prediction results,
        if ``target`` is not set.

    flags : saqc.Flags
        Container to store flags of the data.

    results_path: str
        Path to the models parent folder.

    pred_agg: callable, default np.nanmean
        Function for aggregation of multiple predictions associated with the same timestep.

    model_folder: str, None
        Folder containing the model data.
        If ``None`` (default), a folder named ``field`` is searched.
        The folder must contain:

        * the pickled model object, ``model.pkl`` (A sklearn-style model object, implementing
          a ``predict`` method
        * the pickled configuration dictionary, ``config.pkl``.

    drop_na_samples: bool, default None
        Calculate predictions for input samples containing invalid (flagged or NaN)
        values. Defaults to the value the prediction model has been trained with.

    assign_features: dict, default None
        By default, input features to the model have to be (named) the same, as the
        model has been trained with.
        To repplace input variable names, pass a dictionary of the form:
        * {`old_variable_name`:`new_variable_name`}

    dfilter: float, default FILTER_NONE
        Filter applied to the loaded models predictors (not on ``field``).

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    flags : saqc.Flags
        The quality flags of data

    Notes
    -----
    The process of prediction works as follows:

    1. The model stored to ``results_path``/``field``(default) or ``results_path``/``model_folder`` is
       loaded.

    2. Input data to the model is prepared in the same way as it got prepared when training the model.
       This means, that `variable fields` the model has been trained on, have to be present in the data.
       To replace/rename input variables for a certain model, use the ``assign_features`` parameter.

    3. Variable ``field`` gets overridden with the predictions results, if `target`` is not passed. Otherwise
       results are stored to ``target``.

    Note, If a MultiOutput model was trained, it is likely to get multiple predictions for the same timestamps.
       In this case, ``pred_agg`` is applied to aggregate the predictions or select one of them.
       Predictions are generated in a rolling window. Multiple predictions for the same value are ordered
       according to the order of the prediction window they were covered by.

    Note, that your data must be sampled the way, it was sampled when training the model.

    See Also
    --------
    * :py:meth:`saqc.SaQC.trainModel`
    * :py:meth:`saqc.SaQC.modelImpute`
    * :py:meth:`saqc.SaQC.modelFlag`
    """

    assign_features = assign_features or {}
    if model_folder is None:
        model_folder = os.path.join(results_path, field)
    else:
        model_folder = os.path.join(results_path, model_folder)

    with open(os.path.join(model_folder, "config.pkl"), "rb") as f:
        sampler_config = pickle.load(f)

    with open(os.path.join(model_folder, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    sampler_config["predictors"] = [
        p if p not in assign_features.keys() else assign_features[p]
        for p in sampler_config["predictors"]
    ]

    sampler_config["drop_na_samples"] = (
        drop_na_samples or sampler_config["drop_na_samples"]
    )

    window, data_in, x_mask, target, target_idx, na_filter_x = _getSamplerParams(
        data, flags, **sampler_config
    )

    if sampler_config["freq"]:
        if not getFreqDelta(data_in.index) == sampler_config["freq"]:
            raise IndexError(
                f'Prediction data not sampled at the same rate, the model was trained at: {sampler_config["freq"]}'
            )

    if dfilter < np.inf:
        for f in sampler_config["predictors"]:
            data_in.loc[flags[f] >= dfilter, f] = np.nan

    samples = _generateSamples(
        X=sampler_config["predictors"],
        Y=target,
        sub_len=window,
        data=data_in,
        target_idx=target_idx,
        x_mask=x_mask,
        na_filter_x=na_filter_x,
        na_filter_y=False,
    )

    y_pred = model.predict(samples[0])
    if len(target_idx) > 1:
        pred_ser = _mergePredictions(
            data_in.index, len(target_idx), y_pred, samples[2], pred_agg
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
    drop_na_samples: Optional[bool] = None,
    assign_features: Optional[dict] = None,
    dfilter: float = BAD,
    **kwargs,
):
    """
    Use a trained (binary classifier) model for data flagging.

    Parameters
    ----------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldnames of the variable to be flagged.

    flags : saqc.Flags
        Container to store flags of the data.

    results_path : str
        Path to the models parent folder.

    pred_agg : callable, default np.nanmean
        Function for aggregation of multiple predictions associated with the same timestep.

    model_folder : str, default None
        Folder containing the model data.
        If ``None`` (default), a folder named ``field`` is searched.
        The folder must contain:
        * the pickled model object, ``model.pkl`` (A sklearn-style model object, implementing
          a ``predict`` method
        * the pickled configuration dictionary, ``config.pkl``.

    drop_na_samples : bool, default None
        Calculate predictions for input samples containing invalid (flagged or NaN)
        values. Defaults to the value the prediction model has been trained with.

    assign_features : dict, default None
        By default, input features to the model have to be (named) the same, as the
        model has been trained with.
        To repplace input variable names, pass a dictionary of the form:
        * {`old_variable_name`:`new_variable_name`}

    dfilter : float, default BAD
        Filter applied to the loaded models predictors (not on ``field``).

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data

    Notes
    -----
    The process of flags prediction works as follows:

    1. The model stored to ``results_path``/``field``(default) or ``results_path``/``model_folder`` is
       loaded.

    2. Input data to the model is prepared in the same way as it got prepared when training the model.
       This means, that `variable fields` the model has been trained on, have to be present in the data.
       To replace/rename input variables for a certain model, use the ``assign_features`` parameter.

    3. Variable ``field`` gets flagged, where the model predicts the positive class.

    Note, If a MultiOutput model was trained, it is likely to get multiple predictions for the same timestamps.
       In this case, ``pred_agg`` is applied to aggregate the predictions or select one of them.
       Predictions are generated in a rolling window. Multiple predictions for the same value are ordered
       according to the order of the prediction window they were covered by.

    Note, that the model applied must be a binary classification model.

    See Also
    --------
    * :py:meth:`saqc.SaQC.trainModel`
    * :py:meth:`saqc.SaQC.modelImpute`
    * :py:meth:`saqc.SaQC.modelPredict`
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
        drop_na_samples=drop_na_samples,
        dfilter=dfilter,
        assign_features=assign_features,
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
    drop_na_samples: Optional[bool] = None,
    assign_features: Optional[dict] = None,
    dfilter: float = BAD,
    flag: float = UNFLAGGED,
    **kwargs,
):
    """
    Use a trained model for data imputation.

    Imputation is tried to be performed for missing as well as flagged data in field.

    Parameters
    ----------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldnames of the variable to impute.

    flags : saqc.Flags
        Container to store flags of the data.

    results_path : str
        Path to the models parent folder.

    pred_agg : callable, default np.nanmean
        Function for aggregation of multiple predictions associated with the same timestep.

    model_folder : str, None
        Folder containing the model data.
        If ``None`` (default), a folder named ``field`` is searched.
        The folder must contain:
        * the pickled model object, ``model.pkl`` (A sklearn-style model object, implementing
          a ``predict`` method
        * the pickled configuration dictionary, ``config.pkl``.

    drop_na_samples : bool, default None
        Calculate predictions for input samples containing invalid (flagged or NaN)
        values. Defaults to the value the prediction model has been trained with.

    assign_features : dict, default None
        By default, input features to the model have to be (named) the same, as the
        model has been trained with.
        To repplace input variable names, pass a dictionary of the form:
        * {`old_variable_name`:`new_variable_name`}

    dfilter : float, default BAD
        Filter applied to the loaded models predictors (not on ``field``!).

    flag : float, default UNFLAGGED
        The flag level to be assigned to imputed values.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data

    Notes
    -----
    The process of imputation works as follows:

    1. The model stored to ``results_path``/``field``(default) or ``results_path``/``model_folder`` is
       loaded.

    2. Input data to the model is prepared in the same way as it got prepared when training the model.
       This means, that `variable fields` the model has been trained on, have to be present in the data.
       To replace/rename input variables for a certain model, use the ``assign_features`` parameter.

    3. Imputation: Missing and Flagged Values in ``field`` get replaced by model predictions, if one can be calculated.

    Note, if a MultiOutput model was trained, it is likely to get multiple predictions for the same timestamps.
       In this case, ``pred_agg`` is applied to aggregate the predictions or select one of them.
       Predictions are generated in a rolling window. Multiple predictions for the same value are ordered
       according to the order of the prediction window they were covered by.

    See Also
    --------
    * :py:meth:`saqc.SaQC.trainModel`
    * :py:meth:`saqc.SaQC.modelFlag`
    * :py:meth:`saqc.SaQC.modelPredict`
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
        drop_na_samples=drop_na_samples,
        assign_features=assign_features,
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
    data, flags = dropField(data, temp_trg, flags)
    return data, flags
