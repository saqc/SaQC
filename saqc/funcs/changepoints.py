#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Tuple

import numpy as np
import pandas as pd

from saqc import BAD, UNFLAGGED
from saqc.core import flagging, register
from saqc.lib.checking import (
    validateCallable,
    validateChoice,
    validateMinPeriods,
    validateWindow,
)

if TYPE_CHECKING:
    from saqc import SaQC


class ChangepointsMixin:
    @flagging()
    def flagChangePoints(
        self: "SaQC",
        field: str,
        stat_func: Callable[[np.ndarray, np.ndarray], float],
        thresh_func: Callable[[np.ndarray, np.ndarray], float],
        window: str | Tuple[str, str],
        min_periods: int | Tuple[int, int],
        reduce_window: str | None = None,
        reduce_func: Callable[[np.ndarray, np.ndarray], int] = lambda x, _: x.argmax(),
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag values that represent a system state transition.

        Flag data points, where the parametrization of the assumed process generating this data,
        significantly changes.

        Parameters
        ----------
        stat_func :
             A function that assigns a value to every twin window. The backward-facing
             window content will be passed as the first array, the forward-facing window
             content as the second.

        thresh_func :
            A function that determines the value level, exceeding wich qualifies a
            timestamps func value as denoting a change-point.

        window :
            Size of the moving windows. This is the number of observations used for
            calculating the statistic.

            If it is a single frequency offset, it applies for the backward- and the
            forward-facing window.

            If two offsets (as a tuple) is passed the first defines the size of the
            backward facing window, the second the size of the forward facing window.

        min_periods :
            Minimum number of observations in a window required to perform the changepoint
            test. If it is a tuple of two int, the first refer to the backward-,
            the second to the forward-facing window.

        reduce_window :
            The sliding window search method is not an exact CP search method and usually
            there wont be detected a single changepoint, but a "region" of change around
            a changepoint.

            If `reduce_window` is given, for every window of size `reduce_window`, there
            will be selected the value with index `reduce_func(x, y)` and the others will
            be dropped.

            If `reduce_window` is None, the reduction window size equals the twin window
            size, the changepoints have been detected with.

        reduce_func : default argmax
            A function that must return an index value upon input of two arrays x and y.
            First input parameter will hold the result from the stat_func evaluation for
            every reduction window. Second input parameter holds the result from the
            `thresh_func` evaluation.
            The default reduction function just selects the value that maximizes the
            `stat_func`.
        """
        validateCallable(stat_func, "stat_func")
        validateCallable(thresh_func, "thresh_func")
        validateCallable(reduce_func, "reduce_func")
        # Hint: windows are checked in _getChangePoints

        mask = _getChangePoints(
            data=self._data[field],
            stat_func=stat_func,
            thresh_func=thresh_func,
            window=window,
            min_periods=min_periods,
            reduce_window=reduce_window,
            reduce_func=reduce_func,
            result="mask",
        )
        self._flags[mask, field] = flag
        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def assignChangePointCluster(
        self: "SaQC",
        field: str,
        stat_func: Callable[[np.ndarray, np.ndarray], float],
        thresh_func: Callable[[np.ndarray, np.ndarray], float],
        window: str | Tuple[str, str],
        min_periods: int | Tuple[int, int],
        reduce_window: str | None = None,
        reduce_func: Callable[
            [np.ndarray, np.ndarray], float
        ] = lambda x, _: x.argmax(),
        model_by_resids: bool = False,
        **kwargs,
    ) -> "SaQC":
        """
        Label data where it changes significantly.

        The labels will be stored in data. Unless `target` is given the labels will
        overwrite the data in `field`. The flags will always set to `UNFLAGGED`.

        Assigns label to the data, aiming to reflect continuous regimes of the processes
        the data is assumed to be generated by. The regime change points detection is
        based on a sliding window search.


        Parameters
        ----------
        stat_func :
            A function that assigns a value to every twin window. Left window content will
            be passed to first variable,
            right window content will be passed to the second.

        thresh_func :
            A function that determines the value level, exceeding wich qualifies a
            timestamps func value as denoting a changepoint.

        window :
            Size of the rolling windows the calculation is performed in. If it is a single
            frequency offset, it applies for the backward- and the forward-facing window.

            If two offsets (as a tuple) is passed the first defines the size of the
            backward facing window, the second the size of the forward facing window.

        min_periods :
            Minimum number of observations in a window required to perform the changepoint
            test. If it is a tuple of two int, the first refer to the backward-,
            the second to the forward-facing window.

        reduce_window :
            The sliding window search method is not an exact CP search method and usually
            there won't be detected a single changepoint, but a "region" of change around
            a changepoint. If `reduce_window` is given, for every window of size
            `reduce_window`, there will be selected the value with index `reduce_func(x,
            y)` and the others will be dropped. If `reduce_window` is None, the reduction
            window size equals the twin window size, the changepoints have been detected
            with.

        reduce_func : default argmax
            A function that must return an index value upon input of two arrays x and y.
            First input parameter will hold the result from the stat_func evaluation for
            every reduction window. Second input parameter holds the result from the
            thresh_func evaluation. The default reduction function just selects the value
            that maximizes the stat_func.

        model_by_resids :
            If True, the results of `stat_funcs` are written, otherwise the regime labels.
        """
        validateCallable(stat_func, "stat_func")
        validateCallable(thresh_func, "thresh_func")
        validateCallable(reduce_func, "reduce_func")
        # Hint: windows are checked in _getChangePoints

        rtyp = "residual" if model_by_resids else "cluster"
        cluster = _getChangePoints(
            data=self._data[field],
            stat_func=stat_func,
            thresh_func=thresh_func,
            window=window,
            min_periods=min_periods,
            reduce_window=reduce_window,
            reduce_func=reduce_func,
            result=rtyp,  # type: ignore
        )
        self._data[field] = cluster
        # we set flags here against our standard policy,
        # which is not to overwrite existing flags
        self._flags[:, field] = UNFLAGGED
        return self


def _getChangePoints(
    data: pd.Series,
    stat_func: Callable[[np.ndarray, np.ndarray], float],
    thresh_func: Callable[[np.ndarray, np.ndarray], float],
    window: str | Tuple[str, str],
    min_periods: int | Tuple[int, int],
    reduce_window: str | None = None,
    reduce_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, _: x.argmax(),
    result: Literal["cluster", "residual", "mask"] = "mask",
) -> pd.Series:
    """
    TODO: missing docstring

    Parameters
    ----------
    data :
    stat_func :
    thresh_func :
    window :
    min_periods :
    reduce_window :
    reduce_func :
    result :

    Returns
    -------
    """
    validateChoice(result, "result", ["cluster", "residual", "mask"])

    orig_index = data.index
    data = data.dropna()  # implicit copy

    if isinstance(window, (list, tuple)):
        bwd_window, fwd_window = window
        validateWindow(fwd_window, name="window[0]", allow_int=False)
        validateWindow(bwd_window, name="window[1]", allow_int=False)
    else:
        validateWindow(window, name="window", allow_int=False)
        bwd_window = fwd_window = window

    if isinstance(min_periods, (list, tuple)):
        bwd_min_periods, fwd_min_periods = min_periods
        validateMinPeriods(bwd_min_periods, "min_periods[0]")
        validateMinPeriods(fwd_min_periods, "min_periods[1]")
    else:
        validateMinPeriods(min_periods)
        bwd_min_periods = fwd_min_periods = min_periods

    if reduce_window is None:
        s = int(
            pd.Timedelta(bwd_window).total_seconds()
            + pd.Timedelta(fwd_window).total_seconds()
        )
        reduce_window = f"{s}s"
    validateWindow(reduce_window, name="reduce_window", allow_int=False)

    # find window bounds arrays..
    num_index = pd.Series(range(len(data)), index=data.index, dtype=int)
    # ... for the normal (backwards) case..
    rolling = num_index.rolling(bwd_window, min_periods=0)
    bwd_start = rolling.min().to_numpy(dtype=int)
    bwd_end = rolling.max().to_numpy(dtype=int) + 1
    # ... and aging for the forward case.
    rolling = num_index[::-1].rolling(fwd_window, min_periods=0, closed="left")
    fwd_start = rolling.min().fillna(len(num_index)).to_numpy(dtype=int)[::-1]
    fwd_end = (rolling.max() + 1).fillna(len(num_index)).to_numpy(dtype=int)[::-1]

    min_mask = (fwd_end - fwd_start >= fwd_min_periods) & (
        bwd_end - bwd_start >= bwd_min_periods
    )

    fwd_end = fwd_end[min_mask]
    split = bwd_end[min_mask]
    bwd_start = bwd_start[min_mask]
    masked_index = data.index[min_mask]
    check_len = len(fwd_end)
    data_arr = data.values

    args = data_arr, bwd_start, fwd_end, split, stat_func, thresh_func, check_len
    stat_arr, thresh_arr = _slidingWindowSearch(*args)

    result_arr = stat_arr > thresh_arr

    if result == "residuals":
        residuals = pd.Series(np.nan, index=orig_index)
        residuals[masked_index] = stat_arr
        return residuals

    det_index = masked_index[result_arr]
    detected = pd.Series(True, index=det_index)

    length = len(detected)
    # find window bounds arrays
    num_index = pd.Series(range(length), index=detected.index, dtype=int)
    rolling = num_index.rolling(window=reduce_window, closed="both", center=True)
    start = rolling.min().to_numpy(dtype=int)
    end = (rolling.max() + 1).to_numpy(dtype=int)

    detected = _reduceCPCluster(
        stat_arr[result_arr],
        thresh_arr[result_arr],
        start,
        end,
        reduce_func,
        length,
    )
    det_index = det_index[detected]

    # The changepoint is the point "after" the change.
    # So the detected index has to be shifted by one
    # with regard to the data index.
    shifted = (
        pd.Series(True, index=det_index)
        .reindex(data.index, fill_value=False)
        .shift(fill_value=False)
    )
    det_index = shifted.index[shifted]

    mask = pd.Series(False, index=orig_index)
    mask[det_index] = True
    if result == "mask":
        return mask

    cluster = mask.cumsum()
    cluster += 1  # start cluster labels with one, not zero
    if result == "cluster":
        return cluster

    raise ValueError(
        f"'result' must be one of 'cluster', 'mask' or 'residuals' not {result}"
    )


def _slidingWindowSearch(
    data_arr, bwd_start, fwd_end, split, stat_func, thresh_func, num_val
):
    stat_arr = np.zeros(num_val)
    thresh_arr = np.zeros(num_val)
    for win_i in range(0, num_val - 1):
        x = data_arr[bwd_start[win_i] : split[win_i]]
        y = data_arr[split[win_i] : fwd_end[win_i]]
        stat_arr[win_i] = stat_func(x, y)
        thresh_arr[win_i] = thresh_func(x, y)
    return stat_arr, thresh_arr


def _reduceCPCluster(stat_arr, thresh_arr, start, end, obj_func, num_val):
    out_arr = np.zeros(shape=num_val, dtype=bool)
    for win_i in range(num_val):
        s, e = start[win_i], end[win_i]
        x = stat_arr[s:e]
        y = thresh_arr[s:e]
        pos = s + obj_func(x, y)
        out_arr[s:e] = False
        out_arr[pos] = True

    return out_arr
