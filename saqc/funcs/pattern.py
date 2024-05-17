#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING

import fastdtw

from saqc import BAD
from saqc.core import flagging
from saqc.lib.rolling import removeRollingRamps
from saqc.lib.tools import getFreqDelta

if TYPE_CHECKING:
    from saqc import SaQC

import functools
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import signal

def _fullfillTasks(tasks, F):
    if len(tasks) > 1:
        with mp.Pool(len(tasks)) as pool:
            result = pool.map(F, tasks, chunksize=1)
    else:
        result = [F(tasks[0])]
    return result


def _makeTasks(scale_vals, d_len, min_tasks=50):
    core_count = min(mp.cpu_count(), len(scale_vals) // min_tasks)
    print(f"Making {core_count} tasks")
    enum = np.arange(len(scale_vals))
    return [
        list(zip(enum[k :: int(core_count)], scale_vals[k :: int(core_count)]))
        for k in range(int(core_count))
    ]


def _evalScoredScales(
    qc_arr,
    scales,
    base_series,
    min_j,
    idx_map,
    scale_vals,
    bound_scales,
    width_factor,
    mi_ma,
):
    critical = qc_arr.any(axis=1)
    scales[~qc_arr] = np.nan

    r_test_vals = scales.copy()
    r_test_vals[~critical, :] = np.nan
    idx_map = idx_map[critical]
    idx_bool = critical
    test_vals = scales[critical, :].copy()
    agged_vals = np.zeros(scales.shape)
    for v in range(r_test_vals.shape[1]):
        width = scale_vals[v]
        agged_vals[:, v] = (
            pd.Series(r_test_vals[:, v])
            .rolling(width, center=True, min_periods=0)
            .max()
            .values
        )

    agged_counts = (~np.isnan(agged_vals)).sum(axis=1)
    idx_bool = idx_bool & (agged_counts != 1)

    test_vals = test_vals[idx_bool[idx_map]]
    idx_map = idx_map[idx_bool[idx_map]]

    critical_stamps, critical_scales_idx = _getCritical(
        idx_map, agged_vals, agged_counts
    )

    critical_stamps, critical_scales_idx = _rmBounds(
        critical_stamps,
        critical_scales_idx,
        (bound_scales, len(scale_vals - bound_scales)),
    )
    to_flag = _edgeDetect(
        base_series=base_series,
        critical_stamps=critical_stamps,
        critical_scales=scale_vals[critical_scales_idx],
        width_factor=width_factor,
        test_vals=test_vals,
        min_j=min_j,
        scale_vals=scale_vals,
        idx_map=idx_map,
        mi_ma=mi_ma,
    )
    return to_flag


def _mpFunc(s, scale, opt_kwargs):
    similarity_scores, similarity_scores_inv, reduction_factor = _waveSimilarityScoring(
        scale,
        signal.ricker(s * 10, s),
        opt_kwargs=opt_kwargs,
    )
    similarity_scores = _similarityScoreReduction(
        result=similarity_scores,
        scale=scale,
        width=s * 10,
    )
    similarity_scores_inv = _similarityScoreReduction(
        result=similarity_scores_inv,
        scale=-scale,
        width=s * 10,
    )

    return similarity_scores, reduction_factor, similarity_scores_inv


def _mpTask(S, scales, base_series, opt_kwargs, thresh):
    out_r = [None] * len(S)
    i = 0
    for s in S:
        out = _mpFunc(s[1], scales[s[0]], opt_kwargs)
        qc_arr = np.zeros(len(base_series)).astype(
            bool
        )
        qc_arr_inv = np.zeros(len(base_series)).astype(bool)
        order = s[1] // out[1]
        d = _argminSer(out[0].values[:: out[1]], order=order, max=thresh)
        d_inv = _argminSer(out[2].values[:: out[1]], order=order, max=thresh)
        qc_arr[:: out[1]] = d
        qc_arr_inv[:: out[1]] = d_inv
        out_r[i] = (qc_arr, s[1], out[1], qc_arr_inv)
        i += 1

    return out_r


def _getValueSlice(idx, base_range, value_ser):
    slice_range = int(base_range * 0.5)
    value_slice = slice(
        idx - slice_range,
        idx + slice_range + 1,
    )
    return value_ser.iloc[value_slice].values, slice_range


def _getAnomalyCenter(test_scale, idx_map, critical_stamp):
    s = ~np.isnan(test_scale)
    s = idx_map[s]
    idx = np.argmin(np.abs(critical_stamp - s))
    idx = s[idx]
    return idx


def _getEdgeIdx(x, y, min_jump):
    _, path = fastdtw.fastdtw(x, y, radius=int(len(y)))
    path = np.array(path)
    offset_start = path[:, 0].argmax()
    if offset_start == (len(y) - 1):
        return -1
    start_jump = y[offset_start] - y[offset_start - 1]

    if start_jump < min_jump:
        return -1
    return offset_start


def _getEdges(anomaly_ser, anomaly_range, m_start, m_end, min_jump, mi_ma):
    offset_start = _getEdgeIdx(
        x=np.array([anomaly_ser[0], m_start]),
        y=anomaly_ser[:anomaly_range],
        min_jump=min_jump,
    )
    y_inv = anomaly_ser[anomaly_range:][::-1].copy()
    offset_end = _getEdgeIdx(
        x=np.array([anomaly_ser[-1], m_end]), y=y_inv, min_jump=min_jump
    )
    if (offset_start < 0) or (offset_end < 0):
        return -1, -1
    offset_end = len(y_inv) - offset_end
    L = offset_end - offset_start
    if (L < mi_ma[0]) | (L > mi_ma[1]):
        return -1, -1
    return offset_start, offset_end


def _getCritical(idx_map, agged_vals, agged_counts):
    ac = agged_counts == 0
    agged_groups = ac ^ np.roll(ac, 1)
    agged_groups = np.cumsum(agged_groups)
    agged_groups = agged_groups[idx_map]
    agged_counts = agged_counts[idx_map]
    agged_counts = (
        pd.Series(agged_counts, index=idx_map).groupby(by=agged_groups).idxmax()
    )

    critical_stamps = agged_counts.values
    agged_vals = agged_vals[critical_stamps, :]
    critical_scales_idx = np.nanargmax(agged_vals, axis=1)
    sort_idx = critical_scales_idx.argsort(kind="stable")
    critical_stamps = critical_stamps[sort_idx]
    critical_scales_idx = critical_scales_idx[sort_idx]
    return critical_stamps, critical_scales_idx


def _rmBounds(critical_stamps, critical_scales_idx, bounds):
    bound_mask = (critical_scales_idx > bounds[0]) & (critical_scales_idx < (bounds[1]))
    return (critical_stamps[bound_mask], critical_scales_idx[bound_mask])


def _waveSimilarityScoring(scale, wv, opt_kwargs={"thresh": 500, "factor": 5}):
    width = len(wv)
    wv = wv - wv.min()
    wv = wv / wv.max()

    if opt_kwargs["thresh"] is None:
        reduction_factor = 1
    else:
        if not isinstance(opt_kwargs["thresh"], list):
            thr = np.array([0, opt_kwargs["thresh"]])
            fc = np.array([1, opt_kwargs["factor"]])
        else:
            thr = np.array([0] + opt_kwargs["thresh"])
            fc = np.array([1] + opt_kwargs["factor"])
        lv = np.where(width >= thr)[0][-1]
        reduction_factor = int(fc[lv])
    wv_ = wv[::reduction_factor]

    r, r_inv = _strideTrickser(scale[::reduction_factor], wv_.shape[0], wv_)

    result = np.full([len(scale)], np.nan)
    result_inv = result.copy()
    w = width // 2
    result[w : w + (len(r) * reduction_factor) : reduction_factor] = r
    result_inv[w : w + (len(r_inv) * reduction_factor) : reduction_factor] = r_inv
    result = pd.Series(result)
    result_inv = pd.Series(result_inv)
    if reduction_factor > 1:
        result = result.interpolate("linear", limit=reduction_factor)
        result_inv = result_inv.interpolate("linear", limit=reduction_factor)

    return result, result_inv, reduction_factor


def _edgeDetect(
    base_series,
    critical_stamps,
    critical_scales,
    width_factor,
    test_vals,
    min_j,
    scale_vals,
    idx_map,
    mi_ma,
):
    critical_widths = [width_factor * w for w in critical_scales]
    to_finally_flag = np.zeros(len(base_series)).astype(bool)
    to_flag = to_finally_flag.copy()

    for c in enumerate(critical_stamps):
        to_flag[:] = False
        scale_iloc = scale_vals.searchsorted(critical_scales[c[0]])
        idx = _getAnomalyCenter(test_vals[:, scale_iloc], idx_map, c[1])
        anomaly_ser, anomaly_range = _getValueSlice(
            idx, critical_widths[c[0]], base_series
        )
        inner_ser, inner_range = _getValueSlice(idx, critical_scales[c[0]], base_series)

        if min_j is None:
            min_jump = 2*np.median(np.abs(np.diff(anomaly_ser)))
        else:
            min_jump = min_j

        if len(inner_ser) == 1:
            m_start = m_end = inner_ser[0]
        else:
            m_start = np.median(inner_ser[:inner_range])  # inner_ser[:c[1]].median()
            m_end = np.median(inner_ser[inner_range:])

        offset_start, offset_end = _getEdges(
            anomaly_ser=anomaly_ser,
            anomaly_range=anomaly_range,
            m_start=m_start,
            m_end=m_end,
            min_jump=min_jump,
            mi_ma=mi_ma,
        )
        if offset_start < 0:
            continue

        outlier_slice = slice(idx - anomaly_range + offset_start, idx + offset_end)
        to_flag[outlier_slice] = True

        to_finally_flag |= to_flag
    return to_finally_flag


def _strideTrickser(data, win_len, wave):
    stack_view = np.lib.stride_tricks.sliding_window_view(data, win_len, (0))
    samples = stack_view.shape[0]
    mi = stack_view.min(axis=1).reshape(samples, 1)
    r = (stack_view - mi) / (stack_view.max(axis=1).reshape(samples, 1) - mi)
    r1 = np.abs(r - wave).mean(axis=1)
    r2 = np.abs((-r) - (wave-1)).mean(axis=1)
    return r1, r2


def _argminSer(x, order, max=100):
    idx = signal.argrelmin(x, order=order)[0]
    thresh_mask = x < max
    y = np.zeros(len(x)).astype(bool)
    y[idx] = True
    y = y & thresh_mask
    return y


def _similarityScoreReduction(
    result,
    scale,
    width,
    bumb_cond_factor=1.5,
):

    bool_signs = scale > 0
    switches = ~(bool_signs == np.roll(bool_signs, 1))
    signum_groups = np.cumsum(switches)

    consec_sign_groups_vals = result.groupby(by=signum_groups)
    filter_func = lambda x: x.count() > bumb_cond_factor * (width / 10)
    filtered = consec_sign_groups_vals.filter(filter_func, dropna=False)

    out = pd.Series(filtered.values, index=result.index)
    return out


def offSetSearch(
    base_series,
    scale_vals,
    wavelet,
    min_j,
    mi_ma,
    thresh=0.1,
    width_factor=2.5,
    bound_scales=10,
    opt_kwargs={"thresh": 500, "factor": 5},
):
    idx_map = np.arange(len(base_series))
    scales = signal.cwt(base_series.values, wavelet, scale_vals)

    qc_arr = np.zeros([len(base_series), len(scale_vals)]).astype(bool)
    qc_arr_inv = qc_arr.copy()
    scale_order = {s: n for n, s in enumerate(scale_vals)}

    tasks = _makeTasks(scale_vals, len(base_series))

    worker_func = functools.partial(
        _mpTask,
        scales=scales,
        base_series=base_series,
        opt_kwargs=opt_kwargs,
        thresh=thresh,
    )
    result = _fullfillTasks(tasks, worker_func)
    for task_return in result:
        for r in task_return:
            qc_arr[:, scale_order[r[1]]] = r[0]
            qc_arr_inv[:, scale_order[r[1]]] = r[3]

    scales = scales.T

    to_flag = _evalScoredScales(
        qc_arr,
        scales.copy(),
        base_series,
        min_j,
        idx_map,
        scale_vals,
        bound_scales,
        width_factor,
        mi_ma,
    )
    to_flag = to_flag | _evalScoredScales(
        qc_arr_inv,
        -scales,
        -base_series,
        min_j,
        idx_map,
        scale_vals,
        bound_scales,
        width_factor,
        mi_ma,
    )
    return to_flag


def calculateDistanceByDTW(
    data: pd.Series, reference: pd.Series, forward: bool = True, normalize: bool = True
):
    """
    Calculate the DTW-distance of data to pattern in a rolling calculation.

    The data is compared to pattern in a rolling window.
    The size of the rolling window is determined by the timespan defined
    by the first and last timestamp of the reference data's datetime index.

    For details see the linked functions in the `See also` section.

    Parameters
    ----------
    data :
        Data series. Must have datetime-like index, and must be regularly sampled.

    reference :
        Reference series. Must have datetime-like index, must not contain NaNs
        and must not be empty.

    forward:
        If `True`, the distance value is set on the left edge of the data chunk. This
        means, with a perfect match, `0.0` marks the beginning of the pattern in
        the data. If `False`, `0.0` would mark the end of the pattern.

    normalize :
        If `False`, return unmodified distances.
        If `True`, normalize distances by the number of observations in the reference.
        This helps to make it easier to find a good cutoff threshold for further
        processing. The distances then refer to the mean distance per datapoint,
        expressed in the datas units.

    Returns
    -------
    distance : pd.Series

    Notes
    -----
    The data must be regularly sampled, otherwise a ValueError is raised.
    NaNs in the data will be dropped before dtw distance calculation.

    See also
    --------
    flagPatternByDTW : flag data by DTW
    """
    if reference.hasnans or reference.empty:
        raise ValueError("reference must not have nan's and must not be empty.")

    winsz: pd.Timedelta = reference.index.max() - reference.index.min()
    reference = reference.to_numpy()

    def isPattern(chunk):
        if forward:
            return fastdtw.fastdtw(chunk[::-1], reference)[0]
        else:
            return fastdtw.fastdtw(chunk, reference)[0]

    # generate distances, excluding NaNs
    nonas = data.dropna()
    rollover = nonas[::-1] if forward else nonas
    arr = rollover.rolling(winsz, closed="both").apply(isPattern, raw=True).to_numpy()
    distances = pd.Series(arr[::-1] if forward else arr, index=nonas.index)
    removeRollingRamps(distances, window=winsz, inplace=True)

    if normalize:
        distances /= len(reference)

    return distances.reindex(index=data.index)  # reinsert NaNs


class PatternMixin:
    # todo should we mask `reference` even if the func fail if reference has NaNs
    @flagging()
    def flagPlateau(
        self: "SaQC",
        field: str,
        min_length: int | str,
        max_length: int | str,
        granularity: int | str = None,
        min_jump: float = None,
        opt_strategy: int = None,
        opt_thresh: int = None,
        fill_strat: str = "pad",
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag anomalous value plateaus discriminated by temporal extension.

        Parameters
        ----------
        min_length:
            Minimum temporal extension of values to qualify as plateau

        max_length:
            Maximum temporal extension of values to qualify as plateau (upper detection limit)

        granularity:
            Precision of search: The smaller the better, but also, the more numerically expensive.

        min_jump:
            minimum margin anomalies/plateaus have to differ from directly preceding and succeeding periods.
            If not passed an explicit value (default), the minimum jump threshold will be derived automatically from the median
            of the local absolute difference between any two periods in the vicinity of any potential anomaly.

        Notes
        -----
        Minimum length of plateaus should be selected higher than 5 times the sampling rate.
        To search for shorter plateaus/anomalies, use :py:meth:~`saqc.SaQC.flagUniLOF` or :py:meth:~`saqc.SaQC.flagZScore`.
        """
        opt_strategy = opt_strategy or [5, 10, 20, 50, 100, 1000, 10000]
        opt_thresh = opt_thresh or [250, 500, 1000, 2500, 5000, 50000, 500000]
        bound_scales = 10
        datcol = self.data[field]
        datcol = datcol.interpolate(fill_strat)
        datcol = datcol.ffill().bfill()
        freq = getFreqDelta(datcol.index)
        if freq is None:
            raise ValueError("Not a unitary sampling rate")
        if isinstance(min_length, str):
            min_length = pd.Timedelta(min_length) // freq
            max_length = pd.Timedelta(max_length) // freq
        granularity = granularity or 5
        if isinstance(granularity, str):
            granularity = pd.Timedelta(granularity) // freq

        mi_ma = min_length, max_length
        min_length = max(min_length // 2, 1)
        max_length = max_length // 2

        scale_vals = list(np.arange(min_length, max_length, granularity))
        bounding = [
            min_length - b for b in range(1, min(bound_scales, scale_vals[0]))
        ] + [max_length + b for b in range(1, bound_scales)]

        scale_vals = np.array(scale_vals + bounding)
        scale_vals.sort(kind="stable")

        to_flag = offSetSearch(
            base_series=datcol,
            scale_vals=scale_vals,
            wavelet=signal.ricker,
            min_j=min_jump,
            thresh=0.1,
            bound_scales=bound_scales,
            mi_ma=mi_ma,
            opt_kwargs={"thresh": opt_thresh, "factor": opt_strategy},
        )
        self._flags[to_flag, field] = flag
        return self

    @flagging()
    def flagPatternByDTW(
        self: "SaQC",
        field: str,
        reference: str,
        max_distance: float = 0.0,
        normalize: bool = True,
        plot: bool = False,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Pattern Recognition via Dynamic Time Warping.

        The steps are:
        1. work on a moving window

        2. for each data chunk extracted from each window, a distance
           to the given pattern is calculated, by the dynamic time warping
           algorithm [1]

        3. if the distance is below the threshold, all the data in the
           window gets flagged

        Parameters
        ----------
        reference :
            The name in `data` which holds the pattern. The pattern must
            not have NaNs, have a datetime index and must not be empty.

        max_distance :
            Maximum dtw-distance between chunk and pattern, if the distance
            is lower than ``max_distance`` the data gets flagged. With
            default, ``0.0``, only exact matches are flagged.

        normalize :
            If `False`, return unmodified distances.
            If `True`, normalize distances by the number of observations
            of the reference. This helps to make it easier to find a
            good cutoff threshold for further processing. The distances
            then refer to the mean distance per datapoint, expressed
            in the datas units.

        plot :
            Show a calibration plot, which can be quite helpful to find
            the right threshold for `max_distance`. It works best with
            `normalize=True`. Do not use in automatic setups / pipelines.
            The plot show three lines:

            - data: the data the function was called on
            - distances: the calculated distances by the algorithm
            - indicator: have to distinct levels: `0` and the value of
              `max_distance`. If `max_distance` is `0.0` it defaults to
              `1`. Everywhere where the indicator is not `0` the data
              will be flagged.

        Notes
        -----
        The window size of the moving window is set to equal the temporal
        extension of the reference datas datetime index.

        References
        ----------
        Find a nice description of underlying the Dynamic Time Warping
        Algorithm here:

        [1] https://cran.r-project.org/web/packages/dtw/dtw.pdf
        """
        ref = self._data[reference]
        dat = self._data[field]

        distances = calculateDistanceByDTW(dat, ref, forward=True, normalize=normalize)
        winsz = ref.index.max() - ref.index.min()

        # prevent nan propagation
        distances = distances.fillna(max_distance + 1)

        # find minima filter by threshold
        fw_min = distances[::-1].rolling(window=winsz, closed="both").min()[::-1]
        bw_min = distances.rolling(window=winsz, closed="both").min()
        minima = (fw_min == bw_min) & (distances <= max_distance)

        # Propagate True's to size of pattern.
        mask = minima.rolling(window=winsz, closed="both").sum() > 0

        if plot:
            df = pd.DataFrame()
            df["data"] = dat
            df["distances"] = distances
            df["indicator"] = mask.astype(float) * (max_distance or 1)
            df.plot()

        self._flags[mask, field] = flag
        return self
