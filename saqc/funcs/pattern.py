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

import numpy as np
import pandas as pd
from scipy import signal

import saqc


def _getCritical(idx_map, agged_vals, agged_counts, width_factor):
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


def _variationCheck(data, score, width, width_factor):
    d_r = data.rolling(int((width * 0.1) * width_factor), center=True)
    diff_md = 4 * data.diff().abs().median()
    resid = d_r.max() - d_r.min()
    mask_r = resid > diff_md
    score[~mask_r.values] = np.nan
    return score


def _waveSimilarityScoring(
    scale, wv, data, width_factor=2, opt_kwargs={"thresh": 500, "factor": 5}
):
    # derive width of the wavelet -> (current scale = width*.1)
    width = len(wv)
    # scale wavelt to [0,1]
    wv = wv - wv.min()
    wv = wv / wv.max()
    # get statistics that checks if the data is sufficiently changing to qualify as outlierish in any window of size scale*width_factor
    d_r = data.rolling(int((width * 0.1) * width_factor), center=True)
    diff_md = 4 * data.diff().abs().median()
    resid = d_r.max() - d_r.min()

    # check any width-sized window of scale, if it resembles the wavelet (in terms of the mean absolute comparison error)
    reduction_factor = 1
    print(width)
    if not isinstance(opt_kwargs["thresh"], list):
        thr = np.array([0, opt_kwargs["thresh"]])
        fc = np.array([1, opt_kwargs["factor"]])
    else:
        thr = np.array([0] + opt_kwargs["thresh"])
        fc = np.array([1] + opt_kwargs["factor"])
    lv = np.where(width >= thr)[0][-1]
    reduction_factor = int(fc[lv])
    print(f"LV:{lv}, FC:{reduction_factor}")
    wv_ = wv[::reduction_factor]

    r = stride_trickser(scale[::reduction_factor], wv_.shape[0], wv_)

    result = np.full([len(scale)], np.nan)
    w = width // 2
    result[w : w + (len(r) * reduction_factor) : reduction_factor] = r
    result = pd.Series(result)
    if reduction_factor > 1:
        result = result.interpolate("linear", limit=reduction_factor)
    # check where the scale has enough consecutive sign values to qualify for resambling the wavelet
    # bumb - qualify
    return result, reduction_factor


def _edgeDetect(
    base_series,
    critical_stamps,
    critical_scales,
    width_factor,
    test_vals,
    min_j,
    scale_vals,
    idx_map,
):
    critical_widths = [width_factor * w for w in critical_scales]
    op_series = base_series.copy()
    to_finally_flag = pd.Series(False, index=op_series.index)
    # looping over the critical timstamps (that supposedly lie in the middle of any detected anomaly

    for c in enumerate(critical_stamps):
        # in every loop prepare a series that will hold the timestamps to set flags at:
        to_flag = pd.Series(False, index=op_series.index)
        scale_iloc = scale_vals.searchsorted(critical_scales[c[0]])
        # at first, we do not apply flagging, if the detected anomaly is at a scale that is too
        # close to the limits of the scales we searched at: because this could mean it is actually
        # smaller or wider that detected and we would thus overflag/underflag

        # --------------------------------
        # correct/validate found anomalies
        # --------------------------------
        # we derive the most likely timestamp in the middle of the anomaly under test:
        s = ~np.isnan(test_vals[:, scale_iloc])
        s = idx_map[s]
        idx = np.argmin(np.abs(c[1] - s))
        idx = s[idx]
        # we cut out the a piece of the target timseries that is likely wider than the anomaly:
        anomaly_range = int(critical_widths[c[0]] * 0.5)
        anomaly_slice = slice(
            idx - anomaly_range,
            idx + anomaly_range + 1,
        )
        anomaly_ser = op_series.iloc[anomaly_slice].values
        if min_j is None:
            min_jump = np.quantile(np.abs(np.diff(anomaly_ser)), 0.9)
        else:
            min_jump = min_j

        # we cut out a smaller piece from the target timeseries, that likely only contains the actual anomaleous plateau:
        inner_range = int(critical_scales[c[0]] * 0.5)
        inner_slice = slice(
            idx - inner_range,
            idx + inner_range + 1,
        )
        inner_ser = op_series.iloc[inner_slice].values

        # deriving some statistics and making another plausibility check
        if len(inner_ser) == 1:
            m_start = m_end = inner_ser[0]
        else:
            m_start = np.median(inner_ser[:inner_range])  # inner_ser[:c[1]].median()
            m_end = np.median(inner_ser[inner_range:])

        # now we try to find the real start and ending points of the anomaly
        # the idea basically is to map the 2-periods series containing one value likely outside the
        # anomaly and the inner most value of the anomaly onto the first half of the anomaly slice with fastdtw:
        # The point where the algorithm switches from mapping the outside value to mapping the inside value,
        # is a most likely candidate for the start of the anomaly. Than, we do the same for
        # other half to get the ending point of the anomaly:

        x = np.array([anomaly_ser[0], m_start])

        y = anomaly_ser[:anomaly_range]
        _, path = fastdtw.fastdtw(x, y, radius=int(len(y)))

        path = np.array(path)
        offset_start = path[:, 0].argmax()
        if offset_start == (len(y) - 1):
            continue
        start_jump = y[offset_start] - y[offset_start - 1]

        if start_jump < min_jump:
            continue

        # repeat process for the second half of the anomaly
        x = np.array([m_end, anomaly_ser[-1]])

        y = anomaly_ser[anomaly_range:]
        _, path = fastdtw.fastdtw(x, y, radius=int(len(y)))

        path = np.array(path)
        offset_end = path[:, 0].argmax() - 1
        if offset_end == (len(y) - 1):
            continue
        end_jump = y[offset_end] - y[offset_end + 1]
        if end_jump < min_jump:
            continue

        s = idx - anomaly_range + offset_start
        e = idx + offset_end
        outlier_slice = slice(s, e)

        if (e - s) < (critical_scales[c[0]]):
            continue

        N = min([((e - s) // 2) - 1, 20])
        uniLofScores = (
            saqc.SaQC(pd.Series(anomaly_ser, name="test"))
            .assignUniLOF("test", n=N)
            .data["test"]
        )
        if (uniLofScores[offset_start] >= -1) | (uniLofScores[offset_end] >= -1):
            continue
        to_flag.iloc[outlier_slice] = True

        # we remove the anomaly from the series, so that it wont interfere with later checks
        op_series.iloc[outlier_slice] = np.nan
        op_series = op_series.interpolate("linear")

        to_finally_flag |= to_flag
    return to_finally_flag


def stride_trickser(data, win_len, wave):
    stack_view = np.lib.stride_tricks.sliding_window_view(data, win_len, (0))
    samples = stack_view.shape[0]
    m = stack_view.min(axis=1).reshape(samples, 1)
    return (
        np.abs(
            (stack_view - m) / (stack_view.max(axis=1).reshape(samples, 1) - m) - wave
        )
    ).mean(axis=1)


def patternSearch(x, wv):
    # pattern search: scales x to [0,1] and returns mean absolute error between x and wv (the wavelet)
    x = x - x.min()
    x = x / x.max()
    return (np.abs(x - wv)).mean()


def argminSer(x, order, max=100):
    # function searches for local minima
    # get indices of local minima
    idx = signal.argrelmin(x.values, order=order)[0]
    # only consider local minima that are small enough
    thresh_mask = x < max
    # make Series holding local minima
    y = pd.Series(False, index=x.index)
    y.iloc[idx] = True
    y = y & thresh_mask
    return y.astype(bool)


def _similarityScoreReduction(
    result,
    scale,
    width,
    bumb_cond_factor=1.5,
):

    bool_signs = scale > 0
    # make series that is True when series switches signum with regard to predecessor and False otherwise
    switches = ~(bool_signs == np.roll(bool_signs, 1))
    # make cumulative sum of switch series: as a result, consecutive values that do not switch signum have the same integer assigned
    signum_groups = np.cumsum(switches)

    # use pandas grouper to group the comparison scores into partitions, where the scale has the same signum:
    consec_sign_groups = result.groupby(by=signum_groups)
    # filter function for checking wich of the groups have sufficiently many consecutive na value
    filter_func = lambda x: x.count() > bumb_cond_factor * (width / 10)
    # apply filter: where scores dont belong to groups where the scale has not siffuciently many consecutive values of same sign, the score is overridden with nan (no_score)
    filtered = consec_sign_groups.filter(filter_func, dropna=False)
    # generate the timeseries to return
    out = pd.Series(filtered.values, index=result.index)
    return out


def offSetSearch(
    base_series,
    scale_vals,
    wavelet,
    min_j,
    thresh=0.1,
    width_factor=2.5,
    bound_scales=10,
    opt_kwargs={"thresh": 500, "factor": 5},
):
    idx_map = np.arange(len(base_series))
    scales = signal.cwt(base_series.values, wavelet, scale_vals)

    res = pd.DataFrame(
        np.nan, index=base_series.index, columns=[f"score_{s}" for s in scale_vals]
    )
    reduction_factors = np.ones(len(scale_vals), dtype=int)
    for s in enumerate(scale_vals):
        # calculate and assign the comparison score series between wavelet and scale:
        similarity_scores, reduction_factor = _waveSimilarityScoring(
            scales[s[0]],
            signal.ricker(s[1] * 10, s[1]),
            base_series,
            width_factor=width_factor,
            opt_kwargs=opt_kwargs,
        )
        similarity_scores = _similarityScoreReduction(
            result=similarity_scores,
            scale=scales[s[0]],
            width=s[1] * 10,
            bumb_cond_factor=1.5,
        )
        similarity_scores = _variationCheck(
            data=base_series,
            score=similarity_scores,
            width=s[1] * 10,
            width_factor=width_factor,
        )
        res[f"score_{s[1]}"] = similarity_scores.values
        reduction_factors[s[0]] = reduction_factor

    # generate a dataframe of scales from the scales array:
    scale_cols = [f'scale_{k.split("_")[-1]}' for k in res.columns]
    scales = scales.T

    qc_frame = pd.DataFrame(False, index=res.index, columns=res.columns)
    for score in zip(res.columns, reduction_factors):
        sc = int(score[0].split("_")[-1])
        d = argminSer(res[score[0]][:: score[1]], order=sc // score[1], max=thresh)
        qc_frame.loc[d[d].index, score[0]] = True

    critical = qc_frame.any(axis=1)
    scales[~qc_frame.values] = np.nan

    r_test_vals = scales.copy()
    r_test_vals[~critical.values, :] = np.nan
    idx_map = idx_map[critical.values]
    idx_bool = critical.values
    test_vals = scales[critical.values, :].copy()
    # for later use, we also generate a dataframe that has the same size as the scales
    # frame, but contains nan values everywhere, besides for the 'critical` timestamps
    # r_test_vals = test_vals.reindex(scales.index)
    # agged_vals = pd.DataFrame(0.0, columns=scales.columns, index=scales.index)
    agged_vals = np.zeros(scales.shape)
    # The maxima for every bumb will not be at exactly the same timestamp at every scale,
    # (due to noise) -> this is why we broaden the maxima points with the rolling operation, so
    # we tha can just look for the column minima to find the optimal
    # scale for every anomaly (wich appears as bumb in the scales)
    for v in range(r_test_vals.shape[1]):
        width = scale_vals[v]
        agged_vals[:, v] = (
            pd.Series(r_test_vals[:, v])
            .rolling(width, center=True, min_periods=0)
            .max()
            .values
        )


    # for every timestamp we count in how much scales it is a part of a bumb
    agged_counts = (~np.isnan(agged_vals)).sum(axis=1)
    # timestamps that only appear on one scale as part of a bumb, are suspiceaous and
    # likely belong to noise
    idx_bool = idx_bool & (agged_counts != 1)

    # we remove those suspiceous indices from the test_values frame
    test_vals = test_vals[idx_bool[idx_map]]
    idx_map = idx_map[idx_bool[idx_map]]

    critical_stamps, critical_scales_idx = _getCritical(
        idx_map, agged_vals, agged_counts, width_factor
    )

    critical_stamps, critical_scales_idx = _rmBounds(
        critical_stamps,
        critical_scales_idx,
        (bound_scales, len(scale_vals - bound_scales)),
    )
    # ----------------------------------------------------------------
    # Next part we mainly detect the exact starting and ending points of
    # any anomaly with the help of the 'dynamic timewarping' (fastdtw) algorithm.
    # ----------------------------------------------------------------
    to_flag = _edgeDetect(
        base_series=base_series,
        critical_stamps=critical_stamps,
        critical_scales=scale_vals[critical_scales_idx],
        width_factor=width_factor,
        test_vals=test_vals,
        min_j=min_j,
        scale_vals=scale_vals,
        idx_map=idx_map,
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
        granularity: int | str = 5,
        min_jump: float = None,
        opt_strategy: int = None,
        opt_thresh: int = 500,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag anomalous value plateaus discriminated by temporal extension.

        Parameters
        ----------
        min_length:
            Minimum temporal extension of plateau/outlier

        max_length:
            Maximum temporal extension of plateau/outlier

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
        bound_scales = 10
        datcol = self.data[field]
        datcol = datcol.interpolate("time")
        freq = getFreqDelta(datcol.index)
        if freq is None:
            raise ValueError("Not a unitary sampling rate")
        if isinstance(min_length, str):
            min_length = pd.Timedelta(min_length) // freq
            max_length = pd.Timedelta(max_length) // freq
        if isinstance(granularity, str):
            granularity = pd.Timedelta(granularity) // freq

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
            opt_kwargs={"thresh": opt_thresh, "factor": opt_strategy or granularity},
        )
        to_flag |= offSetSearch(
            base_series=datcol.max() - datcol,
            scale_vals=scale_vals,
            wavelet=signal.ricker,
            min_j=min_jump,
            thresh=0.1,
            bound_scales=bound_scales,
            opt_kwargs={"thresh": opt_thresh, "factor": opt_strategy or granularity},
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
