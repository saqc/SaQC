#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-


"""
Detecting breaks in data.

This module provides functions to detect and flag breaks in data, for example temporal
gaps (:py:func:`flagMissing`), jumps and drops (:py:func:`flagJumps`) or temporal
isolated values (:py:func:`flagIsolated`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from saqc import BAD, FILTER_ALL
from saqc.core import flagging, register
from saqc.funcs.changepoints import _getChangePoints
from saqc.lib.checking import validateMinPeriods, validateWindow
from saqc.lib.tools import isunflagged

if TYPE_CHECKING:
    from saqc.core.core import SaQC


class BreaksMixin:
    @register(mask=[], demask=[], squeeze=["field"])
    def flagMissing(
        self: "SaQC",
        field: str,
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> "SaQC":
        """
        Flag NaNs in data.

        By default, only NaNs are flagged, that not already have a flag.
        `dfilter` can be used to pass a flag that is used as threshold.
        Each flag worse than the threshold is replaced by the function.
        This is, because the data gets masked (with NaNs) before the
        function evaluates the NaNs.
        """

        datacol = self._data[field]
        mask = datacol.isna()

        mask = isunflagged(self._flags[field], dfilter) & mask

        self._flags[mask, field] = flag
        return self

    @flagging()
    def flagIsolated(
        self: "SaQC",
        field: str,
        gap_window: str,
        group_window: str,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Find and flag temporal isolated groups of data.

        The function flags arbitrarily large groups of values, if they are surrounded by
        sufficiently large data gaps. A gap is a timespan containing either no data at all
        or NaNs only.

        Parameters
        ----------
        gap_window :
            Minimum gap size required before and after a data group to consider it
            isolated. See condition (2) and (3)

        group_window :
            Maximum size of a data chunk to consider it a candidate for an isolated group.
            Data chunks that are bigger than the :py:attr:`group_window` are ignored.
            This does not include the possible gaps surrounding it.
            See condition (1).

        Notes
        -----
        A series of values :math:`x_k,x_{k+1},...,x_{k+n}`, with associated
        timestamps :math:`t_k,t_{k+1},...,t_{k+n}`, is considered to be isolated, if:

        1. :math:`t_{k+1} - t_n <` `group_window`
        2. None of the :math:`x_j` with :math:`0 < t_k - t_j <` `gap_window`, is valid (preceding gap).
        3. None of the :math:`x_j` with :math:`0 < t_j - t_(k+n) <` `gap_window`, is valid (succeeding gap).

        """
        validateWindow(gap_window, name="gap_window", allow_int=False)
        validateWindow(group_window, name="group_window", allow_int=False)

        dat = self._data[field].dropna()
        if dat.empty:
            return self

        gap_ends = dat.rolling(gap_window).count() == 1
        gap_ends.iloc[0] = False
        gap_ends = gap_ends[gap_ends]
        gap_starts = dat[::-1].rolling(gap_window).count()[::-1] == 1
        gap_starts.iloc[-1] = False
        gap_starts = gap_starts[gap_starts]
        if gap_starts.empty:
            return self

        gap_starts = gap_starts[1:]
        gap_ends = gap_ends[:-1]
        isolated_groups = gap_starts.index - gap_ends.index < group_window
        gap_starts = gap_starts[isolated_groups]
        gap_ends = gap_ends[isolated_groups]
        to_flag = pd.Series(False, index=dat.index)
        for s, e in zip(gap_starts.index, gap_ends.index):
            # what gets flagged are the groups between the gaps, those range from
            # the end of one gap (gap_end) to the beginning of the next (gap_start)
            to_flag[e:s] = True

        to_flag = to_flag.reindex(self._data[field].index, fill_value=False)
        self._flags[to_flag.to_numpy(), field] = flag
        return self

    @flagging()
    def flagJumps(
        self: "SaQC",
        field: str,
        thresh: float,
        window: str,
        min_periods: int = 1,
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> "SaQC":
        """
        Flag jumps and drops in data.

        Flag data where the mean of its values significantly changes (where the data "jumps" from one
        value level to another).
        Value changes are detected by comparing the mean for two adjacent rolling windows. Whenever
        the difference between the mean in the two windows exceeds :py:attr:`thresh` , the value between
        the windows is flagged.

        Parameters
        ----------
        thresh :
            Threshold value by which the mean of data has to jump, to trigger flagging.

        window :
            Size of the two moving windows. This determines the number of observations used for
            calculating the mean in every window. The window size should be big enough to yield enough
            samples for a reliable mean calculation, but it should also not be arbitrarily big, since
            it also limits the density of jumps that can be detected.
            More precisely: Jumps that are not distanced to each other by more than three fourth (3/4)
            of the selected :py:attr:`window` size, will not be detected reliably.

        min_periods :
            The minimum number of observations in :py:attr:`window` required to calculate a valid mean value.

        Examples
        --------

        Below picture gives an abstract interpretation of the parameter interplay in case of a positive
        value jump, initialising a new mean level.

        .. figure:: /resources/images/flagJumpsPic.png

           The two adjacent windows of size `window` roll through the whole data series. Whenever the mean values in
           the two windows differ by more than `thresh`, flagging is triggered.

        Notes
        -----

        Jumps that are not distanced to each other by more than three fourth (3/4) of the
        selected window size, will not be detected reliably.
        """
        validateWindow(window, allow_int=False)
        validateMinPeriods(min_periods)

        mask = _getChangePoints(
            data=self._data[field],
            stat_func=lambda x, y: np.abs(np.mean(x) - np.mean(y)),
            thresh_func=lambda x, y: thresh,
            window=window,
            min_periods=min_periods,
            result="mask",
        )

        mask = isunflagged(self._flags[field], dfilter) & mask
        self._flags[mask, field] = flag
        return self
