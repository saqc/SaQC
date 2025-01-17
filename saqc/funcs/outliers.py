#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

import uuid
import warnings
from typing import TYPE_CHECKING, Callable, Sequence, Tuple

import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from outliers import smirnov_grubbs  # noqa, on pypi as outlier-utils
from scipy.stats import median_abs_deviation
from typing_extensions import Literal

from saqc import BAD, UNFLAGGED
from saqc.core import DictOfSeries, Flags, flagging, register
from saqc.lib.checking import (
    isCallable,
    isFloatLike,
    validateChoice,
    validateFraction,
    validateFrequency,
    validateMinPeriods,
    validateValueBounds,
    validateWindow,
)
from saqc.lib.docs import DOC_TEMPLATES
from saqc.lib.rolling import windowRoller
from saqc.lib.tools import getFreqDelta, isflagged, toSequence

if TYPE_CHECKING:
    from saqc import SaQC


class OutliersMixin:
    @staticmethod
    def _validateLOF(algorithm, n, p, density):
        """validate parameter for LOF and UniLOF"""
        validateValueBounds(n, "n", left=0, strict_int=True)
        validateValueBounds(p, "p", left=0, strict_int=True)
        validateChoice(
            algorithm, "algorithm", ["ball_tree", "kd_tree", "brute", "auto"]
        )
        if density != "auto" and not isFloatLike(density) and not isCallable(density):
            raise ValueError(
                f"'density' must be 'auto' or a float or a function, not {density}"
            )

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        multivariate=True,
        handles_target=False,
    )
    def flagLOF(
        self: "SaQC",
        field: Sequence[str],
        n: int = 20,
        thresh: Literal["auto"] | float = 1.5,
        algorithm: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
        p: int = 1,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag values where the Local Outlier Factor (LOF) exceeds cutoff.

        Parameters
        ----------
        n :
            Number of neighbors to be included into the LOF calculation.
            Defaults to ``20``, which is a
            value found to be suitable in the literature.

            * :py:attr:`n` determines the "locality" of an observation
              (its :py:attr:`n` nearest neighbors) and sets the upper
              limit to the number of values in outlier clusters (i.e.
              consecutive outliers). Outlier clusters of size greater
              than :py:attr:`n`/2 may not be detected reliably.
            * The larger :py:attr:`n`, the lesser the algorithm's sensitivity
              to local outliers and small or singleton outliers points.
              Higher values greatly increase numerical costs.

        thresh :
            The threshold for flagging the calculated LOF. A LOF of around
            ``1`` is considered normal and most likely corresponds to
            inlier points.

            * The "automatic" threshing introduced with the publication
              of the algorithm defaults to ``1.5``.
            * In this implementation, :py:attr:`thresh` defaults (``'auto'``)
              to flagging the scores with a modified 3-sigma rule.

        algorithm :
            Algorithm used for calculating the :py:attr:`n`-nearest neighbors.

        p :
            Degree of the metric ("Minkowski"), according to which the
            distance to neighbors is determined. Most important values are:

            * ``1`` - Manhattan Metric
            * ``2`` - Euclidian Metric

        density :
            How to calculate the temporal distance/density for the variable to flag.

            * ``'auto'`` - introduces linear density with an increment
              equal to the median of the absolute diff of the variable to flag.
            * ``float`` - introduces linear density with an increment
              equal to :py:attr:`density`
            * Callable - calculates the density by applying the function
              passed onto the variable to flag (passed as Series).

        Notes
        -----
        * The :py:meth:`~saqc.SaQC.flagLOF` function calculates the Local
          Outlier Factor (LOF) for every point in the input timeseries.
          The *LOF* is a scalar value, that roughly correlates to the
          *reachability*, or "outlierishnes" of the evaluated datapoint.
          If a point is as reachable, as all its :py:attr:`n`-nearest
          neighbors, the *LOF* score evaluates to around ``1``. If it
          is only as half as reachable as all its ``n``-nearest neighbors
          are (so to say, as double as "outlierish"), the score is about
          ``2``. So, the Local Outlier *Factor* relates a point's *reachability*
          to the *reachability* of its :py:attr:`n`-nearest neighbors
          in a multiplicative fashion (as a "factor").
        * The *reachability* of a point thereby is determined as an aggregation
          of the points distances to its :py:attr:`n`-nearest neighbors,
          measured with regard to the minkowski metric of degree :py:attr:`p`
          (usually euclidean).
        * To derive a binary label for every point (outlier: *yes*, or *no*),
          the scores are cut off at a level, determined by :py:attr:`thresh`.

        """
        self._validateLOF(algorithm, n, p, 1.0)
        if thresh != "auto" and not isFloatLike(thresh):
            raise ValueError(f"'thresh' must be 'auto' or a float, not {thresh}")

        fields = toSequence(field)
        field_ = str(uuid.uuid4())
        qc = self.assignLOF(
            field=fields,
            target=field_,
            n=n,
            algorithm=algorithm,
            p=p,
        )
        s = qc.data[field_]
        if thresh == "auto":
            s = pd.concat([s, (-s - 2)])
            s_mask = (s - s.mean() / s.std())[: len(s) // 2].abs() > 3
        else:
            s_mask = s < abs(thresh)

        for f in fields:
            mask = ~isflagged(qc._flags[f], kwargs["dfilter"]) & s_mask
            qc._flags[mask, f] = flag

        return qc.dropField(field_)

    @flagging()
    def flagUniLOF(
        self: "SaQC",
        field: str,
        n: int = 20,
        thresh: Literal["auto"] | float = 1.5,
        algorithm: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
        p: int = 1,
        density: Literal["auto"] | float = "auto",
        fill_na: bool = True,
        slope_correct: bool = True,
        min_offset: float = None,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag "univariate" Local Outlier Factor (LOF) exceeding cutoff.

        The function is a wrapper around a usual LOF implementation, aiming
        for an easy to use, parameter minimal outlier detection function
        for single variables, that does not necessitate prior modelling
        of the variable. LOF is applied onto a concatenation of the `field`
        variable and a "temporal density", or "penalty" variable, that
        measures temporal distance between data points. See notes Section
        for a more exhaustive explaination. See the Notes section for
        more details on the algorithm.

        Parameters
        ----------
        n :
            Number of periods to be included into the LOF calculation.
            Defaults to `20`, which is a value found to be suitable in
            the literature.

            * :py:attr:`n` determines the "locality" of an observation
              (its :py:attr:`n` nearest neighbors) and sets the upper
              limit to the number of values in an outlier clusters (i.e.
              consecutive outliers). Outlier clusters of size greater
              than :py:attr:`n`/2 may not be detected reliably.
            * The larger :py:attr:`n`, the lesser the algorithm's sensitivity
              to local outliers and small or singleton outlier points.
              Higher values greatly increase numerical costs.

        thresh :
            The threshold for flagging the calculated LOF. A LOF of around
            ``1`` is considered normal and most likely corresponds to
            inlier points. This parameter is considered the main calibration
            parameter of the algorithm.

            * The threshing defaults to ``1.5``, wich is the default value
              found to be suitable in the literature.
            * ``'auto'`` enables flagging the scores with a modified 3-sigma
              rule, resulting in a thresh around ``4``, which usually
              greatly mitigates overflagging compared to the literature
              recommendation, but often is too high.
            * sensitive range for the parameter may be ``[1,15]``, assuming
              default settings for the other parameters.

        algorithm :
            Algorithm used for calculating the :py:attr:`n`-nearest neighbors
            needed for LOF calculation.

        p :
            Degree of the metric ("Minkowski"), according to which distance
            to neighbors is determined. Most important values are:

            * ``1`` - Manhatten Metric
            * ``2`` - Euclidian Metric

        density :
            How to calculate the temporal distance/density for the variable
            to flag.

            * ``'auto'`` - introduces linear density with an increment
              equal to the median of the absolute diff of the variable to flag.
            * ``float`` - introduces linear density with an increment
              equal to :py:attr:`density`

        fill_na :
            If True, NaNs in the data are filled with a linear interpolation.

        slope_correct :
            if True, a correction is applied, that removes outlier cluster that actually
            just seem to be steep slopes

        min_offset :
            If set, only those outlier cluster will be flagged, that are preceeded and succeeeded
            by sufficiently large value "jumps". Defaults to estimating the sufficient value jumps from
            the median over the absolute step sizes between data points.

        Notes
        -----

        * The :py:meth:`~saqc.SaQC.flagUniLOF` function calculates an
          univariate Local Outlier Factor (UniLOF) - score for every
          point in the one dimensional input data series. The *UniLOF*
          score of any data point is a scalar value, that roughly correlates
          to its *reachability*, or "outlierishnes" in the 2-dimensional
          space constituted by the data-values and the time axis. So
          the Algorithm basically operates on the "graph", or the "plot"
          of the input timeseries.

        * If a point in this "graph" is as reachable, as all its :py:attr:`n`-nearest
          neighbors, its *UniLOF* score evaluates to around ``1``. If
          it is only as half as reachable as all its :py:attr:`n` neighbors
          are (so to say, as double as "outlierish"), its score evaluates
          to ``2`` roughly. So, the Univariate Local Outlier *Factor*
          relates a points *reachability* to the *reachability* of its
          :py:attr:`n`-nearest neighbors in a multiplicative fashion
          (as a "factor").

        * The *reachability* of a point thereby is derived as an aggregation
          of the points distance to its :py:attr:`n`-nearest neighbors,
          measured with regard to the minkowski metric of degree :py:attr:`p`
          (usually euclidean).

        * The parameter :py:attr:`density` thereby determines how dimensionality
          of the time is removed, to make it a dimensionless, real valued
          coordinate.

        * To derive a binary label for every point (outlier: *yes*, or
          *no*), the scores are cut off at a level, determined by :py:attr:`thresh`.

        Examples
        --------

        See the :ref:`outlier detection cookbook
        <cookbooks/OutlierDetection:Outlier Detection>` for a detailed
        introduction into the usage and tuning of the function.

        .. plot::
           :context: reset
           :include-source: False

           import matplotlib
           import saqc
           import pandas as pd
           data = pd.read_csv('../resources/data/hydro_data.csv')
           data = data.set_index('Timestamp')
           data.index = pd.DatetimeIndex(data.index)
           qc = saqc.SaQC(data)

        Example usage with default parameter configuration:

        Loading data via pandas csv file parser, casting index to DateTime,
        generating a :py:class:`~saqc.SaQC` instance from the data and
        plotting the variable representing light scattering at 254 nanometers
        wavelength.

        .. doctest:: flagUniLOFExample

           >>> import saqc
           >>> data = pd.read_csv('./resources/data/hydro_data.csv')
           >>> data = data.set_index('Timestamp')
           >>> data.index = pd.DatetimeIndex(data.index)
           >>> qc = saqc.SaQC(data)
           >>> qc.plot('sac254_raw') # doctest: +SKIP

        .. plot::
           :context:
           :include-source: False
           :class: center

           qc.plot('sac254_raw')

        We apply :py:meth:`~saqc.SaqC.flagUniLOF` in with default parameter
        values. Meaning, that the main calibration paramters :py:attr:`n`
        and :py:attr:`thresh` evaluate to `20` and `1.5` respectively.

        .. doctest:: flagUniLOFExample

           >>> import saqc
           >>> qc = qc.flagUniLOF('sac254_raw')
           >>> qc.plot('sac254_raw') # doctest: +SKIP

        .. plot::
           :context: close-figs
           :include-source: False
           :class: center

           qc = qc.flagUniLOF('sac254_raw')
           qc.plot('sac254_raw')

        See also
        --------
        :ref:`introduction to outlier detection with saqc <cookbooks/OutlierDetection:Outlier Detection>`
        """
        self._validateLOF(algorithm, n, p, density)
        if thresh != "auto" and not isFloatLike(thresh):
            raise ValueError(f"'thresh' must be 'auto' or a float, not {thresh}")

        tmp_field = str(uuid.uuid4())
        qc = self.assignUniLOF(
            field=field,
            target=tmp_field,
            n=n,
            algorithm=algorithm,
            p=p,
            density=density,
            fill_na=fill_na,
        )
        s = qc.data[tmp_field]
        if thresh == "auto":
            _s = pd.concat([s, (-s - 2)])
            s_mask = ((_s - _s.mean()) / _s.std()).iloc[: int(s.shape[0])].abs() > 3
        else:
            s_mask = s < -abs(thresh)
        s_mask = ~isflagged(qc._flags[field], kwargs["dfilter"]) & s_mask

        if slope_correct:
            g_mask = s_mask.diff()
            g_mask = g_mask.cumsum()
            dat = self._data[field]
            od_groups = dat.interpolate("linear").groupby(by=g_mask)
            first_vals = od_groups.first()
            last_vals = od_groups.last()
            max_vals = od_groups.max()
            min_vals = od_groups.min()
            if min_offset is None:
                if density == "auto":
                    d_diff = dat.diff()
                    eps = d_diff.abs().median()
                    if eps == 0:
                        eps = d_diff[d_diff != 0].abs().median()
                else:
                    eps = density
                eps = 3 * eps
            else:
                eps = min_offset
            up_slopes = (min_vals + eps >= last_vals.shift(1)) & (
                max_vals - eps <= first_vals.shift(-1)
            )
            down_slopes = (max_vals - eps <= last_vals.shift(1)) & (
                min_vals + eps >= first_vals.shift(-1)
            )
            corrections = up_slopes | down_slopes
            for s_id in corrections[corrections].index:
                correct_idx = od_groups.get_group(s_id).index
                s_mask[correct_idx] = False

        qc._flags[s_mask, field] = flag
        qc = qc.dropField(tmp_field)
        return qc

    @flagging()
    def flagRange(
        self: "SaQC",
        field: str,
        min: float = -np.inf,
        max: float = np.inf,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Function flags values exceeding the closed
        interval [:py:attr:`min`, :py:attr:`max`].

        Parameters
        ----------
        min :
            Lower bound for valid data.
        max :
            Upper bound for valid data.
        """
        # using .values is much faster
        datacol = self._data[field].to_numpy()
        mask = (datacol < min) | (datacol > max)
        self._flags[mask, field] = flag
        return self

    @flagging()
    def flagByStray(
        self: "SaQC",
        field: str,
        window: int | str | None = None,
        min_periods: int = 11,
        iter_start: float = 0.5,
        alpha: float = 0.05,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag outliers in 1-dimensional (score) data using the STRAY Algorithm.

        For more details about the algorithm please refer to [1].

        Parameters
        ----------

        window :
            Determines the segmentation of the data into partitions, the
            kNN algorithm is applied onto individually.

            * ``None``: Apply Scoring on whole data set at once
            * ``int``: Apply scoring on successive data chunks of periods
              with the given length. Must be greater than 0.
            * offset String : Apply scoring on successive partitions of
              temporal extension matching the passed offset string

        min_periods :
            Minimum number of periods per partition that have to be present
            for a valid outlier detection to be made in this partition

        iter_start :
            Float in ``[0, 1]`` that determines which percentage of data
            is considered "normal". ``0.5`` results in the stray algorithm
            to search only the upper 50% of the scores for the cut off
            point. (See reference section for more information)

        alpha :
            Level of significance by which it is tested, if a score might
            be drawn from another distribution than the majority of the data.

        References
        ----------
        [1]  Priyanga Dilini Talagala, Rob J. Hyndman & Kate Smith-Miles (2021):
             Anomaly Detection in High-Dimensional Data,
             Journal of Computational and Graphical Statistics, 30:2, 360-374,
             DOI: 10.1080/10618600.2020.1807997
        """
        scores = self._data[field].dropna()

        if window is None:
            window = len(scores)
        if not isinstance(window, int):
            validateFrequency(window, "window")

        validateMinPeriods(min_periods)
        validateValueBounds(iter_start, "iter_start", left=0, right=1, closed="both")

        if scores.empty:
            return self

        if isinstance(window, int):
            s = pd.Series(data=np.arange(0, len(scores)), index=scores.index)
            s = s.transform(lambda x: int(np.floor(x / window)))
            partitions = scores.groupby(s)
        else:  # pd.Timedelta pd.DateOffset or str
            partitions = scores.groupby(pd.Grouper(freq=window))

        # calculate flags for every window
        for _, partition in partitions:
            sample_size = len(partition)

            if partition.empty or sample_size < min_periods:
                continue

            sorted_i = partition.values.argsort()
            resids = partition.values[sorted_i]
            gaps = np.append(0, np.diff(resids))

            tail_size = int(max(min(np.floor(sample_size / 4), 50), 2))
            tail_indices = np.arange(2, tail_size + 1)

            i_start = int(max(np.floor(sample_size * iter_start), 1) + 1)
            ghat = np.array([np.nan] * sample_size)

            for i in range(i_start - 1, sample_size):
                ghat[i] = sum(
                    (tail_indices / (tail_size - 1)) * gaps[i - tail_indices + 1]
                )

            log_alpha = np.log(1 / alpha)
            for iter_index in range(i_start - 1, sample_size):
                if gaps[iter_index] > log_alpha * ghat[iter_index]:
                    index = partition.index[sorted_i[iter_index:]]
                    self._flags[index, field] = flag
                    break

        return self

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        multivariate=True,
        handles_target=False,
        docstring={"field": DOC_TEMPLATES["field"]},
    )
    def flagMVScores(
        self: "SaQC",
        field: Sequence[str],
        trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
        alpha: float = 0.05,
        n: int = 10,
        func: Callable[[pd.Series], float] | str = "sum",
        iter_start: float = 0.5,
        window: int | str | None = None,
        min_periods: int = 11,
        stray_range: str | None = None,
        drop_flagged: bool = False,  # TODO: still a case ?
        thresh: float = 3.5,
        min_periods_r: int = 1,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        The algorithm implements a 3-step outlier detection procedure for
        simultaneously flagging of higher dimensional data (dimensions > 3).

        In [1], the procedure is introduced and exemplified with an application on
        hydrological data. See the notes section for an overview over the algorithms
        basic steps.

            .. deprecated:: 2.6.0
               Deprecated Function. Please refer to :py:meth:`~saqc.SaQC.flagByStray`.

        Parameters
        ----------
        trafo :
            Transformation to be applied onto every column before scoring. For more
            fine-grained control, the data could also be transformed before
            :py:meth:`~saqc.SaQC.flagMVScores` is called.

        alpha :
            Level of significance by which it is tested, if an observations score might
            be drawn from another distribution than the majority of the data.

        n :
            Number of neighbors included in the scoring process for every datapoint.

        func :
            Function that aggregates a value's k-smallest distances, returning a
            scalar score.

        iter_start :
            Value in ``[0,1]`` that determines which percentage of data is considered
            "normal". 0.5 results in the threshing algorithm to search only the upper
            50% of the scores for the cut-off point. (See reference section for more
            information)

        window :
            Only effective if :py:attr:`threshing` is set to ``'stray'``. Determines
            the size of the data partitions, the data is decomposed into. Each
            partition is checked seperately for outliers. Either given as an Offset
            String, denoting the windows temporal extension or as an integer,
            denoting the windows number of periods. ``NaN`` also count as periods. If
            ``None``, all data points share the same scoring window, which than
            equals the whole data.

        min_periods :
            Only effective if :py:attr:`threshing` is set to ``'stray'`` and
            :py:attr:`partition` is an integer. Minimum number of periods per
            :py:attr:`partition` that have to be present for a valid outlier
            detection to be made in this partition.

        stray_range :
            If not ``None``, it is tried to reduce the stray result onto single
            outlier components of the input :py:attr:`field`. The offset string
            denotes the range of the temporal surrounding to include into the MAD
            testing while trying to reduce flags.

        drop_flagged :
            Only effective when :py:attr:`stray_range` is not ``None``. Whether or
            not to drop flagged values from the temporal surroundings.

        thresh :
            Only effective when :py:attr:`stray_range` is not ``None``. The
            'critical' value, controlling wheather the MAD score is considered
            referring to an outlier or not. Higher values result in less rigid
            flagging. The default value is widely considered apropriate in the
            literature.

        min_periods_r :
            Only effective when :py:attr:`stray_range` is not ``None``. Minimum
            number of measurements necessary in an interval to actually perform the
            reduction step.

        Notes
        -----
        The basic steps are:

        1. transforming

        The different data columns are transformed via timeseries transformations to
        (a) make them comparable and
        (b) make outliers more stand out.

        This step is usually subject to a phase of research/try and error. See [1]
        for more details.

        Note, that the data transformation as a built-in step of the algorithm,
        will likely get deprecated in the future. It's better to transform the data in
        a processing step, preceeding the multivariate flagging process. Also,
        by doing so, one gets mutch more control and variety in the transformation
        applied, since the `trafo` parameter only allows for application of the same
        transformation to all the variables involved.

        2. scoring

        Every observation gets assigned a score depending on its k nearest neighbors.
        See the `scoring_method` parameter description for details on the different
        scoring methods. Furthermore, [1] may give some insight in the pro and cons of
        the different methods.

        3. threshing

        The gaps between the (greatest) scores are tested for beeing drawn from the
        same distribution as the majority of the scores. If a gap is encountered,
        that, with sufficient significance, can be said to not be drawn from the same
        distribution as the one all the smaller gaps are drawn from, than the
        observation belonging to this gap, and all the observations belonging to gaps
        larger than this gap, get flagged outliers. See description of the
        `threshing` parameter for more details. Although [1] gives a fully detailed
        overview over the `stray` algorithm.

        References
        ----------
        [1]  Priyanga Dilini Talagala, Rob J. Hyndman & Kate Smith-Miles (2021):
             Anomaly Detection in High-Dimensional Data,
             Journal of Computational and Graphical Statistics, 30:2, 360-374,
             DOI: 10.1080/10618600.2020.1807997
        """

        # parameter deprecations

        if "partition" in kwargs:
            warnings.warn(
                """
                The parameter `partition` is deprecated and will be removed in version
                2.7 of saqc. Please us the parameter `window` instead.
                """,
                DeprecationWarning,
            )
            window = kwargs["partition"]

        if "partition_min" in kwargs:
            warnings.warn(
                """
                The parameter `partition_min` is deprecated and will be removed in
                version 2.7 of saqc. Please us the parameter `min_periods` instead.
                """,
                DeprecationWarning,
            )
            min_periods = kwargs["partition_min"]

        if min_periods != 11:
            warnings.warn(
                """
                You were setting a customary value for the `min_periods` parameter:
                note that this parameter does no longer refer to the reduction interval
                length, but now controls the number of periods having to be present in
                an interval of size `window` (deprecated:`partition`) for the algorithm
                to be performed in that interval.
                To alter the size of the reduction window, use the parameter
                `min_periods_r`. Changes readily apply.
                This warning will be removed in saqc version 2.7.
                """,
                DeprecationWarning,
            )

        warnings.warn(
            """
                flagMVScores is deprecated and will be removed with Version 2.8.
                To replicate the function, transform the different fields involved
                via explicit applications of some transformations, than calculate the
                kNN scores via `saqc.SaQC.assignkNScores` and finally assign the STRAY
                algorithm via `saqc.SaQC.flagByStray`.
                """,
            DeprecationWarning,
        )

        # Hint: checking is delegated to the called functions

        fields = toSequence(field)

        qc = self
        fields_ = []
        for f in fields:
            field_ = str(uuid.uuid4())
            qc = qc.copyField(field=f, target=field_)
            qc = qc.transform(field=field_, func=trafo, freq=window)
            fields_.append(field_)

        knn_field = str(uuid.uuid4())
        qc = qc.assignKNNScore(
            field=fields_,
            target=knn_field,
            n=n,
            func=func,
            freq=window,
            algorithm="ball_tree",
            min_periods=min_periods,
            **kwargs,
        )
        for field_ in fields_:
            qc = qc.dropField(field_)

        qc = qc.flagByStray(
            field=knn_field,
            freq=window,
            min_periods=min_periods,
            iter_start=iter_start,
            alpha=alpha,
            flag=flag,
            **kwargs,
        )

        qc._data, qc._flags = _evalStrayLabels(
            data=qc._data,
            field=knn_field,
            target=fields,
            flags=qc._flags,
            reduction_range=stray_range,
            reduction_drop_flagged=drop_flagged,
            reduction_thresh=thresh,
            reduction_min_periods=min_periods_r,
            flag=flag,
            **kwargs,
        )
        return qc.dropField(knn_field)

    @flagging()
    def flagRaise(
        self: "SaQC",
        field: str,
        thresh: float,
        raise_window: str,
        freq: str,
        average_window: str | None = None,
        raise_factor: float = 2.0,
        slope: float | None = None,
        weight: float = 0.8,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        The function flags raises and drops in value courses, that exceed a certain threshold within a certain timespan.

        .. deprecated:: 2.6.0
           Function is deprecated since its not humanly parameterisable. Also more suitable alternatives are available. Depending on use case, use: :py:meth:`~saqc.SaQC.flagUniLOF`, :py:meth:`~saqc.SaQC.flagZScore`, :py:meth:`~saqc.SaQC.flagJumps` instead.

        Parameters
        ----------
        thresh :
            The threshold, for the total rise (:py:attr:`thresh` ``> 0``),
            or total drop (:py:attr:`thresh` ``< 0``), value courses must
            not exceed within a timespan of length :py:attr:`raise_window`.

        raise_window :
            An offset string, determining the timespan, the rise/drop
            thresholding refers to. Window is inclusively defined.

        freq :
            An offset string, determining the frequency, the timeseries
            to flag is supposed to be sampled at. The window is inclusively
            defined.

        average_window :
            See condition (2) of the description given in the Notes. Window
            is inclusively defined, defaults to 1.5 times the size of
            :py:attr:`raise_window`.

        raise_factor :
            See condition (2).

        slope :
            See condition (3).

        weight :
            See condition (3).

        Notes
        -----
        The dataset is NOT supposed to be harmonized to a time series with an
        equidistant requency grid.

        The value :math:`x_{k}` of a time series :math:`x` with associated
        timestamps :math:`t_i`, is flagged a raise, if:

        1. There is any value :math:`x_{s}`, preceeding :math:`x_{k}` within
           :py:attr:`raise_window` range, so that
           :math:`M = |x_k - x_s | >`  :py:attr:`thresh` :math:`> 0`

        2. The weighted average :math:`\\mu^{*}` of the values, preceding
           :math:`x_{k}` within :py:attr:`average_window` range indicates,
           that :math:`x_{k}` does not return from an "outlierish" value
           course, meaning that
           :math:`x_k > \\mu^* + ( M` / :py:attr:`raise_factor` :math:`)`

        3. Additionally, if :py:attr:`slope` is not ``None``, :math:`x_{k}`
           is checked or being sufficiently divergent from its very predecessor
           :math:`x_{k-1}`, meaning that, it is additionally checked if:
           * :math:`x_k - x_{k-1} >` :py:attr:`slope`
           * :math:`t_k - t_{k-1} >` :py:attr:`weight` :math:`\\times` :py:attr:`freq`
        """

        warnings.warn(
            "The function flagRaise is deprecated with no 100% exact replacement function."
            "When looking for changes in the value course, the use of flagRaise can be replicated and more "
            "easily aimed for, via the method flagJump.\n"
            "When looking for raises to outliers or plateaus, use one of: "
            "flagZScore (outliers), flagUniLOF (outliers and small plateaus) or flagOffset (plateaus)",
            DeprecationWarning,
        )

        validateWindow(raise_window, "raise_window", allow_int=False)
        validateWindow(freq, "freq", allow_int=False)
        validateWindow(average_window, "average_window", allow_int=False, optional=True)

        # prepare input args
        dataseries = self._data[field].dropna()
        raise_window_td = pd.Timedelta(raise_window)
        freq_dt = pd.Timedelta(freq)
        if slope is not None:
            slope = np.abs(slope)

        if average_window is None:
            average_window = 1.5 * raise_window_td

        if thresh < 0:
            dataseries *= -1
            thresh *= -1

        def raise_check(x, thresh):
            test_set = x[-1] - x[0:-1]
            max_val = np.max(test_set)
            if max_val >= thresh:
                return max_val
            else:
                return np.nan

        def custom_rolling_mean(x):
            return np.sum(x[:-1])

        # get invalid-raise/drop mask:
        raise_series = dataseries.rolling(raise_window_td, min_periods=2, closed="both")

        raise_series = raise_series.apply(raise_check, args=(thresh,), raw=True)

        if raise_series.isna().all():
            return self

        # "unflag" values of insufficient deviation to their predecessors
        if slope is not None:
            w_mask = (
                pd.Series(dataseries.index).diff().dt.total_seconds()
                / freq_dt.total_seconds()
            ) > weight
            slope_mask = np.abs(dataseries.diff()) < slope
            to_unflag = raise_series.notna() & w_mask.values & slope_mask
            raise_series[to_unflag] = np.nan

        # calculate and apply the weighted mean weights (pseudo-harmonization):
        weights = (
            pd.Series(dataseries.index).diff(periods=2).shift(-1).dt.total_seconds()
            / freq_dt.total_seconds()
            / 2
        )

        weights.iloc[0] = 0.5 + (
            pd.Timestamp(dataseries.index[1]) - pd.Timestamp(dataseries.index[0])
        ).total_seconds() / (freq_dt.total_seconds() * 2)

        weights.iloc[-1] = 0.5 + (
            pd.Timestamp(dataseries.index[-1]) - pd.Timestamp(dataseries.index[-2])
        ).total_seconds() / (freq_dt.total_seconds() * 2)

        weights[weights > 1.5] = 1.5
        weights.index = dataseries.index
        weighted_data = dataseries.mul(weights)

        # rolling weighted mean calculation
        weighted_rolling_mean = weighted_data.rolling(
            average_window, min_periods=2, closed="both"
        )
        weights_rolling_sum = weights.rolling(
            average_window, min_periods=2, closed="both"
        )
        weighted_rolling_mean = weighted_rolling_mean.apply(
            custom_rolling_mean, raw=True
        )
        weights_rolling_sum = weights_rolling_sum.apply(custom_rolling_mean, raw=True)

        weighted_rolling_mean = weighted_rolling_mean / weights_rolling_sum
        # check means against critical raise value:
        to_flag = dataseries >= weighted_rolling_mean + (raise_series / raise_factor)
        to_flag &= raise_series.notna()
        self._flags[to_flag[to_flag].index, field] = flag

        return self

    @flagging()
    def flagMAD(
        self: "SaQC",
        field: str,
        window: str | int | None = None,
        z: float = 3.5,
        min_residuals: int | None = None,
        min_periods: int | None = None,
        center: bool = False,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag outiers using the modified Z-score outlier detection method.

        See references [1] for more details on the algorithm.

            .. deprecated:: 2.6.0
               Deprecated Function. Please refer to :py:meth:`~saqc.SaQC.flagZScore`.


        Note
        ----
        Data needs to be sampled at a regular equidistant time grid.

        Parameters
        ----------
        window :
            Size of the window. Either given as an Offset String, denoting the window's temporal extension or
            as an integer, denoting the window's number of periods. ``NaN`` also count as periods.
            If ``None``, all data points share the same scoring window, which than equals the whole data.
        z :
            The value the Z-score is tested against. Defaulting to ``3.5`` (Recommendation of [1])
        min_periods :
            Minimum number of valid meassurements in a scoring window, to consider the resulting score valid.
        center :
            Weather or not to center the target value in the scoring window. If ``False``, the
            target value is the last value in the window.

        References
        ----------
        [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        """
        warnings.warn(
            f"The method `flagMAD` is deprecated and will be removed in "
            "version 2.7 of saqc. To achieve the same behavior use:"
            f"`qc.flagZScore(field={field}, window={window}, method='modified', "
            f"thresh={z}, min_residuals={min_residuals}, min_periods={min_periods}, "
            f"center={center})`",
            DeprecationWarning,
        )
        return self.flagZScore(
            field,
            window=window,
            thresh=z,
            min_residuals=min_residuals,
            model_func=np.median,
            norm_func=lambda x: median_abs_deviation(
                x, scale="normal", nan_policy="omit"
            ),
            center=center,
            min_periods=min_periods,
            flag=flag,
        )

    @flagging()
    def flagOffset(
        self: "SaQC",
        field: str,
        tolerance: float,
        window: int | str,
        thresh: float | None = None,
        thresh_relative: float | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        A basic outlier test that works on regularly and irregularly sampled data.

        The test classifies values/value courses as outliers by detecting not only a rise
        in value, but also, by checking for a return to the initial value level.

        Notes
        -----
        This definition of a "spike" not only includes one-value outliers, but also plateau-ish value courses.

        Values :math:`x_n, x_{n+1}, .... , x_{n+k}` of a timeseries :math:`x` with
        associated timestamps :math:`t_n, t_{n+1}, .... , t_{n+k}` are considered spikes, if:

        1. :math:`|x_{n-1} - x_{n + s}| >` :py:attr:`thresh`, for all :math:`s \\in [0,1,2,...,k]`
        2. if :py:attr:`thresh_relative` > 0, :math:`x_{n + s} > x_{n - 1}*(1+` :py:attr:`thresh_relative` :math:`)`
        3. if :py:attr:`thresh_relative` < 0, :math:`x_{n + s} < x_{n - 1}*(1+` :py:attr:`thresh_relative` :math:`)`
        4. :math:`|x_{n-1} - x_{n+k+1}| <` :py:attr:`tolerance`
        5. :math:`|t_{n-1} - t_{n+k+1}| <` :py:attr:`window`


        Parameters
        ----------
        tolerance :
            Maximum difference allowed between the value, directly preceding and the value directly
            succeeding an offset to trigger flagging of the offsetting values. See condition (4).
        window :
            Maximum length allowed for offset value courses, to trigger flagging of the offsetting values.
            See condition (5). Integer defined window length are only allowed for regularly sampled
            timeseries.
        thresh :
            Minimum difference between a value and its successors, to consider the successors an anomalous
            offset group. See condition (1). If ``None``, condition (1) is not tested.
        thresh_relative :
            Minimum relative change between a value and its successors, to consider the successors an anomalous
            offset group. See condition (2). If ``None``, condition (2) is not tested.

        Examples
        --------
        Below picture gives an abstract interpretation of the parameter interplay in case of a positive
        value jump, initialising an offset course.

        .. figure:: /resources/images/flagOffsetPic.png

           The four values marked red, are flagged, because (1) the initial value jump *exceeds* the value
           given by :py:attr:`thresh`, (2) the temporal extension of the group does *not exceed* the range
           given by `window` and (3) the returning value after the group, lies *within* the value range
           determined by :py:attr:`tolerance`


        .. plot::
           :context:
           :include-source: False

           import matplotlib
           import saqc
           import pandas as pd
           data = pd.DataFrame({'data':np.array([5,5,8,16,17,7,4,4,4,1,1,4])}, index=pd.date_range('2000',freq='1h', periods=12))


        Lets generate a simple, regularly sampled timeseries with an hourly sampling rate and generate an
        :py:class:`saqc.SaQC` instance from it.

        .. doctest:: flagOffsetExample

           >>> import saqc
           >>> data = pd.DataFrame({'data':np.array([5,5,8,16,17,7,4,4,4,1,1,4])}, index=pd.date_range('2000',freq='1h', periods=12))
           >>> data
                                data
           2000-01-01 00:00:00     5
           2000-01-01 01:00:00     5
           2000-01-01 02:00:00     8
           2000-01-01 03:00:00    16
           2000-01-01 04:00:00    17
           2000-01-01 05:00:00     7
           2000-01-01 06:00:00     4
           2000-01-01 07:00:00     4
           2000-01-01 08:00:00     4
           2000-01-01 09:00:00     1
           2000-01-01 10:00:00     1
           2000-01-01 11:00:00     4
           >>> qc = saqc.SaQC(data)

        Now we are applying :py:meth:`~saqc.SaQC.flagOffset` and try to flag offset courses, that dont extend
        longer than *6 hours* in time (:py:attr:`window`) and that have an initial value jump higher than ``2``
        (:py:attr:`thresh`), and that do return to the initial value level within a tolerance of ``1.5``
        (:py:attr:`tolerance`).

        .. doctest:: flagOffsetExample

           >>> qc = qc.flagOffset("data", thresh=2, tolerance=1.5, window='6h')
           >>> qc.plot('data')  # doctest: +SKIP

        .. plot::
           :context: close-figs
           :include-source: False

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagOffset("data", thresh=2, tolerance=1.5, window='6h')
           >>> qc.plot('data')  # doctest: +SKIP

        Note, that both, negative and positive jumps are considered starting points of negative or positive
        offsets. If you want to impose the additional condition, that the initial jump must exceed
        +90%* of the value level, you can additionally set the :py:attr:`thresh_relative` parameter:

        .. doctest:: flagOffsetExample

           >>> qc = qc.flagOffset("data", thresh=2, thresh_relative=.9, tolerance=1.5, window='6h')
           >>> qc.plot('data') # doctest:+SKIP

        .. plot::
           :context: close-figs
           :include-source: False

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagOffset("data", thresh=2, thresh_relative=.9, tolerance=1.5, window='6h')
           >>> qc.plot('data')  # doctest: +SKIP

        Now, only positive jumps, that exceed a value gain of +90%* are considered starting points of offsets.

        In the same way, you can aim for only negative offsets, by setting a negative relative threshold.
        The below example only flags offsets, that fall off by at least *50%* in value, with an absolute
        value drop of at least 2.

        .. doctest:: flagOffsetExample

           >>> qc = qc.flagOffset("data", thresh=2, thresh_relative=-.5, tolerance=1.5, window='6h')
           >>> qc.plot('data') # doctest:+SKIP

        .. plot::
           :context: close-figs
           :include-source: False

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagOffset("data", thresh=2, thresh_relative=-.5, tolerance=1.5, window='6h')
           >>> qc.plot('data')  # doctest: +SKIP
        """
        validateWindow(window)
        if thresh is None and thresh_relative is None:
            raise ValueError(
                "At least one of parameters 'thresh' and 'thresh_relative' "
                "has to be given. Got 'thresh'=None, 'thresh_relative'=None "
                "instead."
            )
        if thresh is None:
            thresh = 0

        dat = self._data[field].dropna()
        if thresh_relative is not None:
            rel_jumps = np.sign(thresh_relative) * dat > np.sign(
                thresh_relative
            ) * dat.shift(+1) * (1 + thresh_relative)

        data_diff = dat.diff()
        initial_jumps = data_diff.abs() > thresh
        if thresh_relative:
            initial_jumps &= rel_jumps
        return_in_time = (
            dat[::-1]
            .rolling(window, min_periods=2)
            .apply(lambda x: np.abs(x[-1] - x[:-1]).min() < tolerance, raw=True)[::-1]
            .astype(bool)
        )
        return_in_time = return_in_time & initial_jumps.reindex(
            dat.index, fill_value=False
        ).shift(-1, fill_value=False)
        offset_start_candidates = dat[return_in_time]
        win_delta = pd.Timedelta(window)
        corners = pd.Series(False, index=dat.index)
        to_flag = pd.Series(False, index=dat.index)
        ns = pd.Timedelta("1ns")
        for c in zip(offset_start_candidates.index, offset_start_candidates.values):
            ret = (dat[c[0]] - dat[c[0] + ns : c[0] + win_delta]).abs()[1:] < tolerance
            if not ret.empty:
                r = ret.idxmax()
                chunk = dat[c[0] : r]
                sgn = np.sign(chunk.iloc[1] - c[1])
                t_val = ((chunk[1:-1] - c[1]) * sgn > thresh).all()
                r_val = True
                if thresh_relative:
                    r_val = (
                        np.sign(thresh_relative) * chunk[1:-1]
                        > np.sign(thresh_relative) * c[1] * (1 + thresh_relative)
                    ).all()
                if t_val and r_val and (not corners[c[0]]):
                    flag_i = dat[c[0] + ns : chunk.index[-1] - ns].index
                    to_flag[flag_i] = True
                    corners.loc[flag_i[-1]] = True
        to_flag = to_flag.reindex(self._data[field].index, fill_value=False)

        self._flags[to_flag, field] = flag
        return self

    @flagging()
    def flagByGrubbs(
        self: "SaQC",
        field: str,
        window: str | int,
        alpha: float = 0.05,
        min_periods: int = 8,
        pedantic: bool = False,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag outliers using the Grubbs algorithm.

        .. deprecated:: 2.6.0
           Use :py:meth:`~saqc.SaQC.flagUniLOF` or :py:meth:`~saqc.SaQC.flagZScore` instead.

        Parameters
        ----------
        window :
            Size of the testing window.
            If an integer, the fixed number of observations used for each window.
            If an offset string the time period of each window.

        alpha :
            Level of significance, the grubbs test is to be performed at. Must be between 0 and 1.

        min_periods :
            Minimum number of values needed in a :py:attr:`window` in order to perform the grubs test.
            Ignored if :py:attr:`window` is an integer.

        pedantic :
            If ``True``, every value gets checked twice. First in the initial rolling :py:attr:`window`
            and second in a rolling window that is lagging by :py:attr:`window` / 2. Recommended to avoid
            false positives at the window edges. Ignored if :py:attr:`window` is an offset string.

        References
        ----------
        introduction to the grubbs test:

        [1] https://en.wikipedia.org/wiki/Grubbs%27s_test_for_outliers
        """

        warnings.warn(
            "The function flagByGrubbs is deprecated due to its inferior performance, with "
            "no 100% exact replacement function. When looking for outliers use one of: "
            "flagZScore, flagUniLOF",
            DeprecationWarning,
        )

        validateWindow(window)
        validateFraction(alpha, "alpha")
        validateMinPeriods(min_periods, optional=False)

        datcol = self._data[field].copy()
        rate = getFreqDelta(datcol.index)

        # if timeseries that is analyzed, is regular,
        # window size can be transformed to a number of periods:
        if rate and isinstance(window, str):
            window = pd.Timedelta(window) // rate

        to_group = pd.DataFrame(data={"ts": datcol.index, "data": datcol})
        to_flag = pd.Series(False, index=datcol.index)

        # period number defined test intervals
        if isinstance(window, int):
            grouper_series = pd.Series(
                data=np.arange(0, len(datcol)), index=datcol.index
            )
            grouper_series_lagged = grouper_series + (window / 2)
            grouper_series = grouper_series.transform(lambda x: x // window)
            grouper_series_lagged = grouper_series_lagged.transform(
                lambda x: x // window
            )
            partitions = to_group.groupby(grouper_series)
            partitions_lagged = to_group.groupby(grouper_series_lagged)

        # offset defined test intervals:
        else:
            partitions = to_group.groupby(pd.Grouper(freq=window))
            partitions_lagged = []

        for _, partition in partitions:
            if partition.shape[0] > min_periods:
                detected = smirnov_grubbs.two_sided_test_indices(
                    partition["data"].values, alpha=alpha
                )
                detected = partition["ts"].iloc[detected]
                to_flag[detected.index] = True

        if isinstance(window, int) and pedantic:
            to_flag_lagged = pd.Series(False, index=datcol.index)

            for _, partition in partitions_lagged:
                if partition.shape[0] > min_periods:
                    detected = smirnov_grubbs.two_sided_test_indices(
                        partition["data"].values, alpha=alpha
                    )
                    detected = partition["ts"].iloc[detected]
                    to_flag_lagged[detected.index] = True

            to_flag &= to_flag_lagged

        self._flags[to_flag, field] = flag
        return self

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        multivariate=True,
        docstring={"field": DOC_TEMPLATES["field"]},
    )
    def flagZScore(
        self: "SaQC",
        field: Sequence[str],
        method: Literal["standard", "modified"] = "standard",
        window: str | int | None = None,
        thresh: float = 3,
        min_residuals: int | None = None,
        min_periods: int | None = None,
        center: bool = True,
        axis: int = 0,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag data where its (rolling) Zscore exceeds a threshold.

        The function implements flagging derived from standard or modified Zscore calculation. To handle non
        stationary data, the Zscoring can be applied with a rolling window. Therefor, the function
        allows for a minimum residual to be specified in order to mitigate overflagging in local
        regimes of low variance.

        See the Notes section for a detailed overview of the calculation

        Parameters
        ----------
        window :
            Size of the window. Either determined via an offset string, denoting the windows temporal
            extension or by an integer, denoting the windows number of periods. ``NaN`` also count as
            periods. If ``None`` is passed, all data points share the same scoring window, which than
            equals the whole data.
        method :
            Which method to use for ZScoring:

            * `"standard"`: standard Zscoring, using *mean* for the expectation and *standard deviation (std)* as scaling factor
            * `"modified"`: modified Zscoring, using *median* as the expectation and *median absolute deviation (MAD)* as the scaling Factor

            See notes section for detailed scoring formula
        thresh :
            Cutoff level for the Zscores, above which associated points are marked as outliers.
        min_residuals :
            Minimum residual value points must have to be considered outliers.
        min_periods :
            Minimum number of valid meassurements in a scoring window, to consider the resulting score valid.
        center :
            Weather or not to center the target value in the scoring window. If ``False``, the
            target value is the last value in the window.
        axis :
            Along which axis to calculate the scoring statistics:

            * `0` (default) - calculate statistics along time axis
            * `1` - calculate statistics over multiple variables

            See Notes section for a visual clarification of the workings
            of `axis` and `window`.

        Notes
        -----

        The flag for :math:`x` is determined as follows:

        1. Depending on ``window`` and ``axis``, the context population :math:`X` is collected (see pictures below)

           * If ``axis=0``, any value is flagged in the context of those values of the same variable (``field``), that are
             in `window` range.
           * If ``axis=1``, any value is flagged in the context of all values of all variables (``fields``), that are
             in `window` range.
           * If ``axis=0`` and ``window=1``, any value is flagged in the context of all values of all variables (``fields``),
             that share the same timestamp.

        .. figure:: /resources/images/ZscorePopulation.png
           :class: with-border




        2. Depending on ``method``, a score :math:`Z` is calculated for :math:`x` via :math:`Z = \\frac{|E(X) - X|}{S(X)}`

           * ``method="standard"``: :math:`E(X)=mean(X)`, :math:`S(X)=std(X)`
           * ``method="modified"``: :math:`E(X)=median(X)`, :math:`S(X)=MAD(X)`

        3. :math:`x` is flagged, if :math:`Z >` ``thresh``
        """

        if "norm_func" in kwargs or "model_func" in kwargs:
            warnings.warn(
                "Parameters norm_func and model_func are deprecated, use parameter method instead.\n"
                'To model with mean and scale with standard deviation, use method="standard".\n'
                'To model with median and scale with median absolute deviation (MAD) use method="modified".\n'
                "Other/Custom model and scaling functions are not supported any more"
            )
            if (
                "mean" in kwargs.get("model_func", "").__name__
                or "std" in kwargs.get("norm_func", "").__name__
            ):
                method = "standard"
            elif (
                "median" in kwargs.get("model_func", lambda x: x).__name__
                or "median" in kwargs.get("norm_func", lambda x: x).__name__
            ):
                method = "modified"
            else:
                raise ValueError(
                    "Support for scoring with functions not similar to "
                    "either Zscore or modified Zscore is not supported "
                    "anymore"
                )

        validateChoice(method, "method", ["standard", "modified"])
        validateWindow(window, optional=True)
        validateMinPeriods(min_periods)

        min_residuals = min_residuals or 0
        min_periods = min_periods or 0

        dat = self._data[field].to_pandas(how="outer")
        if dat.empty:
            return self

        if window is None:
            if dat.notna().sum().sum() >= min_periods:
                if method == "standard":
                    mod = pd.DataFrame(
                        {f: dat[f].mean() for f in dat.columns}, index=dat.index
                    )
                    norm = pd.DataFrame(
                        {f: dat[f].std() for f in dat.columns}, index=dat.index
                    )

                else:
                    mod = pd.DataFrame(
                        {f: dat[f].median() for f in dat.columns}, index=dat.index
                    )
                    norm = pd.DataFrame(
                        {f: (dat[f] - mod[f]).abs().median() for f in dat.columns},
                        index=dat.index,
                    )
            else:
                return self

        else:  # window is not None
            if axis == 0:
                if method == "standard":
                    mod = dat.rolling(
                        window, center=center, min_periods=min_periods
                    ).mean()
                    norm = dat.rolling(
                        window, center=center, min_periods=min_periods
                    ).std()
                else:
                    mod = dat.rolling(
                        window, center=center, min_periods=min_periods
                    ).median()
                    norm = (
                        (mod - dat)
                        .abs()
                        .rolling(window, center=center, min_periods=min_periods)
                        .median()
                    )

            else:  # axis == 1:
                if window == 1:
                    if method == "standard":
                        mod = dat.mean(axis=1)
                        norm = dat.std(axis=1)
                    else:  # method == 'modified'
                        mod = dat.median(axis=1)
                        norm = (dat.subtract(mod, axis=0)).abs().median(axis=1)
                else:  # window > 1
                    if method == "standard":
                        mod = windowRoller(dat, window, "mean", min_periods, center)
                        norm = windowRoller(dat, window, "std", min_periods, center)
                    else:  # method == 'modified'
                        mod = windowRoller(dat, window, "median", min_periods, center)
                        norm = windowRoller(
                            dat.subtract(mod, axis=0).abs(),
                            window,
                            "median",
                            min_periods,
                            center,
                        )
        residuals = dat.subtract(mod, axis=0).abs()
        score = residuals.divide(norm, axis=0)

        to_flag = (score.abs() > thresh) & (residuals >= min_residuals)
        for f in field:
            self._flags[to_flag[f], f] = flag
        return self


def _evalStrayLabels(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: Sequence[str],
    reduction_range: str | None = None,
    reduction_drop_flagged: bool = False,  # TODO: still a case ?
    reduction_thresh: float = 3.5,
    reduction_min_periods: int = 1,
    at_least_one: bool = True,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function "reduces" an observations flag to components of it, by applying MAD
    (See references) test onto every components temporal surrounding.

    Parameters
    ----------
    data :
        A dictionary of pandas.Series, holding all the data.

    field :
        The fieldname of the column, holding the labels to be evaluated.

    flags :
        Container to store quality flags to data.

    target :
        A list of strings, holding the column names of the variables, the stray labels
        shall be projected onto.

    val_frame :
        Input NxM DataFrame of observations, where N is the number of observations and
        M the number of components per observation.

    to_flag_frame :
        Input dataframe of observations to be tested, where N is the number of
        observations and M the number of components per observation.

    reduction_range :
        An offset string, denoting the range of the temporal surrounding to include
        into the MAD testing. If ``None`` is passed, no testing will be performed and
        all targets will have the stray flag projected.

    reduction_drop_flagged :
        Wheather or not to drop flagged values other than the value under test, from the
        temporal surrounding before checking the value with MAD.

    reduction_thresh :
        The `critical` value, controlling wheather the MAD score is considered
        referring to an outlier or not. Higher values result in less rigid flagging.
        The default value is widely used in the literature. See references section
        for more details ([1]).

    at_least_one :
        If none of the variables, the outlier label shall be reduced to, is an outlier
        with regard to the test, all (True) or none (False) of the variables are flagged

    flag : float, default BAD
        flag to set.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    val_frame = data[target].to_pandas()
    stray_detects = flags[field] > UNFLAGGED
    stray_detects = stray_detects[stray_detects]
    to_flag_frame = pd.DataFrame(False, columns=target, index=stray_detects.index)

    if reduction_range is None:
        for field in to_flag_frame.columns:
            flags[to_flag_frame.index, field] = flag
        return data, flags

    for var in target:
        for index in enumerate(to_flag_frame.index):
            index_slice = slice(
                index[1] - pd.Timedelta(reduction_range),
                index[1] + pd.Timedelta(reduction_range),
            )
            test_slice = val_frame[var][index_slice].dropna()

            # check, wheather value under test is sufficiently centered:
            first = test_slice.first_valid_index()
            last = test_slice.last_valid_index()
            min_range = pd.Timedelta(reduction_range) / 4

            if (
                pd.Timedelta(index[1] - first) < min_range
                or pd.Timedelta(last - index[1]) < min_range
            ):
                polydeg = 0
            else:
                polydeg = 2

            if reduction_drop_flagged:
                test_slice = test_slice.drop(to_flag_frame.index, errors="ignore")

            if len(test_slice) < reduction_min_periods:
                to_flag_frame.loc[index[1], var] = True
                continue

            x = test_slice.index.values.astype(float)
            x_0 = x[0]
            x = (x - x_0) / 10**12

            polyfitted = poly.polyfit(y=test_slice.values, x=x, deg=polydeg)

            testval = poly.polyval(
                (float(index[1].to_numpy()) - x_0) / 10**12, polyfitted
            )
            testval = val_frame[var][index[1]] - testval

            resids = test_slice.values - poly.polyval(x, polyfitted)
            med_resids = np.median(resids)
            MAD = np.median(np.abs(resids - med_resids))
            crit_val = 0.6745 * (abs(med_resids - testval)) / MAD

            if crit_val > reduction_thresh:
                to_flag_frame.loc[index[1], var] = True

    if at_least_one:
        to_flag_frame[~to_flag_frame.any(axis=1)] = True

    for field in to_flag_frame.columns:
        col = to_flag_frame[field]
        flags[col[col].index, field] = flag

    return data, flags
