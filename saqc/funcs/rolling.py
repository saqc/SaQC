#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from saqc.core.register import register


@register(masking='field')
def roll(data, field, flagger, winsz, func=np.mean, eval_flags=True, min_periods=0, center=True,
         _return_residues=False, **kwargs):
    """
        Models the data with the rolling mean and returns the residues.

        Note, that the residues will be stored to the `field` field of the input data, so that the data that is modelled
        gets overridden.

        Parameters
        ----------
        data : dios.DictOfSeries
            A dictionary of pandas.Series, holding all the data.
        field : str
            The fieldname of the column, holding the data-to-be-modelled.
        flagger : saqc.flagger.BaseFlagger
            A flagger object, holding flags and additional Informations related to `data`.
        winsz : {int, str}
            The size of the window you want to roll with. If an integer is passed, the size
            refers to the number of periods for every fitting window. If an offset string is passed,
            the size refers to the total temporal extension.
            For regularly sampled timeseries, the period number will be casted down to an odd number if
            center = True.
        func : Callable[np.array, float], default np.mean
            Function to apply on the rolling window and obtain the curve fit value.
        eval_flags : bool, default True
            Wheather or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
            flag present in the interval, the data for its calculation was obtained from.
            Currently not implemented in combination with not-harmonized timeseries.
        min_periods : int, default 0
            The minimum number of periods, that has to be available in every values fitting surrounding for the mean
            fitting to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
            regardless of the number of values present.
        center : bool, default True
            Wheather or not to center the window the mean is calculated of around the reference value. If False,
            the reference value is placed to the right of the window (classic rolling mean with lag.)

        Returns
        -------
        data : dios.DictOfSeries
            A dictionary of pandas.Series, holding all the data.
            Data values may have changed relatively to the data input.
        flagger : saqc.flagger.BaseFlagger
            The flagger object, holding flags and additional Informations related to `data`.
            Flags values may have changed relatively to the flagger input.
        """

    data = data.copy()
    to_fit = data[field]
    flags = flagger.getFlags(field)
    if to_fit.empty:
        return data, flagger

    # starting with the annoying case: finding the rolling interval centers of not-harmonized input time series:
    if (to_fit.index.freqstr is None) and center:
        if isinstance(winsz, int):
            raise NotImplementedError(
                "Integer based window size is not supported for not-harmonized"
                'sample series when rolling with "center=True".'
            )
        # get interval centers
        centers = np.floor((to_fit.rolling(pd.Timedelta(winsz) / 2, closed="both", min_periods=min_periods).count()))
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        roller = to_fit.rolling(pd.Timedelta(winsz), closed="both", min_periods=min_periods)
        try:
            means = getattr(roller, func.__name__)()
        except AttributeError:
            means = to_fit.rolling(pd.Timedelta(winsz), closed="both", min_periods=min_periods).apply(func)

        def center_func(x, y=centers):
            pos = x.index[int(len(x) - y[x.index[-1]])]
            return y.index.get_loc(pos)

        centers_iloc = centers.rolling(winsz, closed="both").apply(center_func, raw=False).astype(int)
        temp = means.copy()
        for k in centers_iloc.iteritems():
            means.iloc[k[1]] = temp[k[0]]
        # last values are false, due to structural reasons:
        means[means.index[centers_iloc[-1]]: means.index[-1]] = np.nan

    # everything is more easy if data[field] is harmonized:
    else:
        if isinstance(winsz, str):
            winsz = int(np.floor(pd.Timedelta(winsz) / pd.Timedelta(to_fit.index.freqstr)))
        if (winsz % 2 == 0) & center:
            winsz = int(winsz - 1)

        roller = to_fit.rolling(window=winsz, center=center, closed="both")
        try:
            means = getattr(roller, func.__name__)()
        except AttributeError:
            means = to_fit.rolling(window=winsz, center=center, closed="both").apply(func)

    if _return_residues:
        residues = means - to_fit

    data[field] = residues
    if eval_flags:
        num_cats, codes = flags.factorize()
        num_cats = pd.Series(num_cats, index=flags.index).rolling(winsz, center=True, min_periods=min_periods).max()
        nan_samples = num_cats[num_cats.isna()]
        num_cats.drop(nan_samples.index, inplace=True)
        to_flag = pd.Series(codes[num_cats.astype(int)], index=num_cats.index)
        to_flag = to_flag.align(nan_samples)[0]
        to_flag[nan_samples.index] = flags[nan_samples.index]
        flagger = flagger.setFlags(field, to_flag.values, **kwargs)

    return data, flagger