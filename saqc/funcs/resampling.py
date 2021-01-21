#! /usr/bin/env python
# -*- coding: utf-8 -*-


import dios

import numpy as np
import logging

import pandas as pd

from saqc.core.register import register
from saqc.funcs.tools import copy, drop, rename
from saqc.funcs.interpolation import interpolateIndex
from saqc.lib.tools import dropper, evalFreqStr
from saqc.lib.ts_operators import shift2Freq, aggregate2Freq

logger = logging.getLogger("SaQC")


METHOD2ARGS = {
    "inverse_fshift": ("backward", pd.Timedelta),
    "inverse_bshift": ("forward", pd.Timedelta),
    "inverse_nshift": ("nearest", lambda x: pd.Timedelta(x) / 2),
    "inverse_fagg": ("bfill", pd.Timedelta),
    "inverse_bagg": ("ffill", pd.Timedelta),
    "inverse_nagg": ("nearest", lambda x: pd.Timedelta(x) / 2),
    "match": (None, lambda x: "0min"),
}


@register(masking='none')
def aggregate(
        data, field, flagger, freq, value_func, flag_func=np.nanmax, method="nagg", to_drop=None, **kwargs
):
    """
    A method to "regularize" data by aggregating (resampling) data at a regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    The data will therefor get aggregated with a function, specified by the `value_func` parameter and
    the result gets projected onto the new timestamps with a method, specified by "method".

    The following method (keywords) are available:

    * ``'nagg'``: (aggreagtion to nearest) - all values in the range (+/- freq/2) of a grid point get aggregated with
      `agg_func`. and assigned to it. Flags get aggregated by `flag_func` and assigned the same way.
    * ``'bagg'``: (backwards aggregation) - all values in a sampling interval get aggregated with agg_func and the
      result gets assigned to the last regular timestamp. Flags get aggregated by `flag_func` and assigned the same way.
    * ``'fagg'``: (forward aggregation) - all values in a sampling interval get aggregated with agg_func and the result
      gets assigned to the next regular timestamp. Flags get aggregated by `flag_func` and assigned the same way.

    Note, that, if there is no valid data (exisitng and not-na) available in a sampling interval assigned to a regular
    timestamp by the selected method, nan gets assigned to this timestamp. The associated flag will be of value
    ``flagger.UNFLAGGED``.

    Note: the method will likely and significantly alter values and shape of ``data[field]``. The original data is kept
    in the data dios and assigned to the fieldname ``field + '_original'``.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.freq
    freq : str
        The sampling frequency the data is to be aggregated (resampled) at.
    value_func : Callable
        The function you want to use for aggregation.
    flag_func : Callable
        The function you want to aggregate the flags with. It should be capable of operating on the flags dtype
        (usually ordered categorical).
    method : {'fagg', 'bagg', 'nagg'}, default 'nagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceeding, succeeding or
        "surrounding" interval). See description above for more details.
    to_drop : {List[str], str}, default None
        Flagtypes you want to drop before aggregation - effectively excluding values that are flagged
        with a flag in to_drop from the aggregation process. Default results in flagger.BAD
        values being dropped initially.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = copy(data, field, flagger, field + '_original')
    data, flagger = resample(
        data,
        field,
        flagger,
        freq,
        agg_func=value_func,
        flag_agg_func=flag_func,
        method=method,
        empty_intervals_flag=flagger.UNFLAGGED,
        to_drop=to_drop,
        all_na_2_empty=True,
        **kwargs,
    )
    return data, flagger


@register(masking='none')
def linear(data, field, flagger, freq, to_drop=None, **kwargs):
    """
    A method to "regularize" data by interpolating linearly the data at regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Interpolated values will get assigned the worst flag within freq-range.

    Note: the method will likely and significantly alter values and shape of ``data[field]``. The original data is kept
    in the data dios and assigned to the fieldname ``field + '_original'``.

    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    ``flagger.UNFLAGGED``.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.freq
    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.
    to_drop : {List[str], str}, default None
        Flagtypes you want to drop before interpolation - effectively excluding values that are flagged
        with a flag in to_drop from the interpolation process. Default results in flagger.BAD
        values being dropped initially.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = copy(data, field, flagger, field + '_original')
    data, flagger = interpolateIndex(
        data, field, flagger, freq, "time", to_drop=to_drop, empty_intervals_flag=flagger.UNFLAGGED, **kwargs
    )
    return data, flagger


@register(masking='none')
def interpolate(data, field, flagger, freq, method, order=1, to_drop=None, **kwargs, ):
    """
    A method to "regularize" data by interpolating the data at regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Interpolated values will get assigned the worst flag within freq-range.

    There are available all the interpolations from the pandas.Series.interpolate method and they are called by
    the very same keywords.

    Note, that, to perform a timestamp aware, linear interpolation, you have to pass ``'time'`` as `method`,
    and NOT ``'linear'``.

    Note: the `method` will likely and significantly alter values and shape of ``data[field]``. The original data is
    kept in the data dios and assigned to the fieldname ``field + '_original'``.

    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    ``flagger.UNFLAGGED``.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.freq
    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.
    order : int, default 1
        If your selected interpolation method can be performed at different *orders* - here you pass the desired
        order.
    to_drop : {List[str], str}, default None
        Flagtypes you want to drop before interpolation - effectively excluding values that are flagged
        with a flag in `to_drop` from the interpolation process. Default results in ``flagger.BAD``
        values being dropped initially.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = copy(data, field, flagger, field + '_original')
    data, flagger = interpolateIndex(
        data,
        field,
        flagger,
        freq,
        method=method,
        inter_order=order,
        to_drop=to_drop,
        empty_intervals_flag=flagger.UNFLAGGED,
        **kwargs,
    )
    return data, flagger


@register(masking='none')
def mapToOriginal(data, field, flagger, method, to_drop=None, **kwargs):
    """
    The Function function "undoes" regularization, by regaining the original data and projecting the
    flags calculated for the regularized data onto the original ones.

    Afterwards the regularized data is removed from the data dios and ``'field'`` will be associated
    with the original data "again".

    Wherever the flags in the original data are "better" then the regularized flags projected on them,
    they get overridden with this regularized flags value.

    Which regularized flags are to be projected on which original flags, is controlled by the "method" parameters.

    Generally, if you regularized with the method "X", you should pass the method "inverse_X" to the deharmonization.
    If you regularized with an interpolation, the method "inverse_interpolation" would be the appropriate choice.
    Also you should pass the same drop flags keyword.

    The deharm methods in detail:
    ("original_flags" are associated with the original data that is to be regained,
    "regularized_flags" are associated with the regularized data that is to be "deharmonized",
    "freq" refers to the regularized datas sampling frequencie)

    * ``'inverse_nagg'``: all original_flags within the range *+/- freq/2* of a regularized_flag, get assigned this
      regularized flags value. (if regularized_flags > original_flag)
    * ``'inverse_bagg'``: all original_flags succeeding a regularized_flag within the range of "freq", get assigned this
      regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_fagg'``: all original_flags preceeding a regularized_flag within the range of "freq", get assigned this
      regularized flags value. (if regularized_flag > original_flag)

    * ``'inverse_interpolation'``: all original_flags within the range *+/- freq* of a regularized_flag, get assigned this
      regularized flags value (if regularized_flag > original_flag).

    * ``'inverse_nshift'``: That original_flag within the range +/- *freq/2*, that is nearest to a regularized_flag,
      gets the regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_bshift'``: That original_flag succeeding a source flag within the range freq, that is nearest to a
      regularized_flag, gets assigned this regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_nshift'``: That original_flag preceeding a regularized flag within the range freq, that is nearest to a
      regularized_flag, gets assigned this regularized flags value. (if source_flag > original_flag)

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-deharmonized.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.freq
    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift',
            'inverse_interpolation'}
        The method used for projection of regularized flags onto original flags. See description above for more
        details.
    to_drop : {List[str], str}, default None
        Flagtypes you want to drop before interpolation - effectively excluding values that are flagged
        with a flag in to_drop from the interpolation process. Default results in flagger.BAD
        values being dropped initially.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    newfield = str(field) + '_original'
    data, flagger = reindexFlags(data, newfield, flagger, method, source=field, to_drop=to_drop, **kwargs)
    data, flagger = drop(data, field, flagger)
    data, flagger = rename(data, newfield, flagger, field)
    return data, flagger


@register(masking='none')
def shift(data, field, flagger, freq, method='nshift', to_drop=None, empty_intervals_flag=None, freq_check=None,
          **kwargs):

    data, flagger = copy(data, field, flagger, field + '_original')
    data, flagger = _shift(data, field, flagger, freq, method=method, to_drop=to_drop,
                          empty_intervals_flag=empty_intervals_flag, freq_check=freq_check, **kwargs)
    return data, flagger


@register(masking='none')
def _shift(data, field, flagger, freq, method='nshift', to_drop=None, empty_intervals_flag=None, freq_check=None,
          **kwargs):
    """
    Function to shift data points to regular (equidistant) timestamps.
    Values get shifted according to the keyword passed to the `method` parameter.

    * ``'nshift'``: every grid point gets assigned the nearest value in its range. (range = +/- 0.5 * `freq`)
    * ``'bshift'``:  every grid point gets assigned its first succeeding value - if there is one available in the
      succeeding sampling interval.
    * ``'fshift'``:  every grid point gets assigned its ultimately preceeding value - if there is one available in
      the preceeding sampling interval.

    Note: all data nans get excluded defaultly from shifting. If `to_drop` is ``None``, - all *BAD* flagged values get
    excluded as well.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-shifted.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    freq : str
        An frequency Offset String that will be interpreted as the sampling rate you want the data to be shifted to.
    method: {'fagg', 'bagg', 'nagg'}, default 'nshift'
        Specifies if datapoints get propagated forwards, backwards or to the nearest grid timestamp. See function
        description for more details.
    empty_intervals_flag : {None, str}, default None
        A Flag, that you want to assign to grid points, where no values are avaible to be shifted to.
        Default triggers flagger.UNFLAGGED to be assigned.
    to_drop : {None, str, List[str]}, default None
        Flags that refer to values you want to drop before shifting - effectively, excluding values that are flagged
        with a flag in to_drop from the shifting process. Default - to_drop = None  - results in flagger.BAD
        values being dropped initially.
    freq_check : {None, 'check', 'auto'}, default None

        * ``None``: do not validate frequency-string passed to `freq`
        * ``'check'``: estimate frequency and log a warning if estimate miss matches frequency string passed to `freq`,
          or if no uniform sampling rate could be estimated
        * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """
    data = data.copy()
    datcol = data[field]
    flagscol = flagger.getFlags(field)

    if empty_intervals_flag is None:
        empty_intervals_flag = flagger.UNFLAGGED

    drop_mask = dropper(field, to_drop, flagger, flagger.BAD)
    drop_mask |= datcol.isna()
    datcol[drop_mask] = np.nan
    datcol.dropna(inplace=True)
    freq = evalFreqStr(freq, freq_check, datcol.index)
    if datcol.empty:
        data[field] = datcol
        reshaped_flagger = flagger.initFlags(datcol).setFlags(field, flag=flagscol, force=True, inplace=True, **kwargs)
        flagger = flagger.slice(drop=field).merge(reshaped_flagger, subset=[field], inplace=True)
        return data, flagger

    flagscol.drop(drop_mask[drop_mask].index, inplace=True)

    datcol = shift2Freq(datcol, method, freq, fill_value=np.nan)
    flagscol = shift2Freq(flagscol, method, freq, fill_value=empty_intervals_flag)
    data[field] = datcol
    reshaped_flagger = flagger.initFlags(datcol).setFlags(field, flag=flagscol, force=True, inplace=True, **kwargs)
    flagger = flagger.slice(drop=field).merge(reshaped_flagger, subset=[field], inplace=True)
    return data, flagger


@register(masking='field')
def resample(
    data,
    field,
    flagger,
    freq,
    agg_func=np.mean,
    method="bagg",
    max_invalid_total_d=np.inf,
    max_invalid_consec_d=np.inf,
    max_invalid_consec_f=np.inf,
    max_invalid_total_f=np.inf,
    flag_agg_func=max,
    empty_intervals_flag=None,
    to_drop=None,
    all_na_2_empty=False,
    freq_check=None,
    **kwargs
):
    """
    Function to resample the data. Afterwards the data will be sampled at regular (equidistant) timestamps
    (or Grid points). Sampling intervals therefor get aggregated with a function, specifyed by 'agg_func' parameter and
    the result gets projected onto the new timestamps with a method, specified by "method". The following method
    (keywords) are available:

    * ``'nagg'``: all values in the range (+/- `freq`/2) of a grid point get aggregated with agg_func and assigned to it.
    * ``'bagg'``: all values in a sampling interval get aggregated with agg_func and the result gets assigned to the last
      grid point.
    * ``'fagg'``: all values in a sampling interval get aggregated with agg_func and the result gets assigned to the next
      grid point.


    Note, that. if possible, functions passed to agg_func will get projected internally onto pandas.resample methods,
    wich results in some reasonable performance boost - however, for this to work, you should pass functions that have
    the __name__ attribute initialised and the according methods name assigned to it.
    Furthermore, you shouldnt pass numpys nan-functions
    (``nansum``, ``nanmean``,...) because those for example, have ``__name__ == 'nansum'`` and they will thus not
    trigger ``resample.func()``, but the slower ``resample.apply(nanfunc)``. Also, internally, no nans get passed to
    the functions anyway, so that there is no point in passing the nan functions.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-resampled.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    freq : str
        An Offset String, that will be interpreted as the frequency you want to resample your data with.
    agg_func : Callable
        The function you want to use for aggregation.
    method: {'fagg', 'bagg', 'nagg'}, default 'bagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceding, succeeding or
        "surrounding" interval). See description above for more details.
    max_invalid_total_d : {np.inf, int}, np.inf
        Maximum number of invalid (nan) datapoints, allowed per resampling interval. If max_invalid_total_d is
        exceeded, the interval gets resampled to nan. By default (``np.inf``), there is no bound to the number of nan
        values in an interval and only intervals containing ONLY nan values or those, containing no values at all,
        get projected onto nan
    max_invalid_consec_d : {np.inf, int}, default np.inf
        Maximum number of consecutive invalid (nan) data points, allowed per resampling interval.
        If max_invalid_consec_d is exceeded, the interval gets resampled to nan. By default (np.inf),
        there is no bound to the number of consecutive nan values in an interval and only intervals
        containing ONLY nan values, or those containing no values at all, get projected onto nan.
    max_invalid_total_f : {np.inf, int}, default np.inf
        Same as `max_invalid_total_d`, only applying for the flags. The flag regarded as "invalid" value,
        is the one passed to empty_intervals_flag (default=``flagger.BAD``).
        Also this is the flag assigned to invalid/empty intervals.
    max_invalid_total_f : {np.inf, int}, default np.inf
        Same as `max_invalid_total_f`, only applying onto flags. The flag regarded as "invalid" value, is the one passed
        to empty_intervals_flag (default=flagger.BAD). Also this is the flag assigned to invalid/empty intervals.
    flag_agg_func : Callable, default: max
        The function you want to aggregate the flags with. It should be capable of operating on the flags dtype
        (usually ordered categorical).
    empty_intervals_flag : {None, str}, default None
        A Flag, that you want to assign to invalid intervals. Invalid are those intervals, that contain nan values only,
        or no values at all. Furthermore the empty_intervals_flag is the flag, serving as "invalid" identifyer when
        checking for `max_total_invalid_f` and `max_consec_invalid_f patterns`. Default triggers ``flagger.BAD`` to be
        assigned.
    to_drop : {None, str, List[str]}, default None
        Flags that refer to values you want to drop before resampling - effectively excluding values that are flagged
        with a flag in to_drop from the resampling process - this means that they also will not be counted in the
        the `max_consec`/`max_total evaluation`. `to_drop` = ``None`` results in NO flags being dropped initially.
    freq_check : {None, 'check', 'auto'}, default None

        * ``None``: do not validate frequency-string passed to `freq`
        * ``'check'``: estimate frequency and log a warning if estimate miss matchs frequency string passed to 'freq', or
          if no uniform sampling rate could be estimated
        * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data = data.copy()
    datcol = data[field]
    flagscol = flagger.getFlags(field)
    if empty_intervals_flag is None:
        empty_intervals_flag = flagger.BAD

    drop_mask = dropper(field, to_drop, flagger, [])
    datcol.drop(datcol[drop_mask].index, inplace=True)
    freq = evalFreqStr(freq, freq_check, datcol.index)
    flagscol.drop(flagscol[drop_mask].index, inplace=True)
    if all_na_2_empty:
        if datcol.dropna().empty:
            datcol = pd.Series([], index=pd.DatetimeIndex([]), name=field)

    if datcol.empty:
        # for consistency reasons - return empty data/flags column when there is no valid data left
        # after filtering.
        data[field] = datcol
        reshaped_flagger = flagger.initFlags(datcol).setFlags(field, flag=flagscol, force=True, inplace=True, **kwargs)
        flagger = flagger.slice(drop=field).merge(reshaped_flagger, subset=[field], inplace=True)
        return data, flagger

    datcol = aggregate2Freq(
        datcol,
        method,
        freq,
        agg_func,
        fill_value=np.nan,
        max_invalid_total=max_invalid_total_d,
        max_invalid_consec=max_invalid_consec_d,
    )
    flagscol = aggregate2Freq(
        flagscol,
        method,
        freq,
        flag_agg_func,
        fill_value=empty_intervals_flag,
        max_invalid_total=max_invalid_total_f,
        max_invalid_consec=max_invalid_consec_f,
    )

    # data/flags reshaping:
    data[field] = datcol
    reshaped_flagger = flagger.initFlags(datcol).setFlags(field, flag=flagscol, force=True, inplace=True, **kwargs)
    flagger = flagger.slice(drop=field).merge(reshaped_flagger, subset=[field], inplace=True)
    return data, flagger


@register(masking='field')
def reindexFlags(data, field, flagger, method, source, freq=None, to_drop=None, freq_check=None, **kwargs):

    """
    The Function projects flags of "source" onto flags of "field". Wherever the "field" flags are "better" then the
    source flags projected on them, they get overridden with this associated source flag value.

    Which "field"-flags are to be projected on which source flags, is controlled by the "method" and "freq"
    parameters.

    method: (field_flag in associated with "field", source_flags associated with "source")

    'inverse_nagg' - all field_flags within the range +/- freq/2 of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)
    'inverse_bagg' - all field_flags succeeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)
    'inverse_fagg' - all field_flags preceeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)

    'inverse_interpolation' - all field_flags within the range +/- freq of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)

    'inverse_nshift' - That field_flag within the range +/- freq/2, that is nearest to a source_flag, gets the source
        flags value. (if source_flag > field_flag)
    'inverse_bshift' - That field_flag succeeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)
    'inverse_nshift' - That field_flag preceeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)

    'match' - any field_flag with a timestamp matching a source_flags timestamp gets this source_flags value
    (if source_flag > field_flag)

    Note, to undo or backtrack a resampling/shifting/interpolation that has been performed with a certain method,
    you can just pass the associated "inverse" method. Also you should pass the same drop flags keyword.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to project the source-flags onto.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift'}
        The method used for projection of source flags onto field flags. See description above for more details.
    source : str
        The source source of flags projection.
    freq : {None, str},default None
        The freq determines the projection range for the projection method. See above description for more details.
        Defaultly (None), the sampling frequency of source is used.
    to_drop : {None, str, List[str]}, default None
        Flags referring to values that are to drop before flags projection. Relevant only when projecting with an
        inverted shift method. Defaultly flagger.BAD is listed.
    freq_check : {None, 'check', 'auto'}, default None
        - None: do not validate frequency-string passed to `freq`
        - 'check': estimate frequency and log a warning if estimate miss matchs frequency string passed to 'freq', or
            if no uniform sampling rate could be estimated
        - 'auto': estimate frequency and use estimate. (Ignores `freq` parameter.)

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """
    flagscol, metacols = flagger.getFlags(source, full=True)
    if flagscol.empty:
        return data, flagger
    target_datcol = data[field]
    target_flagscol, target_metacols = flagger.getFlags(field, full=True)

    if (freq is None) and (method != "match"):
        freq_check = 'auto'

    freq = evalFreqStr(freq, freq_check, flagscol.index)

    if method[-13:] == "interpolation":
        backprojected = flagscol.reindex(target_flagscol.index, method="bfill", tolerance=freq)
        fwrdprojected = flagscol.reindex(target_flagscol.index, method="ffill", tolerance=freq)
        b_replacement_mask = (backprojected > target_flagscol) & (backprojected >= fwrdprojected)
        f_replacement_mask = (fwrdprojected > target_flagscol) & (fwrdprojected > backprojected)
        target_flagscol.loc[b_replacement_mask] = backprojected.loc[b_replacement_mask]
        target_flagscol.loc[f_replacement_mask] = fwrdprojected.loc[f_replacement_mask]

        backprojected_meta = {}
        fwrdprojected_meta = {}
        for meta_key in target_metacols.keys():
            backprojected_meta[meta_key] = metacols[meta_key].reindex(target_metacols[meta_key].index, method='bfill',
                                                                      tolerance=freq)
            fwrdprojected_meta[meta_key] = metacols[meta_key].reindex(target_metacols[meta_key].index, method='ffill',
                                                                      tolerance=freq)
            target_metacols[meta_key].loc[b_replacement_mask] = backprojected_meta[meta_key].loc[b_replacement_mask]
            target_metacols[meta_key].loc[f_replacement_mask] = fwrdprojected_meta[meta_key].loc[f_replacement_mask]

    if method[-3:] == "agg" or method == "match":
        # Aggregation - Inversion
        projection_method = METHOD2ARGS[method][0]
        tolerance = METHOD2ARGS[method][1](freq)
        flagscol = flagscol.reindex(target_flagscol.index, method=projection_method, tolerance=tolerance)
        replacement_mask = flagscol > target_flagscol
        target_flagscol.loc[replacement_mask] = flagscol.loc[replacement_mask]
        for meta_key in target_metacols.keys():
            metacols[meta_key] = metacols[meta_key].reindex(target_metacols[meta_key].index, method=projection_method,
                                                            tolerance=tolerance)
            target_metacols[meta_key].loc[replacement_mask] = metacols[meta_key].loc[replacement_mask]

    if method[-5:] == "shift":
        # NOTE: although inverting a simple shift seems to be a less complex operation, it has quite some
        # code assigned to it and appears to be more verbose than inverting aggregation -
        # that owes itself to the problem of BAD/invalid values blocking a proper
        # shift inversion and having to be outsorted before shift inversion and re-inserted afterwards.
        #
        # starting with the dropping and its memorization:

        drop_mask = dropper(field, to_drop, flagger, flagger.BAD)
        drop_mask |= target_datcol.isna()
        target_flagscol_drops = target_flagscol[drop_mask]
        target_flagscol.drop(drop_mask[drop_mask].index, inplace=True)

        # shift inversion
        projection_method = METHOD2ARGS[method][0]
        tolerance = METHOD2ARGS[method][1](freq)
        flags_merged = pd.merge_asof(
            flagscol,
            pd.Series(target_flagscol.index.values, index=target_flagscol.index, name="pre_index"),
            left_index=True,
            right_index=True,
            tolerance=tolerance,
            direction=projection_method,
        )
        flags_merged.dropna(subset=["pre_index"], inplace=True)
        flags_merged = flags_merged.set_index(["pre_index"]).squeeze()

        # write flags to target
        replacement_mask = flags_merged > target_flagscol.loc[flags_merged.index]
        target_flagscol.loc[replacement_mask[replacement_mask].index] = flags_merged.loc[replacement_mask]

        # reinsert drops
        target_flagscol = target_flagscol.reindex(target_flagscol.index.join(target_flagscol_drops.index, how="outer"))
        target_flagscol.loc[target_flagscol_drops.index] = target_flagscol_drops.values

        for meta_key in target_metacols.keys():
            target_metadrops = target_metacols[meta_key][drop_mask]
            target_metacols[meta_key].drop(drop_mask[drop_mask].index, inplace=True)
            meta_merged = pd.merge_asof(
                metacols[meta_key],
                pd.Series(target_metacols[meta_key].index.values, index=target_metacols[meta_key].index,
                          name="pre_index"),
                left_index=True,
                right_index=True,
                tolerance=tolerance,
                direction=projection_method,
            )
            meta_merged.dropna(subset=["pre_index"], inplace=True)
            meta_merged = meta_merged.set_index(["pre_index"]).squeeze()
            # reinsert drops
            target_metacols[meta_key][replacement_mask[replacement_mask].index] = meta_merged[replacement_mask]
            target_metacols[meta_key] = target_metacols[meta_key].reindex(
                target_metacols[meta_key].index.join(target_metadrops.index, how="outer"))
            target_metacols[meta_key].loc[target_metadrops.index] = target_metadrops.values

    flagger = flagger.setFlags(field, flag=target_flagscol, with_extra=True, **target_metacols)
    return data, flagger