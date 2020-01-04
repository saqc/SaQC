#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from saqc.funcs.break_detection import flagBreaksSpektrumBased
from saqc.funcs.spike_detection import flagSpikes_spektrumBased
from saqc.funcs.constants_detection import flagConstantVarianceBased
from saqc.funcs.register import register
from saqc.lib.tools import retrieveTrustworthyOriginal


@register("soilMoisture_spikes")
def flagSoilmoistureSpikes(
    data,
    field,
    flagger,
    filter_window_size="3h",
    raise_factor=0.15,
    dev_cont_factor=0.2,
    noise_barrier=1,
    noise_window_size="12h",
    noise_statistic="covar",
    **kwargs
):

    """
    The Function provides just a call to flagSpikes_spektrumBased, with parameter defaults, that refer to:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.
    """

    return flagSpikes_spektrumBased(
        data,
        field,
        flagger,
        smooth_window=filter_window_size,
        raise_factor=raise_factor,
        deriv_factor=dev_cont_factor,
        noise_func=noise_statistic,
        noise_window=noise_window_size,
        noise_thresh=noise_barrier,
        **kwargs
    )


@register("soilMoisture_breaks")
def flagSoilMoistureBreaks(
    data,
    field,
    flagger,
    diff_method="raw",
    filter_window_size="3h",
    rel_change_rate_min=0.1,
    abs_change_min=0.01,
    first_der_factor=10,
    first_der_window_size="12h",
    scnd_der_ratio_margin_1=0.05,
    scnd_der_ratio_margin_2=10,
    smooth_poly_order=2,
    **kwargs
):

    """
    The Function provides just a call to flagBreaks_spektrumBased, with parameter defaults that refer to:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    """
    return flagBreaksSpektrumBased(
        data,
        field,
        flagger,
        smooth=diff_method,
        smooth_window=filter_window_size,
        thresh_rel=rel_change_rate_min,
        thresh_abs=abs_change_min,
        first_der_factor=first_der_factor,
        first_der_window=first_der_window_size,
        scnd_der_ratio_range=scnd_der_ratio_margin_1,
        scnd_der_ratio_thresh=scnd_der_ratio_margin_2,
        smooth_poly_deg=smooth_poly_order,
        **kwargs
    )


@register("soilMoisture_frost")
def flagSoilMoistureBySoilFrost(
    data,
    field,
    flagger,
    soil_temp_reference,
    tolerated_deviation="1h",
    frost_level=0,
    **kwargs
):

    """This Function is an implementation of the soil temperature based Soil Moisture flagging, as presented in:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    All parameters default to the values, suggested in this publication.

    Function flags Soil moisture measurements by evaluating the soil-frost-level in the moment of measurement.
    Soil temperatures below "frost_level" are regarded as denoting frozen soil state.

    :param data:                        The pandas dataframe holding the data-to-be flagged, as well as the reference
                                        series. Data must be indexed by a datetime series.
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object.
                                        like thingies that refer to the data(including datestrings).
    :param tolerated_deviation:         Offset String. Denoting the maximal temporal deviation,
                                        the soil frost states timestamp is allowed to have, relative to the
                                        data point to-be-flagged.
    :param soil_temp_reference:         A STRING, denoting the fields name in data,
                                        that holds the data series of soil temperature values,
                                        the to-be-flagged values shall be checked against.
    :param frost_level:                 Value level, the flagger shall check against, when evaluating soil frost level.
    """

    # retrieve reference series
    refseries = data[soil_temp_reference].copy()
    ref_use = flagger.isFlagged(
        soil_temp_reference, flag=flagger.GOOD, comparator="=="
    ) | flagger.isFlagged(soil_temp_reference, flag=flagger.UNFLAGGED, comparator="==")
    # drop flagged values:
    refseries = refseries[ref_use.values]
    # drop nan values from reference series, since those are values you dont want to refer to.
    refseries = refseries.dropna()
    # skip further processing if reference series is empty:
    if refseries.empty:
        return data, flagger

    refseries = refseries.reindex(data[field].dropna().index, method="nearest", tolerance=tolerated_deviation)
    refseries = refseries[refseries < frost_level].index

    flagger = flagger.setFlags(field, refseries, **kwargs)
    return data, flagger


@register("soilMoisture_precipitation")
def flagSoilMoistureByPrecipitationEvents(
    data,
    field,
    flagger,
    prec_reference,
    sensor_meas_depth=0,
    sensor_accuracy=0,
    soil_porosity=0,
    std_factor=2,
    std_factor_range="24h",
    raise_reference=None,
    ignore_missing=False,
    **kwargs
):

    """This Function is an implementation of the precipitation based Soil Moisture flagging, as presented in:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    All parameters default to the values, suggested in this publication. (excluding porosity,sensor accuracy and
    sensor depth)


    Function flags Soil moisture measurements by flagging moisture rises that do not follow up a sufficient
    precipitation event. If measurement depth, sensor accuracy of the soil moisture sensor and the porosity of the
    surrounding soil is passed to the function, an inferior level of precipitation, that has to preceed a significant
    moisture raise within 24 hours, can be estimated. If those values are not delivered, this inferior bound is set
    to zero. In that case, any non zero precipitation count will justify any soil moisture raise.

    A data point y_t is flagged an invalid soil moisture raise, if:

    (1) y_t > y_(t-raise_reference)
    (2) y_t - y_(t-"std_factor_range") > "std_factor" * std(y_(t-"std_factor_range"),...,y_t)
    (3) sum(prec(t-24h),...,prec(t)) > sensor_meas_depth * sensor_accuracy * soil_porosity

    NOTE1: np.nan entries in the input precipitation series will be regarded as susipicious and the test will be
    omited for every 24h interval including a np.nan entrie in the original precipitation sampling rate.
    Only entry "0" will be regarded as denoting "No Rainfall".

    NOTE2: The function wont test any values that are flagged suspicious anyway - this may change in a future version.


    :param data:                        The pandas dataframe holding the data-to-be flagged, as well as the reference
                                        series. Data must be indexed by a datetime series and be harmonized onto a
                                        time raster with seconds precision.
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object. (saqc.flagger.X)
    :param prec_reference:              Fieldname of the precipitation meassurements column in data.
    :param sensor_meas_depth:           Measurement depth of the soil moisture sensor, [m].
    :param sensor_accuracy:             Accuracy of the soil moisture sensor, [-].
    :param soil_porosity:               Porosity of moisture sensors surrounding soil, [-].
    :param std_factor:                  The value determines by which rule it is decided, weather a raise in soil
                                        moisture is significant enough to trigger the flag test or not:
                                        Significants is assumed, if the raise is  greater then "std_factor" multiplied
                                        with the last 24 hours standart deviation.
    :param std_factor_range:            Offset String. Denotes the range over witch the standart deviation is obtained,
                                        to test condition [2]. (Should be a multiple of the sampling rate)
    :param raise_reference:             Offset String. Denotes the distance to the datapoint, relatively to witch
                                        it is decided if the current datapoint is a raise or not. Equation [1].
                                        It defaults to None. When None is passed, raise_reference is just the sample
                                        rate of the data. Any raise reference must be a multiple of the (intended)
                                        sample rate and below std_factor_range.
    :param ignore_missing:
    """


    dataseries, moist_rate = retrieveTrustworthyOriginal(data, field, flagger)

    # data not hamronized:
    refseries = data[prec_reference].dropna()
    # abort processing if any of the measurement series has no valid entries!
    if moist_rate is np.nan:
        return data, flagger
    if refseries.empty:
        return data, flagger

    refseries = refseries.reindex(refseries.index.join(dataseries.index, how='outer'))
    # get 24 h prec. monitor
    prec_count = refseries.rolling(window="1D").sum()
    # exclude data not signifying a raise::
    if raise_reference is None:
        raise_reference = 1
    else:
        raise_reference = int(np.ceil(pd.Timedelta(raise_reference)/moist_rate))

    # first raise condition:
    raise_mask = dataseries > dataseries.shift(raise_reference)

    # second raise condition:
    std_factor_range = int(np.ceil(pd.Timedelta(std_factor_range)/moist_rate))
    if ignore_missing:
        std_mask = dataseries.dropna().rolling(std_factor_range).std() < (
                (dataseries - dataseries.shift(std_factor_range)) / std_factor)
    else:
        std_mask = dataseries.rolling(std_factor_range).std() < (
                    (dataseries - dataseries.shift(std_factor_range)) / std_factor)

    dataseries = dataseries[raise_mask & std_mask]
    invalid_indices = (prec_count[dataseries.index] <= sensor_meas_depth*sensor_accuracy*soil_porosity)
    invalid_indices = invalid_indices[invalid_indices]

    # set Flags
    flagger = flagger.setFlags(field, loc=invalid_indices.index, **kwargs)
    return data, flagger


@register("soilMoisture_constant")
def flagSoilMoistureByConstantsDetection(
    data,
    field,
    flagger,
    plateau_window_min="12h",
    plateau_var_limit=0.0005,
    rainfall_window_range="12h",
    filter_window_size=None,
    var_total_nans=np.inf,
    var_consec_nans=np.inf,
    derivative_maximum_lb=0.0025,
    derivative_minimum_ub=0,
    data_max_tolerance=0.95,
    smooth_poly_order=2,
    **kwargs
):

    """

    Note, function has to be harmonized to equidistant freq_grid

    Note, in current implementation, it has to hold that: (rainfall_window_range >= plateau_window_min)

    :param data:                        The pandas dataframe holding the data-to-be flagged.
                                        Data must be indexed by a datetime series and be harmonized onto a
                                        time raster with seconds precision (skips allowed).
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object. (saqc.flagger.X)
    """

    # get plateaus:
    _, comp_flagger = flagConstantVarianceBased(
        data, field, flagger,
        window=plateau_window_min,
        thresh=plateau_var_limit,
        max_missing=var_total_nans,
        max_consec_missing=var_consec_nans
    )

    new_plateaus = (comp_flagger.getFlags(field)).eq(flagger.getFlags(field))
    # get dataseries at its sampling freq:
    dataseries, moist_rate = retrieveTrustworthyOriginal(data, field, flagger)
    # get valuse referring to dataseries:
    new_plateaus.resample(pd.Timedelta(moist_rate)).asfreq()
    # cut out test_slices for min/max derivatives condition check:
    # offset 2 periods:
    rainfall_window_range = int(np.ceil(pd.Timedelta(rainfall_window_range) / moist_rate))
    plateau_window_min = int(np.ceil(pd.Timedelta(plateau_window_min) / moist_rate))
    period_diff = rainfall_window_range - plateau_window_min
    # we cast plateua series to int - because replace has problems with replacing bools by "method".
    new_plateaus = new_plateaus.astype(int)
    # get plateau groups:
    group_counter = new_plateaus.cumsum()
    group_counter = group_counter[group_counter.diff() == 0]
    group_counter.name = 'group_counter'
    plateau_groups = pd.merge(group_counter, dataseries, left_index=True, right_index=True, how='inner')
    # test mean-condition on plateau groups:
    test_barrier = data_max_tolerance*dataseries.max()
    plateau_group_drops = plateau_groups.groupby('group_counter').filter(lambda x: x[field].mean() <= test_barrier)
    # discard values that didnt pass the test from plateau candidate series:
    new_plateaus[plateau_group_drops.index] = 1

    # we extend the plateaus to cover condition testing sets
    # 1: extend backwards (with a technical "one" added):
    cond1_sets = new_plateaus.replace(1, method='bfill', limit=(rainfall_window_range + plateau_window_min))
    # 2. extend forwards:
    if period_diff > 0:
        cond1_sets = cond1_sets.replace(1, method='ffill', limit=period_diff)

    # get first derivative
    if filter_window_size is None:
        filter_window_size = 3 * pd.Timedelta(moist_rate)
    else:
        filter_window_size = pd.Timedelta(filter_window_size)
    first_derivative = dataseries.diff()
    filter_window_seconds = filter_window_size.seconds
    smoothing_periods = int(np.ceil((filter_window_seconds / moist_rate.n)))
    first_derivate = savgol_filter(
        dataseries,
        window_length=smoothing_periods,
        polyorder=smooth_poly_order,
        deriv=1,
    )
    first_derivate = pd.Series(data=first_derivate, index=dataseries.index, name=dataseries.name)
    # cumsumming to seperate continous plateau groups from each other:
    group_counter = cond1_sets.cumsum()
    group_counter = group_counter[group_counter.diff() == 0]
    group_counter.name = 'group_counter'
    group_frame = pd.merge(group_counter, first_derivate, left_index=True, right_index=True, how='inner')
    group_frame = group_frame.groupby('group_counter')
    condition_passed = group_frame.filter(
        lambda x: (x[field].max() >= derivative_maximum_lb) & (x[field].min() <= derivative_minimum_ub))

    flagger = flagger.setFlags(field, loc=condition_passed.index, **kwargs)

    return data, flagger
