#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from saqc.core.register import register
import numpy as np
import dios


@register(masking='field')
def transform(data, field, flagger, func, partition_freq=None, **kwargs):
    """
    Function to transform data columns with a transformation that maps series onto series of the same length.

    Note, that flags get preserved.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-transformed.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    func : Callable[{pd.Series, np.array}, np.array]
        Function to transform data[field] with.
    partition_freq : {np.inf, float, str}, default np.inf
        Determines the segmentation of the data into partitions, the transformation is applied on individually

        * ``np.inf``: Apply transformation on whole data set at once
        * ``x`` > 0 : Apply transformation on successive data chunks of periods length ``x``
        * Offset String : Apply transformation on successive partitions of temporal extension matching the passed offset
          string

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
    """

    data = data.copy()
    val_ser = data[field]
    # partitioning
    if not partition_freq:
        partition_freq = val_ser.shape[0]

    if isinstance(partition_freq, str):
        grouper = pd.Grouper(freq=partition_freq)
    else:
        grouper = pd.Series(data=np.arange(0, val_ser.shape[0]), index=val_ser.index)
        grouper = grouper.transform(lambda x: int(np.floor(x / partition_freq)))

    partitions = val_ser.groupby(grouper)

    for _, partition in partitions:
        if partition.empty:
            continue
        val_ser[partition.index] = func(partition)

    data[field] = val_ser
    return data, flagger


