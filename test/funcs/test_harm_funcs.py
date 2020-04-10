#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np
import pandas as pd

from dios import dios
from test.common import TESTFLAGGER

from saqc.funcs.harm_functions import (
    harmonize,
    deharmonize,
    _interpolate,
    _interpolateGrid,
    _insertGrid,
    _outsortCrap,
    linear2Grid,
    interpolate2Grid,
    shift2Grid,
    aggregate2Grid,
    downsample
)

RESHAPERS = ["nshift", "fshift", "bshift"]

COFLAGGING = [False, True]

SETSHIFTCOMMENT = [False, True]

INTERPOLATIONS = ["fshift", "bshift", "nshift", "nagg", "bagg"]

INTERPOLATIONS2 = ["fagg", "time", "polynomial"]

FREQS = ["15min", "30min"]


@pytest.fixture
def data():
    index = pd.date_range(
        start="1.1.2011 00:00:00", end="1.1.2011 01:00:00", freq="15min"
    )
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 29, 0))
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 28, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 32, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 31, 0))
    index = index.insert(0, pd.Timestamp(2010, 12, 31, 23, 57, 0))
    index = index.drop(pd.Timestamp("2011-01-01 00:30:00"))
    dat = pd.Series(np.linspace(-50, 50, index.size), index=index, name="data")
    # good to have some nan
    dat[-3] = np.nan
    data = dios.DictOfSeries(dat)
    return data


@pytest.fixture
def multi_data():
    index = pd.date_range(
        start="1.1.2011 00:00:00", end="1.1.2011 01:00:00", freq="15min"
    )
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 29, 0))
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 28, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 32, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 31, 0))
    index = index.insert(0, pd.Timestamp(2010, 12, 31, 23, 57, 0))
    index = index.drop(pd.Timestamp("2011-01-01 00:30:00"))
    dat = pd.Series(np.linspace(-50, 50, index.size), index=index, name="data")
    # good to have some nan
    dat[-3] = np.nan
    data = dat.to_frame()
    data.index = data.index.shift(1, "2min")
    dat2 = data.copy()
    dat2.index = dat2.index.shift(1, "17min")
    dat2.rename(columns={"data": "data2"}, inplace=True)
    dat3 = data.copy()
    dat3.index = dat3.index.shift(1, "1h")
    dat3.rename(columns={"data": "data3"}, inplace=True)
    dat3.drop(dat3.index[2:-2], inplace=True)
    # merge
    data = pd.merge(data, dat2, how="outer", left_index=True, right_index=True)
    data = pd.merge(data, dat3, how="outer", left_index=True, right_index=True)
    return dios.DictOfSeries(data)


@pytest.mark.parametrize("method", INTERPOLATIONS2)
def test_gridInterpolation(data, method):
    freq = "15min"
    data = data.squeeze()
    data = (data * np.sin(data)).append(data.shift(1, "2h")).shift(1, "3s")
    kwds = dict(agg_method="sum", downcast_interpolation=True)

    # we are just testing if the interpolation gets passed to the series without causing an error:
    _interpolateGrid(data, freq, method, order=1, **kwds)
    if method == "polynomial":
        _interpolateGrid(data, freq, method, order=2, **kwds)
        _interpolateGrid(data, freq, method, order=10, **kwds)
        data = _insertGrid(data, freq)
        _interpolate(data, method, inter_limit=3)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_outsortCrap(data, flagger):
    field = data.columns[0]
    s = data[field]
    flagger = flagger.initFlags(data)

    drop_index = s.index[5:7]
    flagger = flagger.setFlags(field, loc=drop_index)
    res, *_ = _outsortCrap(s, field, flagger, drop_flags=flagger.BAD)
    assert drop_index.difference(res.index).equals(drop_index)

    flagger = flagger.setFlags(field, loc=s.iloc[0:1], flag=flagger.GOOD)
    drop_index = drop_index.insert(-1, s.index[0])
    to_drop = [flagger.BAD, flagger.GOOD]
    res, *_ = _outsortCrap(s, field, flagger, drop_flags=to_drop)
    assert drop_index.sort_values().difference(res.index).equals(drop_index.sort_values())



@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("reshaper", RESHAPERS)
@pytest.mark.parametrize("co_flagging", COFLAGGING)
def test_harmSingleVarIntermediateFlagging(data, flagger, reshaper, co_flagging):
    flagger = flagger.initFlags(data)
    # make pre harm copies:
    pre_data = data.copy()
    pre_flags = flagger.getFlags()
    freq = "15min"

    assert len(data.columns) == 1
    field = data.columns[0]

    # harmonize data:
    data, flagger = harmonize(data, "data", flagger, freq, "time", reshaper)

    # flag something bad
    flagger = flagger.setFlags("data", loc=data[field].index[3:4])
    data, flagger = deharmonize(data, "data", flagger, co_flagging=co_flagging)
    d = data[field]

    if reshaper == "nshift":
        if co_flagging is True:
            assert flagger.isFlagged(loc=d.index[3:7]).squeeze().all()
            assert (~flagger.isFlagged(loc=d.index[0:3]).squeeze()).all()
            assert (~flagger.isFlagged(loc=d.index[7:]).squeeze()).all()
        if co_flagging is False:
            assert (
                    flagger.isFlagged().squeeze()
                    == [False, False, False, False, True, False, False, False, False]
            ).all()
    if reshaper == "bshift":
        if co_flagging is True:
            assert flagger.isFlagged(loc=d.index[5:7]).squeeze().all()
            assert (~flagger.isFlagged(loc=d.index[0:5]).squeeze()).all()
            assert (~flagger.isFlagged(loc=d.index[7:]).squeeze()).all()
        if co_flagging is False:
            assert (
                    flagger.isFlagged().squeeze()
                    == [False, False, False, False, False, True, False, False, False]
            ).all()
    if reshaper == "fshift":
        if co_flagging is True:
            assert flagger.isFlagged(loc=d.index[3:5]).squeeze().all()
            assert (~flagger.isFlagged(loc=d.index[0:3]).squeeze()).all()
            assert (~flagger.isFlagged(loc=d.index[5:]).squeeze()).all()
        if co_flagging is False:
            assert (
                    flagger.isFlagged().squeeze()
                    == [False, False, False, False, True, False, False, False, False]
            ).all()

    flags = flagger.getFlags()
    assert pre_data[field].equals(data[field])
    assert len(data[field]) == len(flags[field])
    assert (pre_flags[field].index == flags[field].index).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("freq", FREQS)
def test_harmSingleVarInterpolations(data, flagger, interpolation, freq):
    flagger = flagger.initFlags(data)
    flags = flagger.getFlags()
    # make pre harm copies:
    pre_data = data.copy()
    pre_flags = flags.copy()

    assert len(data.columns) == 1
    field = data.columns[0]

    harm_start = data[field].index[0].floor(freq=freq)
    harm_end = data[field].index[-1].ceil(freq=freq)
    test_index = pd.date_range(start=harm_start, end=harm_end, freq=freq)
    data, flagger = harmonize(
        data,
        "data",
        flagger,
        freq,
        interpolation,
        "fshift",
        reshape_shift_comment=False,
        inter_agg="sum",
    )

    if interpolation == "fshift":
        if freq == "15min":
            exp = pd.Series([np.nan, -37.5, -25.0, 0.0, 37.5, 50.0], index=test_index)
            assert data[field].equals(exp)
        if freq == "30min":
            exp = pd.Series([np.nan, -37.5, 0.0, 50.0], index=test_index)
            assert data[field].equals(exp)
    if interpolation == "bshift":
        if freq == "15min":
            exp = pd.Series([-50.0, -37.5, -25.0, 12.5, 37.5, 50.0], index=test_index)
            assert data[field].equals(exp)
        if freq == "30min":
            exp = pd.Series([-50.0, -37.5, 12.5, 50.0], index=test_index)
            assert data[field].equals(exp)
    if interpolation == "nshift":
        if freq == "15min":
            exp = pd.Series([np.nan, -37.5, -25.0, 12.5, 37.5, 50.0], index=test_index)
            assert data[field].equals(exp)
        if freq == "30min":
            exp = pd.Series([np.nan, -37.5, 12.5, 50.0], index=test_index)
            assert data[field].equals(exp)
    if interpolation == "nagg":
        if freq == "15min":
            exp = pd.Series([np.nan, -87.5, -25.0, 0.0, 37.5, 50.0], index=test_index)
            assert data[field].equals(exp)
        if freq == "30min":
            exp = pd.Series([np.nan, -87.5, -25.0, 87.5], index=test_index)
            assert data[field].equals(exp)
    if interpolation == "bagg":
        if freq == "15min":
            exp = pd.Series([-50.0, -37.5, -37.5, 12.5, 37.5, 50.0], index=test_index)
            assert data[field].equals(exp)
        if freq == "30min":
            exp = pd.Series([-50.0, -75.0, 50.0, 50.0], index=test_index)
            assert data[field].equals(exp)

    data, flagger = deharmonize(data, "data", flagger, co_flagging=True)

    # data, flagger = deharmonize(data, "data", flagger, co_flagging=True)
    flags = flagger.getFlags()

    assert pre_data[field].equals(data[field])
    assert len(data[field]) == len(flags[field])
    assert (pre_flags[field].index == flags[field].index).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("shift_comment", SETSHIFTCOMMENT)
def test_multivariatHarmonization(multi_data, flagger, shift_comment):
    flagger = flagger.initFlags(multi_data)
    flags = flagger.getFlags()
    # for comparison
    pre_data = multi_data.copy()
    pre_flags = flags.copy()
    freq = "15min"

    # harm:
    multi_data, flagger = harmonize(
        multi_data,
        "data",
        flagger,
        freq,
        "time",
        "nshift",
        reshape_shift_comment=shift_comment,
    )

    multi_data, flagger = harmonize(
        multi_data,
        "data2",
        flagger,
        freq,
        "bagg",
        "bshift",
        inter_agg="sum",
        reshape_agg="max",
        reshape_shift_comment=shift_comment,
    )

    multi_data, flagger = harmonize(
        multi_data,
        "data3",
        flagger,
        freq,
        "fshift",
        "fshift",
        reshape_shift_comment=shift_comment,
    )

    for c in multi_data.columns:
        harm_start = multi_data[c].index[0].floor(freq=freq)
        harm_end = multi_data[c].index[-1].ceil(freq=freq)
        test_index = pd.date_range(start=harm_start, end=harm_end, freq=freq)

        assert multi_data[c].index.equals(test_index)
        assert pd.Timedelta(pd.infer_freq(multi_data[c].index)) == pd.Timedelta(freq)

    multi_data, flagger = deharmonize(multi_data, "data3", flagger, co_flagging=False)
    multi_data, flagger = deharmonize(multi_data, "data2", flagger, co_flagging=True)
    multi_data, flagger = deharmonize(multi_data, "data", flagger, co_flagging=True)

    for c in multi_data.columns:
        flags = flagger.getFlags()
        assert pre_data[c].equals(multi_data[pre_data.columns.to_list()][c])
        assert len(multi_data[c]) == len(flags[c])
        assert (pre_flags[c].index == flags[c].index).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)


def test_wrapper(data, flagger):
    # we are only testing, whether the wrappers do pass processing:
    field = data.columns[0]
    freq = '15min'
    flagger = flagger.initFlags(data)
    downsample(data, field, flagger, '15min', '30min', agg_func="sum", sample_func="mean")
    linear2Grid(data, field, flagger, freq, method='nagg', func="max", drop_flags=None)
    aggregate2Grid(data, field, flagger, freq, value_func="sum",
                   flag_func="max", method='nagg', drop_flags=None)
    shift2Grid(data, field, flagger, freq, method='nshift', drop_flags=None)
    interpolate2Grid(data, field, flagger, freq, method="spline")

if __name__ == "__main__":
    flagger=TESTFLAGGER[2]
    dat = data()
    field, *_ = dat.columns
    test_harmSingleVarIntermediateFlagging(dat, flagger, 'fshift', True)
