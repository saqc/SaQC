#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

# see test/functs/fixtures.py for global fixtures "course_..."
import pytest

import saqc
from saqc import UNFLAGGED, DictOfSeries, SaQC
from saqc.core.flags import initFlagsLike
from saqc.lib.ts_operators import linearInterpolation, polynomialInterpolation
from tests.fixtures import char_dict, course_3, course_5  # noqa, todo: fix fixtures


@pytest.mark.parametrize(
    ("window", "center", "expected"),
    [
        (1, True, [3, 2, 3, 2]),
        (2, False, [np.nan, 5, 5, 5]),
        (3, True, [np.nan, 8, 7, np.nan]),
        ("20min", True, [5, 5, 5, np.nan]),
    ],
)
def test_multivariateRolling(window, center, expected):
    data = pd.DataFrame(
        {"a": [1, np.nan, 3, 4], "b": [1, 2, 3, 4], "c": [1, 2, 3, np.nan]},
        index=pd.date_range("2000", periods=4, freq="10min"),
    )
    qc = saqc.SaQC(data)
    qc = qc.rolling(
        ["a", "b", "c"], func="count", target="count", window=window, center=center
    )
    assert np.array_equal(qc.data["count"].values, expected, equal_nan=True)


def test_rollingInterpolateMissing(course_5):
    data, characteristics = course_5(periods=10, nan_slice=[5, 6])
    field = data.columns[0]
    data = DictOfSeries(data)
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).interpolateByRolling(
        field,
        3,
        func=np.median,
        center=True,
        min_periods=0,
        interpol_flag=UNFLAGGED,
    )
    assert qc.data[field][characteristics["missing"]].notna().all()
    qc = SaQC(data, flags).interpolateByRolling(
        field,
        3,
        func=np.nanmean,
        center=False,
        min_periods=3,
        interpol_flag=UNFLAGGED,
    )
    assert qc.data[field][characteristics["missing"]].isna().all()


def test_transform(course_5):
    data, characteristics = course_5(periods=10, nan_slice=[5, 6])
    field = data.columns[0]
    data = DictOfSeries(data)
    flags = initFlagsLike(data)
    qc = SaQC(data, flags)

    result = qc.transform(field, func=linearInterpolation)
    assert result.data[field][characteristics["missing"]].isna().all()

    result = qc.transform(field, func=lambda x: linearInterpolation(x, inter_limit=3))
    assert result.data[field][characteristics["missing"]].notna().all()

    result = qc.transform(
        field,
        func=lambda x: polynomialInterpolation(x, inter_limit=3, inter_order=3),
    )
    assert result.data[field][characteristics["missing"]].notna().all()


def test_resample(course_5):
    data, _ = course_5(freq="1min", periods=30, nan_slice=[1, 11, 12, 22, 24, 26])
    field = data.columns[0]
    data = DictOfSeries(data)
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).resample(
        field,
        "10min",
        "mean",
        maxna=2,
        maxna_group=1,
    )
    assert ~np.isnan(qc.data[field].iloc[0])
    assert np.isnan(qc.data[field].iloc[1])
    assert np.isnan(qc.data[field].iloc[2])


def test_interpolateGrid(course_5, course_3):
    data, _ = course_5()
    data_grid, _ = course_3()
    data["grid"] = data_grid["data"]
    flags = initFlagsLike(data)
    SaQC(data, flags).align("data", "1h", "time")


@pytest.mark.slow
def test_offsetCorrecture():
    data = pd.Series(0, index=pd.date_range("2000", freq="1d", periods=100), name="dat")
    data.iloc[30:40] = -100
    data.iloc[70:80] = 100
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).correctOffset("dat", 40, 20, "3d", 1)
    assert (qc.data["dat"] == 0).all()


# GL-333
def test_resampleSingleEmptySeries():
    qc = saqc.SaQC(pd.DataFrame(1, columns=["a"], index=pd.DatetimeIndex([])))
    qc.resample("a", freq="1d")


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(
            [
                np.random.normal(loc=1 + k * 0.1, scale=3 * (1 - (k * 0.001)))
                for k in range(100)
            ],
            index=pd.date_range("2000", freq="1D", periods=100),
            name="data",
        )
    ],
)
def test_assignZScore(data):
    qc = saqc.SaQC(data)
    qc = qc.assignZScore("data", window="20D")
    mean_res = qc.data["data"].mean()
    std_res = qc.data["data"].std()
    assert -0.1 < mean_res < 0.1
    assert 0.9 < std_res < 1.1
