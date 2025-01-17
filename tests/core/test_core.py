#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import copy

import numpy as np
import pandas as pd
import pytest

from saqc import BAD, FILTER_ALL, FILTER_NONE, UNFLAGGED, SaQC
from saqc.core import DictOfSeries, Flags, flagging, processing, register
from saqc.core.flags import initFlagsLike
from saqc.lib.types import OptionalNone
from tests.common import initData

OPTIONAL = [False, True]
U = UNFLAGGED
B = BAD


@pytest.fixture
def data():
    return initData(3)


@pytest.fixture
def flags(data, optional):
    if optional:
        return None
    return DictOfSeries(initFlagsLike(data[data.columns[::2]]))


def test_errorHandling(data):
    @processing()
    def raisingFunc(saqc, field, **kwargs):
        raise TypeError

    var1 = data.columns[0]
    qc = SaQC(data)

    with pytest.raises(TypeError):
        qc.raisingFunc(var1)


@pytest.mark.parametrize("optional", OPTIONAL)
def test_dtypes(data, flags):
    """
    Test if the categorical dtype is preserved through the core functionality
    """

    @register(mask=["field"], demask=["field"], squeeze=["field"])
    def flagAll(saqc, field, **kwargs):
        saqc._flags[:, field] = BAD
        return saqc

    flags = initFlagsLike(data)
    flags_raw = DictOfSeries(flags)
    var1, var2 = data.columns[:2]

    pflags = SaQC(data, flags=flags_raw).flagAll(var1).flagAll(var2).flags

    for c in pflags.columns:
        assert pflags[c].dtype == flags[c].dtype


def test_autoTranslation():
    data = pd.Series(
        [1, 2], index=pd.date_range("2000", periods=2, freq="D"), name="data"
    )
    flags = pd.DataFrame(["BAD", "UNFLAGGED"], index=data.index, columns=["data"])
    qc = SaQC(data=data, flags=flags, scheme="simple")

    assert (qc.flags["data"] == ["BAD", "UNFLAGGED"]).all()  # external flags
    assert (qc._flags["data"] == [BAD, UNFLAGGED]).all()  # internal flags

    qc.scheme = "float"

    assert (qc.flags["data"] == [BAD, UNFLAGGED]).all()  # external flags
    assert (qc._flags["data"] == [BAD, UNFLAGGED]).all()  # internal flags


def test_new_call(data):
    qc = SaQC(data)
    qc = qc.flagRange("var1", max=5)


def test_SaQC_attributes():
    """Test if all instance attributes are in SaQC._attributes"""
    qc = SaQC()
    for name in [n for n in dir(qc) if not n.startswith("__")]:
        if hasattr(SaQC, name):  # skip class attributes
            continue
        assert name in SaQC._attributes


def test_copy(data):
    qc = SaQC(data)
    qc = qc.flagRange("var1").flagRange("var1", min=0, max=0)

    deep = qc.copy(deep=True)
    shallow = qc.copy(deep=False)
    for copy in [deep, shallow]:
        assert copy is not qc
        for name in [n for n in dir(qc) if not n.startswith("__")]:
            if hasattr(SaQC, name):  # skip class attributes
                continue
            qc_attr = getattr(qc, name)
            other_attr = getattr(copy, name)
            assert qc_attr is not other_attr

    # History is always copied
    assert deep._flags._data["var1"] is not qc._flags._data["var1"]
    assert shallow._flags._data["var1"] is not qc._flags._data["var1"]

    # underling data NOT copied
    assert shallow._data["var1"] is qc._data["var1"]

    # underling data copied
    assert deep._data["var1"] is not qc._data["var1"]


def test_sourceTargetCopy():
    """
    test implicit copies
    """
    data = initData(1)
    var1 = data.columns[0]
    target = "new"

    @register(mask=["field"], demask=["field"], squeeze=["field"], handles_target=False)
    def flagTarget(saqc, field, **kwargs):
        assert "target" not in kwargs
        return saqc

    qc = SaQC(data, flags=Flags({var1: pd.Series(127.0, index=data[var1].index)}))
    qc = qc.flagTarget(field=var1, target=target)

    assert (qc.data[var1] == qc.data[target]).all(axis=None)
    assert all(qc.flags[var1] == qc.flags[target])


def test_sourceTargetNoCopy():
    """
    test bypassing of the imlpicit copy machiners
    """
    data = initData(1)
    var1 = data.columns[0]
    target = "new"

    @register(mask=["field"], demask=["field"], squeeze=["field"], handles_target=True)
    def flagField(saqc, field, **kwargs):
        assert "target" in kwargs
        assert "target" not in saqc._data
        assert "target" not in saqc._flags
        return saqc

    SaQC(data).flagField(field=var1, target=target)


def test_sourceTargetMultivariate():
    """
    test bypassing of the imlpicit copy machinery
    """
    data = initData(3)

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        handles_target=True,
        multivariate=True,
    )
    def flagMulti(saqc, field, **kwargs):
        assert "target" in kwargs
        assert "target" not in saqc._data
        assert "target" not in saqc._flags
        assert field == kwargs["target"]
        return saqc

    SaQC(data).flagMulti(field=data.columns, target=data.columns)


def test_sourceTargetMulti():
    data = initData(3)
    flags = initFlagsLike(data)
    fields = data.columns
    targets = [f"target{i + 1}" for i in range(len(fields))]

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        handles_target=True,
        multivariate=True,
    )
    def flagMulti(saqc, field, target, **kwargs):
        assert len(field) == len(target)
        for src, trg in zip(field, target):
            assert src in saqc._data
            assert src in saqc._flags
            assert trg not in saqc._data
            assert trg not in saqc._flags
        return saqc

    SaQC(data, flags).flagMulti(field=fields, target=targets)


def test_unknown_attribute():
    qc = SaQC()
    with pytest.raises(AttributeError):
        qc._construct(_spam="eggs")


def test_validation(data):
    """Test if validation detects different columns in data and flags."""
    df = pd.DataFrame(
        data=np.arange(8).reshape(4, 2),
        index=pd.date_range("2020", None, 4, "1d"),
        columns=list("ab"),
    )
    qc = SaQC(df)

    @flagging()
    def flagFoo(saqc, field, **kwargs):
        saqc._data["spam"] = saqc._data[field]
        return saqc

    with pytest.raises(RuntimeError):
        qc.flagFoo("a")


def test__copy__():
    orig = SaQC()
    orig.attrs["spam"] = []  # a higher object
    shallow = copy.copy(orig)
    assert shallow is not orig
    assert shallow.attrs["spam"] is orig.attrs["spam"]


def test__deepcopy__():
    orig = SaQC()
    orig.attrs["spam"] = []  # a higher object
    shallow = copy.deepcopy(orig)
    assert shallow is not orig
    assert shallow.attrs["spam"] is not orig.attrs["spam"]


def test_immutability(data):
    field = data.columns[0]
    saqc_before = SaQC(data)
    saqc_after = saqc_before.flagDummy(field)
    for name in SaQC._attributes:
        assert getattr(saqc_before, name) is not getattr(saqc_after, name)


@pytest.mark.parametrize(
    "field,target",
    [
        (["a"], ["x", "y"]),
        (["a", "b"], ["x"]),
    ],
)
def test_fieldsTargetsExpansionFail(field, target):
    # check that the field/target handling works as expected for the
    # different function types

    @register(mask=[], demask=[], squeeze=[], multivariate=False, handles_target=False)
    def foo(saqc, field, **kwargs):
        return saqc

    @register(mask=[], demask=[], squeeze=[], multivariate=False, handles_target=True)
    def bar(saqc, field, target, **kwargs):
        return saqc

    @register(mask=[], demask=[], squeeze=[], multivariate=True, handles_target=False)
    def baz(saqc, field, **kwargs):
        return saqc

    @register(mask=[], demask=[], squeeze=[], multivariate=False, handles_target=True)
    def fooBar(saqc, field, **kwargs):
        return saqc

    data = pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4], "d": [4, 5]})
    qc = SaQC(data)
    with pytest.raises(ValueError):
        qc.foo(field, target=target)
    with pytest.raises(ValueError):
        qc.bar(field, target=target)
    with pytest.raises(ValueError):
        qc.baz(field, target=target)
    with pytest.raises(ValueError):
        qc.fooBar(field, target=target)


@pytest.mark.parametrize(
    "field,target",
    [
        (["a"], ["x"]),
        (["a", "a"], ["x", "y"]),
    ],
)
def test_fieldsTargetsExpansion(field, target):
    # check that the field/target handling works as expected for the
    # different function types

    @register(mask=[], demask=[], squeeze=[], multivariate=False, handles_target=False)
    def foo(saqc, field, **kwargs):
        return saqc

    @register(mask=[], demask=[], squeeze=[], multivariate=False, handles_target=True)
    def bar(saqc, field, target, **kwargs):
        return saqc

    @register(mask=[], demask=[], squeeze=[], multivariate=True, handles_target=False)
    def baz(saqc, field, **kwargs):
        return saqc

    @register(mask=[], demask=[], squeeze=[], multivariate=False, handles_target=True)
    def fooBar(saqc, field, **kwargs):
        return saqc

    data = pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4], "d": [4, 5]})
    qc = SaQC(data)
    qc.foo(field, target=target)
    qc.bar(field, target=target)
    qc.baz(field, target=target)
    qc.fooBar(field, target=target)


@pytest.mark.parametrize(
    "field,target",
    [
        (["a"], ["x"]),
        (["a", "a"], ["x", "y"]),
        (["a"], ["x", "y"]),
        (["a", "a"], ["x"]),
    ],
)
def test_fieldsTargetsExpansionMultivariate(field, target):
    @register(mask=[], demask=[], squeeze=[], multivariate=True, handles_target=True)
    def foo(saqc, field, target, **kwargs):
        return saqc

    @register(mask=[], demask=[], squeeze=[], multivariate=True, handles_target=True)
    def bar(saqc, field, **kwargs):
        return saqc

    data = pd.DataFrame(
        {"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5], "d": [4, 5, 6]}
    )
    qc = SaQC(data)
    qc.foo(field, target)
    qc.bar(field, target)


def test_columnConsitency(data):
    @flagging()
    def flagFoo(saqc, field, **kwargs):
        saqc._flags["spam"] = saqc._flags[field]
        return saqc

    field = data.columns[0]
    qc = SaQC(data)
    with pytest.raises(RuntimeError):
        qc.flagFoo(field)


@pytest.mark.parametrize(
    "user_flag,internal_flag",
    (
        [FILTER_ALL, FILTER_ALL],
        [FILTER_NONE, FILTER_NONE],
        [OptionalNone(), FILTER_ALL],
        ["BAD", BAD],
        ["UNFLAGGED", UNFLAGGED],
    ),
)
def test_dfilterTranslation(data, user_flag, internal_flag):
    @flagging()
    def flagFoo(saqc, field, dfilter, **kwargs):
        assert dfilter == internal_flag
        return saqc

    field = data.columns[0]
    qc = SaQC(data, scheme="simple")
    qc.flagFoo(field, dfilter=user_flag)


@pytest.mark.parametrize(
    "data, expected",
    [
        # 2c + 1c -> 3c
        (
            [
                DictOfSeries(a=pd.Series([1]), b=pd.Series([2])),
                DictOfSeries(c=pd.Series([3])),
            ],
            DictOfSeries(a=pd.Series([1]), b=pd.Series([2]), c=pd.Series([3])),
        ),
        # 1c + 1c + 1c -> 3c
        (
            [
                DictOfSeries(a=pd.Series([1])),
                DictOfSeries(b=pd.Series([2])),
                DictOfSeries(c=pd.Series([3])),
            ],
            DictOfSeries(a=pd.Series([1]), b=pd.Series([2]), c=pd.Series([3])),
        ),
    ],
)
def test_concatDios(data, expected):
    result = SaQC(data)
    assert result.data == expected


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            [
                DictOfSeries(a=pd.Series([1]), b=pd.Series([2])),
                DictOfSeries(b=pd.Series([99])),
            ],
            DictOfSeries(a=pd.Series([1]), b=pd.Series([99])),
        )
    ],
)
def test_concatDios_warning(data, expected):
    with pytest.warns(UserWarning):
        result = SaQC(data)
    assert result.data == expected


def test_atomicWrite():
    def writeAndFail(obj):
        obj._data["foo"] = pd.Series()
        raise ValueError("ups i forgot flags")

    orig = SaQC(pd.DataFrame([1, 2, 3], columns=["a"]))

    # test that atomic write is needed
    qc = orig.copy()
    try:
        writeAndFail(qc)
    except ValueError:
        pass
    assert not qc.columns.equals(orig.columns)  # data changed
    assert qc.data.keys() != qc.flags.keys()  # invariant broken

    # use qc._atomicWrite
    qc = orig.copy()
    try:
        with qc._atomicWrite():
            writeAndFail(qc)
    except ValueError:
        pass
    assert qc.columns.equals(orig.columns)  # data ist still unchanged
    assert qc.data.keys() == qc.flags.keys()  # invariant holds


@pytest.mark.parametrize(
    "columns,key,expected",
    [
        (["a", "b", "c"], ["a", "c"], pd.Index(["a", "c"])),
        (["a", "b", "c"], "a", pd.Index(["a"])),
        (["a", "b", "c", "d", "e"], slice("b", "d"), pd.Index(["b", "c", "d"])),
        (["a", "b", "c"], slice(None), pd.Index(["a", "b", "c"])),
        # empty selection
        (["a", "b", "c"], [], pd.Index([])),
        (["a", "b", "c"], slice("b", "a"), pd.Index([])),
    ],
)
def test__getitem__(columns, key, expected):
    data = [pd.Series(range(3))] * len(columns)
    qc = SaQC(dict(zip(columns, data)))
    result = qc[key]
    assert isinstance(result, SaQC)
    assert result.columns.equals(expected)


def test__getitem__duplicateKey():
    qc = SaQC(dict(zip(["a", "b", "c"], [pd.Series(range(3))] * 3)))
    with pytest.raises(NotImplementedError):
        qc[["a", "a"]]  # noqa


def test__setitem__duplicateKey():
    qc = SaQC(dict(zip(["a", "b", "c"], [pd.Series(range(3))] * 3)))
    with pytest.raises(NotImplementedError):
        qc[["a", "a"]] = SaQC(pd.DataFrame(dict(a=[1, 1], b=[2, 2])))


@pytest.mark.parametrize(
    "value",
    [
        SaQC(pd.DataFrame(dict(a=[1, 2, 3]))),
        pd.Series([1, 2, 3]),
        pd.DataFrame(dict(a=[1, 2, 3])),
        DictOfSeries(dict(a=pd.Series([1, 2, 3]))),
        dict(a=pd.Series([1, 2, 3])),
        [pd.Series([1, 2, 3])],
        (s for s in [pd.Series([1, 2, 3])]),
    ],
)
def test__setitem__value_types(value):
    qc = SaQC()
    qc["x"] = value
    assert qc.data["x"].equals(pd.Series([1, 2, 3]))


@pytest.mark.parametrize(
    "data,key,value,expected",
    [
        (
            # insert a single series (data only)
            SaQC(pd.DataFrame(dict(a=[1, 1]))),
            "c",
            pd.Series([8, 8]),
            SaQC(
                pd.DataFrame(dict(a=[1, 2], c=[8, 8])),
                pd.DataFrame(dict(a=[U, U], c=[U, U])),
            ),
        ),
        (
            # insert data and flags by using SaQC obj
            SaQC(pd.DataFrame(dict(a=[1, 1]))),
            "c",
            SaQC(pd.DataFrame(dict(new=[8, 8])), pd.DataFrame(dict(new=[B, B]))),
            SaQC(
                pd.DataFrame(dict(a=[1, 2], c=[8, 8])),
                pd.DataFrame(dict(a=[U, U], c=[B, B])),
            ),
        ),
        (
            # empty key
            SaQC(pd.DataFrame(dict(a=[1, 2], b=[1, 2]))),
            [],
            SaQC(),
            SaQC(
                pd.DataFrame(dict(a=[1, 2], b=[8, 8])),
                pd.DataFrame(dict(a=[U, U], b=[U, U])),
            ),
        ),
        (
            # overwrite single key
            SaQC(pd.DataFrame(dict(a=[1, 2], b=[1, 2]))),
            "b",
            SaQC(pd.DataFrame(dict(new=[8, 8])), pd.DataFrame(dict(new=[B, B]))),
            SaQC(
                pd.DataFrame(dict(a=[1, 2], b=[8, 8])),
                pd.DataFrame(dict(a=[U, U], b=[B, B])),
            ),
        ),
        (
            # overwrite multi key
            SaQC(pd.DataFrame(dict(a=[1, 2], b=[1, 2]))),
            ["a", "b"],
            SaQC(
                pd.DataFrame(dict(new1=[8, 8], new2=[9, 9])),
                pd.DataFrame(dict(new1=[B, B], new2=[B, B])),
            ),
            SaQC(
                pd.DataFrame(dict(a=[8, 8], b=[9, 9])),
                pd.DataFrame(dict(a=[B, B], b=[B, B])),
            ),
        ),
        (
            # overwrite and insert
            SaQC(pd.DataFrame(dict(a=[1, 2]))),
            ["a", "b"],
            SaQC(
                pd.DataFrame(dict(new1=[8, 8], new2=[9, 9])),
                pd.DataFrame(dict(new1=[B, B], new2=[B, B])),
            ),
            SaQC(
                pd.DataFrame(dict(a=[8, 8], b=[9, 9])),
                pd.DataFrame(dict(a=[B, B], b=[B, B])),
            ),
        ),
    ],
)
def test__setitem__(data, key, value, expected):
    data[key] = value
    assert data.columns.equals(expected.columns)

    # slice to list magic
    if isinstance(key, slice):
        key = pd.Series(index=list("abcdef"))[key].index.tolist()

    for k in [key] if isinstance(key, str) else key:
        assert data.data[k].equals(expected.data[k])
        assert data.flags[k].equals(expected.flags[k])
