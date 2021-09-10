#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
import dios
from pathlib import Path

from saqc.core.reader import fromConfig, readFile
from saqc.core.register import FUNC_MAP, flagging
from saqc.constants import UNTOUCHED

from tests.common import initData, writeIO


@pytest.fixture
def data() -> dios.DictOfSeries:
    return initData(3)


def test_packagedConfig():

    path = Path(__file__).parents[2] / "ressources/data"

    config_path = path / "config_ci.csv"
    data_path = path / "data.csv"

    data = pd.read_csv(
        data_path,
        index_col=0,
        parse_dates=True,
    )
    saqc = fromConfig(config_path, data)
    saqc.getResult()


def test_variableRegex(data):

    header = f"varname;test"
    tests = [
        ("'.*'", data.columns),
        ("'var(1|2)'", [c for c in data.columns if c[-1] in ("1", "2")]),
        ("'var[12]'", [c for c in data.columns if c[-1] in ("1", "2")]),
        ("var[12]", ["var[12]"]),  # not quoted -> not a regex
        ('".*3"', [c for c in data.columns if c[-1] == "3"]),
    ]

    for regex, expected in tests:
        fobj = writeIO(header + "\n" + f"{regex} ; flagtools.flagDummy()")
        saqc = fromConfig(fobj, data=data, lazy=True)
        result = [s.field for s, _ in saqc._planned]
        assert np.all(result == expected)


def test_inlineComments(data):
    """
    adresses issue #3
    """
    config = f"""
    varname ; test
    pre2        ; flagtools.flagDummy() # test
    """

    saqc = fromConfig(writeIO(config), data, lazy=True)
    _, func = saqc._planned[0]
    assert func.func == FUNC_MAP["flagtools.flagDummy"].func


def test_configReaderLineNumbers(data):
    config = f"""
    varname ; test
    #temp1      ; flagtools.flagDummy()
    pre1        ; flagtools.flagDummy()
    pre2        ; flagtools.flagDummy()
    SM          ; flagtools.flagDummy()
    #SM         ; flagtools.flagDummy()
    # SM1       ; flagtools.flagDummy()

    SM1         ; flagtools.flagDummy()
    """
    planned = readFile(writeIO(config))
    expected = [4, 5, 6, 10]
    assert (planned.index == expected).all()


def test_configFile(data):

    # check that the reader accepts different whitespace patterns

    config = f"""
    varname ; test

    #temp1      ; flagtools.flagDummy()
    pre1; flagtools.flagDummy()
    pre2        ;flagtools.flagDummy()
    SM          ; flagtools.flagDummy()
    #SM         ; flagtools.flagDummy()
    # SM1       ; flagtools.flagDummy()

    SM1;flagtools.flagDummy()
    """
    fromConfig(writeIO(config), data)


def test_configChecks(data):

    var1, _, var3, *_ = data.columns

    @flagging(masking="none")
    def flagFunc(data, field, flags, arg, opt_arg=None, **kwargs):
        flags[:, field] = UNTOUCHED
        return data, flags

    header = f"varname;test"
    tests = [
        (f"{var1};flagFunc(mn=0)", TypeError),  # bad argument name
        (f"{var1};flagFunc()", TypeError),  # not enough arguments
        (f"{var3};flagNothing()", NameError),  # unknown function
        (f"{var1}; min", TypeError),  # not a function call
    ]

    for test, expected in tests:
        fobj = writeIO(header + "\n" + test)
        with pytest.raises(expected):
            fromConfig(fobj, data=data).evaluate()


def test_supportedArguments(data):

    # test if the following function arguments
    # are supported (i.e. parsing does not fail)

    # TODO: necessary?

    @flagging(masking="field")
    def func(data, field, flags, kwarg, **kwargs):
        flags[:, field] = UNTOUCHED
        return data, flags

    var1 = data.columns[0]

    header = f"varname;test"
    tests = [
        f"{var1};func(kwarg=NAN)",
        f"{var1};func(kwarg='str')",
        f"{var1};func(kwarg=5)",
        f"{var1};func(kwarg=5.5)",
        f"{var1};func(kwarg=-5)",
        f"{var1};func(kwarg=True)",
        f"{var1};func(kwarg=sum([1, 2, 3]))",
    ]

    for test in tests:
        fobj = writeIO(header + "\n" + test)
        fromConfig(fobj, data)
