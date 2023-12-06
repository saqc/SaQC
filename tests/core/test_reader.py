#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pytest

from saqc.core import DictOfSeries, Flags, SaQC, flagging
from saqc.exceptions import ParsingError
from saqc.parsing.environ import ENVIRONMENT
from saqc.parsing.reader import CsvReader
from tests.common import initData


@pytest.fixture
def data() -> DictOfSeries:
    return initData(3)


def getTestedVariables(flags: Flags, test: str):
    out = []
    for col in flags.columns:
        for m in flags.history[col].meta:
            if m["func"] == test:
                out.append(col)
    return out


@pytest.mark.parametrize(
    "row,expected",
    [
        ("'.*'       ; flagDummy()", ["var1", "var2", "var3"]),
        ("'var(1|2)' ; flagDummy()", ["var1", "var2"]),
        ("'var[12]'  ; flagDummy()", ["var1", "var2"]),
        ("'.*3'      ; flagDummy()", ["var3"]),
    ],
)
def test_variableRegex(data, row, expected):
    qc = SaQC(data)
    saqc = CsvReader(row).read().parse().run(qc)
    result = getTestedVariables(saqc._flags, "flagDummy")
    assert np.all(result == expected)


@pytest.mark.parametrize(
    # not quoted -> no regex
    "row,expected",
    [("var[12]  ; flagDummy()", [])],
)
def test_variableNoRegexWarning(data, row, expected):
    qc = SaQC(data)
    with pytest.warns(RuntimeWarning):
        qc = CsvReader(row).read().parse().run(qc)
    result = getTestedVariables(qc._flags, "flagDummy")
    assert np.all(result == expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_inlineComments(data):
    """
    adresses issue #3
    """
    config = f"""
    varname ; test
    var1    ; flagDummy() # test
    """

    qc = CsvReader(config).read().parse().run(SaQC(data))
    func = qc._flags.history["var1"].meta[0]["func"]
    assert func == "flagDummy"


def test_configReaderLineNumbers():
    config = f"""
    varname ; test
    #temp1      ; flagDummy()
    pre1        ; flagDummy()
    pre2        ; flagDummy()
    SM          ; flagDummy()
    #SM         ; flagDummy()
    # SM1       ; flagDummy()

    SM1         ; flagDummy()
    """
    cnf = CsvReader(config).read()
    linenos = [test.lineno for test in cnf]
    expected = [4, 5, 6, 10]
    assert linenos == expected


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_configFile(data):
    # check that the reader accepts different whitespace patterns

    config = f"""
    varname ; test

    #temp1      ; flagDummy()
    pre1; flagDummy()
    pre2        ;flagDummy()
    SM          ; flagDummy()
    #SM         ; flagDummy()
    # SM1       ; flagDummy()

    SM1;flagDummy()
    """
    cnf = CsvReader(config).read()
    assert len(cnf) == 4


@pytest.mark.parametrize(
    "test, expected",
    [
        (f"var1; min", ParsingError),  # not a function call
        (f"var3; callUnknown()", NameError),  # unknown function
        (f"var1; flagFunc(mn=0)", TypeError),  # bad argument name
        (f"var1; flagFunc()", TypeError),  # not enough arguments
    ],
)
def test_configChecks(data, test, expected):
    @flagging()
    def flagFunc(data, field, flags, arg, opt_arg=None, **kwargs):
        flags[:, field] = np.nan
        return data, flags

    with pytest.raises(expected):
        CsvReader(test).read().parse().run(SaQC(data))


@pytest.mark.parametrize(
    "kwarg, expected",
    [
        ("NAN", np.nan),
        ("nan", np.nan),
        ("inf", np.inf),
        ("'a string'", "a string"),
        ("5", 5),
        ("5.5", 5.5),
        ("-5", -5),
        ("True", True),
        ("sum([1, 2, 3])", 6),
    ],
)
def test_supportedArguments(data, kwarg, expected):
    # test if the following function arguments
    # are supported (i.e. parsing does not fail)

    @flagging()
    def func(saqc, field, x, **kwargs):
        if kwarg.lower() == "nan":
            assert np.isnan(x)
        else:
            assert x == expected
        saqc.func_was_called = True
        return saqc

    qc = CsvReader(f"var1; func(x={kwarg})").read().parse().run(SaQC(data))
    assert qc.func_was_called


@pytest.mark.parametrize(
    "func_string", [k for k, v in ENVIRONMENT.items() if callable(v)]
)
def test_funtionArguments(data, func_string):
    @flagging()
    def testFunction(saqc, field, func, **kwargs):
        assert func is ENVIRONMENT[func_string]
        saqc.func_was_called = True
        return saqc

    config = f"""
    varname ; test
    {data.columns[0]} ; testFunction(func={func_string})
    {data.columns[0]} ; testFunction(func="{func_string}")
    """
    qc = CsvReader(config).read().parse().run(SaQC(data))
    assert qc.func_was_called
