#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
import ast
import io
import json
import logging
import os.path
import warnings
from textwrap import indent
from typing import Any, Dict, Iterable, List, Sequence, TextIO, Tuple, Generic, TypeVar, \
    cast
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd

from saqc import SaQC
from saqc.exceptions import ParsingError
from saqc.lib.tools import isQuoted
from saqc.parsing.visitor import ConfigFunctionParser


def fromConfig(fname, *args, **func_kwargs):
    return CsvReader(fname).read().parse().run(SaQC(*args, **func_kwargs))

class _ConfigReader:

    def __init__(self, *args, **kwargs):
        self.qc = SaQC(*args, **kwargs)

    def readString(self, c):
        self.reader = CsvReader(c).read().parse()

    def run(self):
        self.reader.run(self.qc)


class LoggerMixin:
    """
    Adds a logger to the class, named as the qualified name of the
    class. A super call to init is not necessary. Each instance has its
    own logger, but in the logging backend they refer to the very same
    logger, unless an instance sets a new logger with another name.
    """

    logger: logging.Logger

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.__dict__["logger"] = logging.getLogger(cls.__qualname__)
        return obj


def fileExists(path_or_buffer):
    try:
        return os.path.exists(path_or_buffer)
    except (ValueError, TypeError, OSError):
        return False


def getFileExtension(path) -> str:
    """Returns empty string, if no extension present."""
    return str(os.path.splitext(path)[1].strip().lower())


def isUrl(s: str) -> bool:
    try:
        return bool(urlparse(s).scheme)
    except (TypeError, ValueError):
        return False


class ConfigTest:
    def __init__(self, func, kws, src, lineno):
        self.func_name = func
        self.func = getattr(SaQC, self.func_name)
        self.kws = kws
        self.src = src
        self.lineno = lineno

    def __repr__(self):
        src = "" if self.src is None else f"src={self.src},"
        return f"{src} func={self.func_name}, kws={self.kws}"

    def run(self, qc):
        return self.func(qc, **self.kws)


class RawConfigEntry:
    def __init__(self, var: str, functxt: str, src: str, lineno: int | None = None):
        self.var = var
        self.functxt = functxt
        self.src = src
        self.lineno = lineno

    def __repr__(self):
        src = "" if self.src is None else self.src
        return f"{src}: {{{self.var}, {self.functxt}}}"

    def parse(self) -> ConfigTest:
        tree = ast.parse(self.functxt, mode="eval").body
        func, kws = ConfigFunctionParser().parse(tree)

        field = self.var
        if isQuoted(self.var):
            field = field[1:-1]
            kws["regex"] = True

        if "field" in kws:
            kws["target"] = field
        else:
            kws["field"] = field

        return ConfigTest(func, kws, self.src, self.lineno)

EntryT = TypeVar('EntryT', RawConfigEntry , ConfigTest)

class Config(LoggerMixin, Generic[EntryT]):
    def __init__(self, obj: Iterable, src: str | None = None):
        self.src = src
        self.tests : List[EntryT] = []
        self.is_parsed = False
        for args in obj:
            self.tests.append(RawConfigEntry(*args))

    def __repr__(self):
        cname = self.__class__.__qualname__
        src = f"({self.src!r})" if self.src else ""
        if not self.tests:
            return f"Empty {cname}{src}"
        tests = "\n".join(["[", *[indent(repr(t), " ") for t in self.tests], "]"])
        return f"{cname}{src}\n{tests}\n"

    def __iter__(self) -> EntryT:
        yield from self.tests

    def __getitem__(self, item):
        return self.tests[item]

    def parse(self) -> Config[ConfigTest]:
        # parse all tests or none (on Error)
        if self.is_parsed:
            raise RuntimeError("config is already parsed")
        tests = self.tests.copy()
        for i in range(len(tests)):
            test: RawConfigEntry = tests[i]
            try:
                tests[i] = test.parse()
            except Exception as e:
                raise ConfigParseException(test, e, i) from None
        self.tests = tests
        self.is_parsed = True
        return self

    def run(self, qc: SaQC) -> SaQC:
        if not self.is_parsed:
            raise RuntimeError("config must be parsed first")
        for i in range(len(self.tests)):
            test: ConfigTest = self.tests[i]
            try:
                qc = test.run(qc)
            except Exception as e:
                raise ConfigRunException(test, e, i).with_traceback(
                    e.__traceback__
                ) from None
        return qc



class ConfigRunException(RuntimeError):
    def __init__(
        self,
        test: ConfigTest | RawConfigEntry,
        orig_e: Exception,
        testno: int | None = None,
    ):
        self.src = test.src
        self.lineno = test.lineno
        self.func = None
        if isinstance(test, ConfigTest):
            self.func = test.func_name
        if isinstance(test, RawConfigEntry):
            self.func = test.functxt

        self.ex_typ = type(orig_e).__name__
        self.ex_msg = str(orig_e)
        self.testno = testno

    def __str__(self):
        lno = "" if self.lineno is None else f", line {self.lineno}"
        return (
            f"\n"
            f"  config:       {self.src}{lno}\n"
            f"  test number:  {self.testno}\n"
            f"  saqc-func:    {self.func}\n"
            f"  Exception:    {self.ex_typ}: {self.ex_msg}\n"
        )

class ConfigParseException(ConfigRunException):
    pass

def isOpenFileLike(obj) -> bool:
    return (
        isinstance(obj, io.IOBase) or hasattr(obj, "read") and hasattr(obj, "readlines")
    )


class Reader(abc.ABC, LoggerMixin):
    _supported_file_extensions = tuple()

    def __init__(self, path_or_buffer):
        try:
            ext = getFileExtension(path_or_buffer)
        except (ValueError, TypeError):
            ext = None
        self.file_ext = ext or None
        self.logger.debug(f"{self.file_ext=}")

        if isUrl(path_or_buffer):
            data = urlopen(path_or_buffer).read().decode("utf-8")
            src = path_or_buffer
        elif isOpenFileLike(path_or_buffer):
            data = path_or_buffer.read()
            # io.StringIO has no name attribute
            src = getattr(path_or_buffer, "name", "unknown file object")
        elif (
            self.file_ext is not None
            and self.file_ext in self._supported_file_extensions
            and not fileExists(path_or_buffer)
        ):
            raise FileNotFoundError(f"No such file {path_or_buffer!r}")
        elif fileExists(path_or_buffer):
            with open(path_or_buffer, "r") as fh:
                data = fh.read()
            src = path_or_buffer
        else:  # input string
            data = path_or_buffer
            src = "input string"
        if not isinstance(data, str):
            raise TypeError(f"unsupported type {type(path_or_buffer)}")

        self.data = data
        self.src = src
        self._maybeWarn()

    def _maybeWarn(self):
        if (
            self.file_ext is not None
            and self.file_ext not in self._supported_file_extensions
        ):
            warnings.warn(
                f"File extension is {self.file_ext!r} but the reader "
                f"expects one of {self._supported_file_extensions}",
                category=RuntimeWarning,
                stacklevel=4,  # at call to SomeReader
            )

    @abc.abstractmethod
    def read(self) -> Config[RawConfigEntry]:
        ...


class CsvReader(Reader):
    _supported_file_extensions = (".csv",)

    def __init__(self, path_or_buffer, header=1, comment="#", sep=";"):
        super().__init__(path_or_buffer)
        self.sep = sep
        self.header = header
        self.comment = comment
        if self.src is None:
            self.src = ""
            pass

    def read(self):
        entries = []
        skip = max(self.header or 0, 0) + 1
        comment = self.comment or None
        for i, line in enumerate(self.data.splitlines()):
            lineno = i + 1
            if (skip := skip - 1) > 0:
                continue
            line: str = line.strip()
            if comment is not None and line.startswith(comment):
                continue
            parts = [p.strip() for p in line.split(sep=self.sep)]
            if (n := len(parts)) != 2:
                raise ParsingError(
                    f"The configuration format expects exactly two "
                    f"columns, one for the variable name and one for "
                    f"the tests, but {n} columns were found in line "
                    f"{lineno}.\n\t{line!r}"
                )
            entries.append(parts + [self.src, lineno])

        return Config(entries, src=self.src)


class JsonReader(Reader):
    _supported_file_extensions = (".json",)

    def __init__(self, path_or_buffer, root_key: str | None = None):
        super().__init__(path_or_buffer)
        self.root_key = root_key

    def read(self):
        d = json.loads(self.data)
        if self.root_key is not None:
            d = d[self.root_key]

        if not isinstance(d, list):
            msg = "Expected a json array"
            if self.root_key is not None:
                msg += f" under key {self.root_key}"
            if isinstance(d, dict):
                msg += f", but got a object with keys {d.keys()}"
                if self.root_key is None:
                    msg += ". Maybe use a key as root_key."
            raise ValueError(msg)
        df = pd.DataFrame(d)

        if not df.columns.equals(pd.Index(["varname", "function", "kwargs"])):
            raise ValueError(
                f'expected fields "varname", "function" and "kwargs" '
                f"for each test but got {set(df.columns)}"
            )

        for c in ["varname", "function", "kwargs"]:
            missing = list(df.loc[df[c].isna(), c].index + 1)
            if missing:
                raise ValueError(f"Tests {missing} have no {c!r} entry")

        # todo: maybe try out pypi package `json_source_map`
        #       to get line numbers
        df['lineno'] = None
        df["src"] = self.src

        kws = df["kwargs"].apply(
            lambda e: ", ".join([f"{k}={v}" for k, v in e.items()])
        )
        df["test"] = df["function"] + "(" + kws + ")"
        df = df[
            [
                "varname",
                "test",
                "src",
                "lineno",
            ]
        ]
        return Config(df.itertuples(index=False), src=self.src)


if __name__ == "__main__":
    path0 = "/home/palmb/projects/saqc/ignore/ressources/config.json"
    path1 = "/home/palmb/projects/saqc/ignore/ressources/configArr.json"
    path3 = "/home/palmb/projects/saqc/ignore/ressources/config.csv"

    logging.basicConfig(level="DEBUG")
    config = JsonReader(path0, root_key="tests").read()
    config.parse()
    config = CsvReader(path3).read()
    qc = SaQC(dict(SM4=pd.Series([1, 2, 3]), SM2=pd.Series([2, 3, 4, 5])))
    config.parse().run(qc)
    # config[1].parse().run(SaQC())
    exit(99)
