#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
import ast
import json
import logging
import warnings
from textwrap import indent
from typing import Generic, Iterable, List, TypeVar
from urllib.request import urlopen

import pandas as pd

from saqc import SaQC
from saqc.exceptions import ParsingError, _SpecialKeyError
from saqc.lib.checking import isOpenFileLike, isUrl
from saqc.lib.tools import LoggerMixin, fileExists, getFileExtension, isQuoted
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


def _formatSrc(src: str | None, lineno: int | None, long: bool = False):
    if src is None and lineno is None:
        return ""
    if src is None:
        return f"line {lineno}"
    if lineno is None:
        return f"{src}"
    if long:
        return f"{src}, line {lineno}"
    return f"{src}:{lineno}"


class ConfigTest:
    def __init__(self, func, kws, src, lineno):
        self.func_name = func
        self.func = getattr(SaQC, self.func_name)
        self.kws = kws
        self.src = src
        self.lineno = lineno

    def __repr__(self):
        src = _formatSrc(self.src, self.lineno)
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
        src = _formatSrc(self.src, self.lineno)
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


EntryT = TypeVar("EntryT", RawConfigEntry, ConfigTest)


class Config(LoggerMixin, Generic[EntryT]):
    tests: List[EntryT]

    def __init__(self, obj: Iterable, src: str | None = None):
        self.src = src
        self.tests: List[RawConfigEntry] = []
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

    def __getitem__(self, item) -> EntryT:
        return self.tests[item]

    def parse(self) -> Config[ConfigTest]:
        if self.is_parsed:
            raise RuntimeError("config is already parsed")
        parsed: List[ConfigTest] = []
        for i, test in enumerate(self.tests):
            try:
                parsed.append(test.parse())
            except Exception as e:
                msg = self.formatErrMessage(test, e, "Parsing failed")
                raise ParsingError(msg) from None
        self.tests = parsed
        self.is_parsed = True
        return self

    def run(self, qc: SaQC) -> SaQC:
        if not self.is_parsed:
            raise RuntimeError("config must be parsed first")
        for i, test in enumerate(self.tests):
            assert isinstance(test, ConfigTest)
            try:
                qc = test.run(qc)
            except KeyError as e:
                # We need to handle KeyError differently, because
                # it uses `repr` instead of `str` and would mess
                # up our message with extra text and newlines.
                msg = self.formatErrMessage(test, e, "Executing config failed")
                raise _SpecialKeyError(msg).with_traceback(e.__traceback__) from None
            except Exception as e:
                msg = self.formatErrMessage(test, e, "Executing config failed")
                raise type(e)(msg).with_traceback(e.__traceback__) from None
        return qc

    @staticmethod
    def formatErrMessage(test: EntryT, e: Exception, message: str = "") -> str:
        exc_typ = type(e).__name__
        exc_msg = str(e)
        src = _formatSrc(test.src, test.lineno, long=True)
        if isinstance(test, ConfigTest):
            return (
                f"{message}\n"
                f"  config:     {src}\n"
                f"  SaQC-func:  {test.func_name}\n"
                f"  kwargs:     {test.kws}\n"
                f"  Exception:  {exc_typ}: {exc_msg}\n"
            )
        if isinstance(test, RawConfigEntry):
            return (
                f"{message}\n"
                f"  config:     {src}\n"
                f"  varname:    {test.var}\n"
                f"  test:       {test.functxt}\n"
                f"  Exception:  {exc_typ}: {exc_msg}\n"
            )
        else:
            raise TypeError(f"{test=}")


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
            src = getattr(path_or_buffer, "name", "nameless file-object")
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
            src = "string-config"
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

        # This is more or less a hack to use our custom Parser, because
        # parsing just the special keyword func='some-str-expr' is currently
        # not supported. Most of the args are already correctly parsed to
        # python objects by the json read above, but we transform everything
        # back to a string function `func_name(kw0=val0, ...)` for our Parser.
        to_str = lambda e: ", ".join([f"{k}={v}" for k, v in e.items()])
        kws = df["kwargs"].apply(to_str)
        df["test"] = df["function"] + "(" + kws + ")"

        # todo: maybe we should try out pypi package
        #   `json_source_map` to get line numbers
        df["lineno"] = None
        df["src"] = self.src
        df = df[["varname", "test", "src", "lineno"]]
        return Config(df.itertuples(index=False), src=self.src)


if __name__ == "__main__":
    path0 = "/home/palmb/projects/saqc/ignore/ressources/config.json"
    path1 = "/home/palmb/projects/saqc/ignore/ressources/configArr.json"
    path3 = "/home/palmb/projects/saqc/ignore/ressources/config.csv"
    with open(path3) as f:
        spath = f.read()
    # path3 = spath

    logging.basicConfig(level="DEBUG")
    config = JsonReader(path0, root_key="tests").read()
    # print(config)
    # print(config.parse())
    config = CsvReader(path3).read()
    qc = SaQC(dict(SM4=pd.Series([1, 2, 3]), SM2=pd.Series([2, 3, 4, 5])))
    config.parse().run(qc)
    # config[1].parse().run(SaQC())
    exit(99)
