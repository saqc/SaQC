#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import io
import json
import logging
import textwrap
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TextIO, Tuple
from urllib.request import urlopen

import pandas as pd

from saqc import SaQC
from saqc.exceptions import ParsingError
from saqc.lib.tools import isQuoted
from saqc.parsing.visitor import ConfigFunctionParser


class Explanation(Exception):
    def __str__(self):
        return f"\n{self.args[0]}\n\n>> See the Exception above this one, for the real reason. <<"


def _add_note(e: Exception, note: str) -> Exception:
    if hasattr(e, "add_note"):  # python 3.11+
        e.add_note(note)
        return e
    if len(e.args) == 1 and isinstance(e.args[0], str):
        args = f"{e}\n{note}"
    else:
        args = *e.args, f"\n{note}"
    return type(e)(*args).with_traceback(e.__traceback__)


def _readLines(
    it: Iterable[str], column_sep=";", comment_prefix="#", skip=0
) -> pd.DataFrame:
    out = []
    for i, line in enumerate(it):
        if (skip := skip - 1) > 0:
            continue
        if not (row := line.strip().split(comment_prefix, 1)[0]):
            continue
        parts = [p.strip() for p in row.split(column_sep)]
        if len(parts) != 2:
            raise ParsingError(
                f"The configuration format expects exactly two "
                f"columns, one for the variable name and one for "
                f"the tests, but {len(parts)} columns were found "
                f"in line {i}.\n\t{line!r}"
            )
        out.append([i + 1] + parts)
    if not out:
        raise ParsingError("Config file is empty")
    return pd.DataFrame(out[1:], columns=["lineno", "varname", "test"]).set_index(
        "lineno"
    )


def readFile(fname, skip=1) -> pd.DataFrame:
    """Read and parse a config file to a DataFrame"""

    def _open(file_or_buf) -> TextIO:
        if not isinstance(file_or_buf, (str, Path)):
            return file_or_buf
        try:
            fh = io.open(file_or_buf, "r", encoding="utf-8")
        except (OSError, ValueError):
            fh = io.StringIO(urlopen(str(file_or_buf)).read().decode("utf-8"))
            fh.seek(0)
        return fh

    def _close(fh):
        try:
            fh.close()
        except AttributeError:
            pass

    # mimic `with open(): ...`
    file = _open(fname)
    try:
        return _readLines(file, skip=skip)
    finally:
        _close(file)


def fromConfig(fname, *args, **func_kwargs):
    return _ConfigReader(*args, **func_kwargs).readCsv(fname).run()


class _ConfigReader:
    logger: logging.Logger
    saqc: SaQC
    file: str | None
    config: pd.DataFrame | None
    parsed: List[Tuple[Any, ...]] | None
    regex: bool | None
    varname: str | None
    lineno: int | None
    field: str | None
    test: str | None
    func: str | None
    func_kws: Dict[str, Any] | None

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.saqc = SaQC(*args, **kwargs)
        self.file = None
        self.config = None
        self.parsed = None
        self.regex = None
        self.varname = None
        self.lineno = None
        self.field = None
        self.test = None
        self.func = None
        self.func_kws = None

    def readCsv(self, file: str, skip=1):
        self.logger.debug(f"opening csv file: {file}")
        self.config = readFile(file, skip=skip)
        self.file = file
        return self

    def readRecords(self, seq: Sequence[Dict[str, Any]]):
        self.logger.debug(f"read records: {seq}")
        df = pd.DataFrame.from_records(seq)
        df.columns = ["varname", "func", "kwargs"]
        kws = df["kwargs"].apply(
            lambda e: ", ".join([f"{k}={v}" for k, v in e.items()])
        )
        df["test"] = df["func"] + "(" + kws + ")"
        self.config = df.loc[:, ["varname", "test"]].copy()
        return self

    def _readJson(self, d, unpack):
        if unpack is not None:
            d = unpack(d)
        elif isinstance(d, dict):
            raise TypeError("parsed json resulted in a dict, but a array/list is need")
        return self.readRecords(d)

    def readJson(self, file: str, unpack: callable | None = None):
        self.logger.debug(f"opening json file: {file}")
        with open(file, "r") as fh:
            d = json.load(fh)
        self.file = file
        return self._readJson(d, unpack)

    def readJsonString(self, jn: str, unpack: callable | None = None):
        self.logger.debug(f"read json string: {jn}")
        d = json.loads(jn)
        return self._readJson(d, unpack)

    def readString(self, s: str, line_sep="\n", column_sep=";"):
        self.logger.debug(f"read config string: {s}")
        lines = s.split(line_sep)
        self.config = _readLines(lines, column_sep=column_sep)
        return self

    def _parseLine(self):
        self.logger.debug(f"parse line {self.lineno}: {self.varname!r}; {self.test!r}")
        self.regex = isQuoted(self.varname)
        self.field = self.varname[1:-1] if self.regex else self.varname

        try:
            tree = ast.parse(self.test, mode="eval").body
            func, kws = ConfigFunctionParser().parse(tree)
        except Exception as e:
            # We raise a NEW exception here, because the
            # traceback hold no relevant info for a CLI user.
            err = type(e) if isinstance(e, NameError) else ParsingError
            meta = self._getFormattedInfo(
                "The exception occurred during parsing of a config"
            )
            if hasattr(e, "add_note"):  # python 3.11+
                e = err(*e.args)
                e.add_note(meta)
            else:
                e = err(f"{e}\n{meta}")
            raise e from None

        if "field" in kws:
            kws["target"] = self.field
        else:
            kws["field"] = self.field
        self.func = func
        self.func_kws = kws

    def _execLine(self):
        self.logger.debug(
            f"execute line {self.lineno}: {self.varname!r}; {self.test!r}"
        )
        # We explicitly route all function calls through SaQC.__getattr__
        # in order to do a FUNC_MAP lookup. Otherwise, we wouldn't be able
        # to overwrite existing test functions with custom register calls.
        try:
            self.saqc = self.saqc.__getattr__(self.func)(
                regex=self.regex, **self.func_kws
            )
        except Exception as e:
            # We use a special technique for raising here, because we
            # want this location of rising, line up in the traceback,
            # instead of showing up at last (most relevant). Also, we
            # want to include some meta information about the config.
            meta = self._getFormattedInfo(
                "The exception occurred during execution of a config"
            )
            if hasattr(e, "add_note"):  # python 3.11+
                e.add_note(meta)
                raise e
            raise type(e)(str(e) + meta).with_traceback(e.__traceback__) from None

    def _getFormattedInfo(self, msg=None, indent=2):
        prefix = " " * indent
        info = textwrap.indent(
            f"file:    {self.file!r}\n"
            f"line:    {self.lineno}\n"
            f"varname: {self.varname!r}\n"
            f"test:    {self.test!r}\n",
            prefix,
        )
        if msg:
            info = textwrap.indent(f"{msg}\n{info}", prefix)
        return f"\n{info}"

    def run(self):
        """Parse and execute a config line by line."""
        assert self.config is not None
        for lineno, varname, test in self.config.itertuples():
            self.lineno = lineno
            self.varname = varname
            self.test = test
            self._parseLine()
            self._execLine()
        return self.saqc


# #################################################################
# new impl
# #################################################################

import abc
import io
import json
import os.path
from textwrap import indent
from typing import Iterable, overload
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd


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
    def __init__(self, func, kws, src):
        self.funcname = func
        self.func = getattr(SaQC, self.funcname)
        self.kws = kws
        self.src = src

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        src = "" if self.src is None else self.src
        return f"{src}: parsed to: {self.funcname}({self.kws})"

    def run(self, qc):
        # We explicitly route all function calls through SaQC.__getattr__
        # in order to do a FUNC_MAP lookup. Otherwise, we wouldn't be able
        # to overwrite existing test functions with custom register calls.
        try:
            return self.func(qc, **self.kws)
        except Exception as e:
            meta = (
                f"The exception occurred during execution of "
                f"the config line:\n{indent(str(self), '  ')}"
            )
            if hasattr(e, "add_note"):  # python 3.11+
                e.add_note(meta)
                raise e
            raise Explanation(meta) from e


class RawConfigEntry(LoggerMixin):
    def __init__(self, var: str, functxt: str, src: int | None):
        self.var = var
        self.functxt = functxt
        self.src = src

    def __iter__(self):
        yield from [self.var, self.functxt, self.src]

    def __repr__(self):
        src = "" if self.src is None else self.src
        return f"{src}: {{{self.var}, {self.functxt}}}"

    def parse(self) -> ConfigTest:
        field = self.var
        if regex := isQuoted(self.var):
            field = field[1:-1]

        try:
            tree = ast.parse(self.functxt, mode="eval").body
            func, kws = ConfigFunctionParser().parse(tree)
        except Exception as e:
            meta = (
                f"The exception occurred during parsing of "
                f"the config line:\n{indent(str(self), '  ')}"
            )
            # We raise a NEW exception here, because the
            # traceback hold no relevant info for a CLI user.
            e = (type(e) if isinstance(e, NameError) else ParsingError)(*e.args)
            if hasattr(e, "add_note"):  # python 3.11+
                e.add_note(meta)
                raise e
            raise Explanation(meta) from e

        if "field" in kws:
            kws["target"] = field
        else:
            kws["field"] = field

        if regex:
            kws["regex"] = True

        return ConfigTest(func, kws, self.src)


class RawConfig:
    def __init__(self, obj: Iterable, src=None):
        self.src = src
        self.tests = []
        for args in obj:
            self.tests.append(RawConfigEntry(*args))

    def __repr__(self):
        cname = self.__class__.__qualname__
        src = f"({self.src!r})" if self.src else ""
        if not self.tests:
            return f"Empty {cname}{src}"
        tests = "\n".join(["[", *[indent(repr(t), " ") for t in self.tests], "]"])
        return f"{cname}{src}\n{tests}\n"

    def __iter__(self):
        yield from self.tests

    def __getitem__(self, item):
        return self.tests[item]


def isOpenFileLike(obj) -> bool:
    return (
        isinstance(obj, io.IOBase) or hasattr(obj, "read") and hasattr(obj, "readlines")
    )


class R:
    # _default = CsvReader
    def __new__(cls, path_or_buffer, *args, **kwargs):
        if cls == Reader:
            cls = cls._default
            if ext := cls._getExtension(path_or_buffer):
                if ext == "json":
                    cls = JsonReader
                elif ext == "csv":
                    cls = CsvReader
        return object.__new__(cls)


class Reader(abc.ABC, LoggerMixin):
    _supported_file_extensions = tuple()

    def __init__(self, path_or_buffer):
        try:
            ext = getFileExtension(path_or_buffer)
        except (ValueError, TypeError):
            ext = None
        self.file_ext = ext or None
        self.logger.debug(f"{self.file_ext=}")

        src = None
        if isUrl(path_or_buffer):
            data = urlopen(path_or_buffer).read().decode("utf-8")
            src = path_or_buffer
        elif isOpenFileLike(path_or_buffer):
            data = path_or_buffer.read()
            # io.StringIO has no name attribute
            src = getattr(path_or_buffer, "name", None)
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
                stacklevel=4,  # at call to AnyReader
            )

    @abc.abstractmethod
    def read(self) -> RawConfig:
        ...


class CsvReader(Reader):
    _supported_file_extensions = (".csv",)

    def __init__(self, path_or_buffer, header=1, comment="#", sep=";"):
        super().__init__(path_or_buffer)
        self.sep = sep
        self.header = header
        self.comment = comment

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
            entries.append(parts + [lineno])

        return RawConfig(entries)


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
        df["src"] = self.src

        kws = df["kwargs"].apply(
            lambda e: ", ".join([f"{k}={v}" for k, v in e.items()])
        )
        df["test"] = df["function"] + "(" + kws + ")"
        df = df[["varname", "test", "src"]]
        return RawConfig(df.itertuples(index=False), src=self.src)


if __name__ == "__main__":
    path0 = "/home/palmb/projects/saqc/ignore/ressources/config.json"
    path1 = "/home/palmb/projects/saqc/ignore/ressources/configArr.json"
    path3 = "/home/palmb/projects/saqc/ignore/ressources/config.csv"

    logging.basicConfig(level="DEBUG")
    cr = CsvReader(path3).read()
    cr[1].parse().run(SaQC())
    exit(99)
    # jr = JsonReader(path0)
    jr = JsonReader(path0, root_key="tests")
    cf = jr.read()

    print(cf.tests[1])
