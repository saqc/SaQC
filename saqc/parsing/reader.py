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
from typing import Collection, Iterable, Iterator, List, Literal
from urllib.request import urlopen

import pandas as pd

from saqc import SaQC
from saqc.exceptions import ParsingError, _SpecialKeyError
from saqc.lib.checking import isOpenFileLike, isUrl, isQuoted
from saqc.lib.tools import LoggerMixin, fileExists, getFileExtension
from saqc.parsing.visitor import ConfigFunctionParser


def fromConfig(fname, *args, **func_kwargs):
    return CsvReader(fname).read().parse().run(SaQC(*args, **func_kwargs))


class _ConfigReader:
    def __init__(self, *args, **kwargs):
        self.qc = SaQC(*args, **kwargs)

    def readJson(self, c):
        self.config: Config = JsonReader(c).read()
        self.reader: Config = self.config.parse()
        return self

    def readString(self, c):
        self.config: Config = CsvReader(c).read()
        self.reader: Config = self.config.parse()
        return self

    readCsv = readString

    def run(self):
        return self.reader.run(self.qc)


class ConfigEntry:
    def __init__(self, var: str, func_text: str, src: str, lineno: int | None = None):
        self.var = var
        self.functxt = func_text
        self.src = src
        self.lineno = lineno

        # filled by parse
        self.func_name = None
        self.kws = None

    def __repr__(self):
        lno = ""
        if self.lineno is not None:
            lno = f"line {self.lineno}: "
        if self.parsed:
            first = self.func_name
            second = self.kws
        else:
            first = self.var
            second = self.functxt

        return f"{lno}{{{first}, {second}}}"

    def __str__(self):
        return (
            f"{self.__class__.__qualname__}\n"
            f"  source:       {self.src}\n"
            f"  lineno:       {self.lineno}\n"
            f"  varname:      {self.var}\n"
            f"  functext:     {self.functxt}\n"
            f"  parsed:       {self.parsed}\n"
            f"  parsed-func:  {self.func_name}\n"
            f"  parsed-kws:   {self.kws}\n"
        )

    @property
    def parsed(self):
        return self.func_name is not None

    def parse(self):
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

        self.func_name = func
        self.kws = kws
        return self

    def run(self, qc):
        if not self.parsed:
            raise RuntimeError(f"{self.__class__.__qualname__} must be parsed first")
        return getattr(qc, self.func_name)(**self.kws)


class Config(LoggerMixin, Collection[ConfigEntry]):
    def __contains__(self, item: object) -> bool:
        return item in self.tests

    def __len__(self):
        return len(self.tests)

    def __iter__(self) -> Iterator[ConfigEntry]:
        yield from self.tests

    def __getitem__(self, item) -> ConfigEntry:
        return self.tests[item]

    def __init__(self, obj: Iterable, src: str | None = None):
        self.src = src
        self.tests: List[ConfigEntry] = []
        self.is_parsed = False
        for args in obj:
            self.tests.append(ConfigEntry(*args))

    def __repr__(self):
        cname = self.__class__.__qualname__
        src = f"({self.src!r})" if self.src else ""
        if not self.tests:
            return f"Empty {cname}{src}"
        tests = "\n".join(["[", *[indent(repr(t), " ") for t in self.tests], "]"])
        return f"{cname}{src}\n{tests}\n"

    @staticmethod
    def _formatErrMsg(e: Exception, test: ConfigEntry, msg: str = "") -> str:
        return "\n".join([str(e), msg, str(test)])

    def parse(self) -> Config[ConfigEntry]:
        if self.is_parsed:
            raise RuntimeError("config is already parsed")
        parsed: List[ConfigEntry] = []
        msg = "Parsing config failed"
        for i, test in enumerate(self.tests):
            try:
                parsed.append(test.parse())
            except NameError as e:
                raise NameError(self._formatErrMsg(e, test, msg)) from None
            except Exception as e:
                raise ParsingError(self._formatErrMsg(e, test, msg)) from None
        self.tests = parsed
        self.is_parsed = True
        return self

    def run(self, qc: SaQC) -> SaQC:
        if not self.is_parsed:
            raise RuntimeError("config must be parsed first")
        msg = f"Executing config failed"
        for i, test in enumerate(self.tests):
            try:
                qc = test.run(qc)
            except KeyError as e:
                # We need to handle KeyError differently, because
                # it uses `repr` instead of `str` and would mess
                # up our message with extra text and newlines.
                e = _SpecialKeyError(self._formatErrMsg(e, test, msg))
                raise e.with_traceback(e.__traceback__) from None
            except Exception as e:
                e = type(e)(self._formatErrMsg(e, test, msg))
                raise e.with_traceback(e.__traceback__) from None
        return qc


class Reader(abc.ABC, LoggerMixin):
    _supported_file_extensions = frozenset()

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
            src = "String-config"
            self.file_ext = None
        if not isinstance(data, str):
            raise TypeError(f"unsupported type {type(path_or_buffer)}")

        self.data = data
        self.src = src
        self._maybeWarnExtension()

    def _maybeWarnExtension(self):
        if (
            self.file_ext is not None
            and self.file_ext not in self._supported_file_extensions
        ):
            warnings.warn(
                f"File extension is {self.file_ext!r} but the reader "
                f"expects one of {set(self._supported_file_extensions)}",
                category=RuntimeWarning,
                stacklevel=4,  # at call to SomeReader
            )

    @abc.abstractmethod
    def read(self) -> Config[ConfigEntry]:
        ...


class CsvReader(Reader):
    _supported_file_extensions = frozenset({".csv"})

    def __init__(
        self,
        path_or_buffer,
        header: int | Literal["infer"] = "infer",
        comment: str = "#",
        sep: str = ";",
    ):
        """

        Parameters
        ----------
        path_or_buffer :
        header :
            If 'infer' the reader ignores a header if one is present.
            If header is an integer, this many lines are ignored.
            Commented lines or empty lines are not counted.
        comment :
        sep :
        """
        super().__init__(path_or_buffer)
        self.sep = sep
        self.header = header
        self.comment = comment
        if self.src is None:
            self.src = ""
            pass

    def read(self):
        entries = []
        infer = self.header == "infer"
        skip = 0 if infer else (self.header + 1)
        comment = self.comment or None
        for i, line in enumerate(self.data.splitlines()):
            lineno = i + 1
            line: str = line.strip()
            if not line or comment is not None and line.startswith(comment):
                continue
            if infer and line.replace(" ", "") == f"varname{self.sep}test":
                infer = False
                continue
            infer = False
            if skip := max(skip - 1, 0):
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
    _supported_file_extensions = frozenset({".json"})

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
    path3 = spath

    logging.basicConfig(level="DEBUG")
    config = JsonReader(path0, root_key="tests").read()
    print(config)
    print(config.parse())
    config = CsvReader(path3).read()
    qc = SaQC(dict(SM4=pd.Series([1, 2, 3]), SM2=pd.Series([2, 3, 4, 5])))
    config.parse().run(qc)
    # config[1].parse().run(SaQC())
    exit(99)
