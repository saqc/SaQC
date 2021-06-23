#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import inspect
import copy as stdcopy
from typing import Tuple, Union, Optional
from typing_extensions import Literal

import pandas as pd
import numpy as np

from dios import DictOfSeries, to_dios

from saqc.core.flags import initFlagsLike, Flags
from saqc.core.lib import APIController, ColumnSelector
from saqc.core.register import FUNC_MAP, SaQCFunction
from saqc.core.modules import FuncModules
from saqc.core.translator.basetranslator import Translator, FloatTranslator
from saqc.constants import BAD
from saqc.lib.tools import toSequence
from saqc.lib.types import (
    ExternalFlag,
    CallGraph,
    MaterializedGraph,
    PandasLike,
)


logger = logging.getLogger("SaQC")


def _handleErrors(
    exc: Exception,
    field: str,
    control: APIController,
    func: SaQCFunction,
    policy: Literal["ignore", "warn", "raise"],
):
    message = "\n".join(
        [
            f"Exception:\n{type(exc).__name__}: {exc}",
            f"field: {field}",
            f"{func.errorMessage()}",
            f"{control.errorMessage()}",
        ]
    )

    if policy == "ignore":
        logger.debug(message)
    elif policy == "warn":
        logger.warning(message)
    else:
        logger.error(message)
        raise exc


# TODO: shouldt the code/function go to Saqc.__init__ ?
def _prepInput(
    data: PandasLike, flags: Optional[Union[DictOfSeries, pd.DataFrame, Flags]]
) -> Tuple[DictOfSeries, Optional[Flags]]:
    dios_like = (DictOfSeries, pd.DataFrame)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    if not isinstance(data, dios_like):
        raise TypeError(
            "'data' must be of type pd.Series, pd.DataFrame or dios.DictOfSeries"
        )

    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex) or isinstance(
            data.columns, pd.MultiIndex
        ):
            raise TypeError("'data' should not use MultiIndex")
        data = to_dios(data)

    if not hasattr(data.columns, "str"):
        raise TypeError("expected dataframe columns of type string")

    if flags is not None:

        if isinstance(flags, pd.DataFrame):
            if isinstance(flags.index, pd.MultiIndex) or isinstance(
                flags.columns, pd.MultiIndex
            ):
                raise TypeError("'flags' should not use MultiIndex")

        if isinstance(flags, (DictOfSeries, pd.DataFrame, Flags)):
            # NOTE: only test common columns, data as well as flags could
            # have more columns than the respective other.
            cols = flags.columns.intersection(data.columns)
            for c in cols:
                if not flags[c].index.equals(data[c].index):
                    raise ValueError(
                        f"the index of 'flags' and 'data' missmatch in column {c}"
                    )

        # this also ensures float dtype
        if not isinstance(flags, Flags):
            flags = Flags(flags, copy=True)

    return data, flags


def _setup():
    # NOTE:
    # the import is needed to trigger the registration
    # of the built-in (test-)functions
    import saqc.funcs

    # warnings
    pd.set_option("mode.chained_assignment", "warn")
    np.seterr(invalid="ignore")


_setup()


class SaQC(FuncModules):
    def __init__(
        self,
        data,
        flags=None,
        scheme: Translator = None,
        nodata=np.nan,
        error_policy="raise",
        lazy=False,
    ):
        super().__init__(self)
        data, flags = _prepInput(data, flags)
        self._data = data
        self._nodata = nodata
        self._flags = self._initFlags(data, flags)
        self._error_policy = error_policy
        self._lazy = lazy
        self._translator = scheme or FloatTranslator()

        # NOTE:
        # We need two lists to represent the future and the past computations
        # on a `SaQC`-Object. Due to the dynamic nature of field expansion
        # with regular expressions, we can't just reuse the original execution
        # plan to infer all translation related information.
        self._planned: CallGraph = []  # will be filled by calls to `_wrap`
        self._computed: MaterializedGraph = self._translator.buildGraph(
            self._flags
        )  # will be filled in `evaluate`

    @staticmethod
    def _initFlags(data: DictOfSeries, flags: Optional[Flags]) -> Flags:
        """
        Init the internal Flags-object.

        Ensures that all data columns are present and user passed
        flags from a frame or an already initialised Flags-object
        are used.
        """
        if flags is None:
            return initFlagsLike(data)

        # add columns that are present in data but not in flags
        for c in data.columns.difference(flags.columns):
            flags[c] = initFlagsLike(data[c])

        return flags

    def _construct(self, **injectables) -> SaQC:
        """
        Construct a new `SaQC`-Object from `self` and optionally inject
        attributes with any chechking and overhead.

        Parameters
        ----------
        **injectables: any of the `SaQC` data attributes with name and value

        Note
        ----
        For internal usage only! Setting values through `injectables` has
        the potential to mess up certain invariants of the constructed object.
        """
        out = SaQC(
            data=DictOfSeries(),
            flags=Flags(),
            nodata=self._nodata,
            error_policy=self._error_policy,
            scheme=self._translator,
        )
        for k, v in injectables.items():
            if not hasattr(out, k):
                raise AttributeError(f"failed to set unknown attribute: {k}")
            setattr(out, k, v)
        return out

    def readConfig(self, fname):
        from saqc.core.reader import readConfig

        out = stdcopy.deepcopy(self)
        out._planned.extend(readConfig(fname, self._data, self._nodata))
        if self._lazy:
            return out
        return out.evaluate()

    def evaluate(self):
        """
        Realize all the registered calculations and return a updated SaQC Object

        Parameters
        ----------

        Returns
        -------
        An updated SaQC Object incorporating the requested computations
        """

        # NOTE: It would be nicer to separate the plotting into an own
        #       method instead of intermingling it with the computation
        data, flags = self._data, self._flags
        computed: MaterializedGraph = []
        for selector, control, function in self._planned:
            logger.debug(
                f"processing: {selector.field}, {function.name}, {function.keywords}"
            )
            assert data.columns.difference(flags.columns).empty

            try:
                data_result, flags_result = function(data, selector.field, flags)
                # we check the passed function-kwargs after the actual call,
                # because now "hard" errors would already have been raised
                # (eg. `TypeError: got multiple values for argument 'data'`,
                # when the user pass data=...)
                _warnForUnusedKwargs(function, self._translator)
                computed.append((selector, function))
            except Exception as e:
                _handleErrors(e, selector.field, control, function, self._error_policy)
                continue

            data = data_result
            flags = flags_result

        return self._construct(
            _flags=flags, _data=data, _computed=self._computed + computed
        )

    def getResult(
        self, raw=False
    ) -> Union[Tuple[DictOfSeries, Flags], Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Realize the registered calculations and return the results

        Returns
        -------
        data, flags: (DictOfSeries, DictOfSeries)
        """

        realization = self.evaluate()
        data, flags = realization._data, realization._flags

        if raw:
            return data, flags

        return data.to_df(), self._translator.backward(flags, realization._computed)

    def _wrap(self, func: SaQCFunction):
        def inner(
            field: str,
            *fargs,
            target: str = None,
            regex: bool = False,
            flag: ExternalFlag = BAD,
            inplace: bool = False,
            **fkwargs,
        ) -> SaQC:

            if regex and target is not None:
                raise ValueError("explicit `target` not supported with `regex=True`")

            fkwargs.setdefault("to_mask", self._translator.TO_MASK)

            out = self if inplace else self.copy(deep=True)

            control = APIController()
            partial = func.bind(
                *fargs,
                **{"nodata": self._nodata, "flag": self._translator(flag), **fkwargs},
            )

            # expand regular expressions
            if regex:
                fields = self._data.columns.str.match(field)
                fields = self._data.columns[fields]
            else:
                fields = toSequence(field)

            for field in fields:
                target = target if target is not None else field
                if field != target:
                    copy_func = FUNC_MAP["tools.copy"]
                    out._planned.append(
                        (
                            ColumnSelector(field, target),
                            control,
                            copy_func.bind(new_field=target),
                        )
                    )
                    field = target

                out._planned.append((ColumnSelector(field, target), control, partial))

            if self._lazy:
                return out

            return out.evaluate()

        return inner

    def __getattr__(self, key):
        """
        All failing attribute accesses are redirected to
        __getattr__. We use this mechanism to make the
        registered functions as `SaQC`-methods without
        actually implementing them.
        """
        if key not in FUNC_MAP:
            raise AttributeError(f"no such attribute: '{key}'")
        return self._wrap(FUNC_MAP[key])

    def copy(self, deep=True):
        if deep:
            return stdcopy.deepcopy(self)
        return stdcopy.copy(self)


def _warnForUnusedKwargs(func, translator: Translator):
    """Warn for unused kwargs, passed to a SaQC.function.

    Parameters
    ----------
    func: SaqcFunction
        Saqc internal data structure that hold all function info.

    Returns
    -------
    None

    Notes
    -----
    A single warning via the logging module is thrown, if any number of
    missing kws are detected, naming each missing kw.
    """
    sig_kws = inspect.signature(func.func).parameters

    # we need to ignore kws that are injected or by default hidden in ``**kwargs``
    ignore = ("nodata", "to_mask")

    missing = []
    for kw in func.keywords:
        # there is no need to check for
        # `kw in [KEYWORD_ONLY, VAR_KEYWORD or POSITIONAL_OR_KEYWORD]`
        # because this would have raised an error beforehand.
        if kw not in sig_kws and kw not in ignore and kw not in translator.ARGUMENTS:
            missing.append(kw)

    if missing:
        missing = ", ".join(missing)
        logging.warning(f"Unused argument(s): {missing}")
