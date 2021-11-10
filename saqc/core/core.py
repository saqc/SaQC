#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import warnings
from typing import (
    Any,
    Callable,
    List,
    Sequence,
    Tuple,
    Union,
    Optional,
    Mapping,
    Hashable,
)
from copy import deepcopy, copy as shallowcopy

import pandas as pd
import numpy as np

from dios import DictOfSeries, to_dios

from saqc.core.flags import initFlagsLike, Flags
from saqc.core.register import FUNC_MAP
from saqc.core.modules import FunctionsMixin
from saqc.core.translator.basetranslator import Translator, FloatTranslator
from saqc.lib.tools import toSequence
from saqc.lib.types import (
    ExternalFlag,
    PandasLike,
)


# TODO: shouldn't the code/function go to SaQC.__init__ ?
def _prepInput(
    data: PandasLike,
    flags: Optional[Union[DictOfSeries, pd.DataFrame, Flags]],
    copy: bool,
) -> Tuple[DictOfSeries, Optional[Flags]]:
    dios_like = (DictOfSeries, pd.DataFrame)

    if copy:
        data = deepcopy(data)

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
            flags = Flags(flags, copy=copy)

    return data, flags


def _setup():
    # NOTE:
    # the import is needed to trigger the registration
    # of the built-in (test-)functions
    import saqc.funcs  # noqa

    # warnings
    pd.set_option("mode.chained_assignment", "warn")
    np.seterr(invalid="ignore")


_setup()


class Accessor:
    def __init__(self, obj: Union[DictOfSeries, pd.DataFrame, Flags]):
        self._obj = obj

    def __getitem__(self, key):
        return self._obj[key]

    @property
    def columns(self):
        return self._obj.columns

    def __len__(self):
        return len(self.columns)

    def __repr__(self):
        return self._obj.__repr__()


class SaQC(FunctionsMixin):
    _attributes = {
        "_data",
        "_flags",
        "_translator",
        "_attrs",
        "called",
    }

    def __init__(self, data, flags=None, scheme: Translator = None, copy: bool = True):
        data, flags = _prepInput(data, flags, copy)
        self._data = data
        self._flags = self._initFlags(data, flags)
        self._translator = scheme or FloatTranslator()
        self.called = []
        self._attrs = {}

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

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """
        Dictionary of global attributes of this dataset.
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = dict(value)

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
        out = SaQC(data=DictOfSeries(), flags=Flags(), scheme=self._translator)
        out.attrs = self._attrs
        for k, v in injectables.items():
            if k not in self._attributes:
                raise AttributeError(f"SaQC has no attribute {repr(k)}")
            setattr(out, k, v)
        return out

    @property
    def data(self) -> Accessor:
        return Accessor(self._data)

    @property
    def flags(self) -> Accessor:
        return Accessor(self._translator.backward(self._flags, attrs=self.attrs))

    def getResult(
        self, raw=False
    ) -> Union[Tuple[DictOfSeries, Flags], Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Realize the registered calculations and return the results

        Returns
        -------
        data, flags: (DictOfSeries, DictOfSeries)
        """

        data, flags = self._data, self._flags

        if raw:
            return data, flags

        return data.to_df(), self._translator.backward(flags, attrs=self.attrs)

    def _wrap(self, func: Callable):
        """Enrich a function by special saqc-functionality.

        For each saqc function this realize
            - regex's in field,
            - use default of translator for ``to_mask`` if not specified by user,
            - translation of ``flag`` and
            - working inplace.
        Therefore it adds the following keywords to each saqc function:
        ``regex`` and ``inplace``.

        The returned function returns a Saqc object.
        """

        def inner(
            field: str | Sequence[str],
            *args,
            regex: bool = False,
            flag: ExternalFlag = None,
            **kwargs,
        ) -> SaQC:

            kwargs.setdefault("to_mask", self._translator.TO_MASK)

            # translation
            if flag is not None:
                kwargs["flag"] = self._translator(flag)

            # expand regular expressions
            if regex:
                fmask = self._data.columns.str.match(field)
                fields = self._data.columns[fmask].tolist()
            else:
                fields = toSequence(field)

            if func.__multivariate__:
                # we wrap field again to generalize the down stream loop work as expected
                fields = [fields]

            out = self

            for f in fields:
                out = out._callFunction(
                    func,
                    data=out._data,
                    flags=out._flags,
                    field=f,
                    *args,
                    **kwargs,
                )
            return out

        return inner

    def _callFunction(
        self,
        function: Callable,
        data: DictOfSeries,
        flags: Flags,
        field: str | Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> SaQC:

        data, flags = function(data=data, flags=flags, field=field, *args, **kwargs)

        planned = self.called + [(field, (function, args, kwargs))]

        # keep consistence: if we modify data and flags inplace in a function,
        # but data is the original and flags is a copy (as currently implemented),
        # data and flags of the original saqc obj may change inconsistently.
        self._data = data
        self._flags = flags

        return self._construct(_data=data, _flags=flags, called=planned)

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
        copyfunc = deepcopy if deep else shallowcopy
        new = self._construct()
        for attr in self._attributes:
            setattr(new, attr, copyfunc(getattr(self, attr)))
        return new

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memodict=None):
        return self.copy(deep=True)
