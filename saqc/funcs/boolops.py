from __future__ import annotations

import operator
import warnings
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import pandas as pd

import saqc

OPERATOR_RELATION = {"&": "__le__", "|": "__ge__", "+": "__le__"}

if TYPE_CHECKING:
    from saqc import SaQC


class BoolOpsMixin:
    def _booloperate(self, other, operator, s_rep = None, o_rep=None):
        if len(self.columns) != len(other.columns):
            raise ValueError(
                "boolean saqc object combination only supported for saqc objects of the same number of variables."
            )
        if type(self.scheme) != type(other.scheme):
            raise ValueError(
                "Cant combine differently schemed saqc objects in boolean operation."
            )
        for f, of in zip(self.columns, other.columns):
            if len(self._flags.history[f].meta) == 0:
                new_meta = {
                        "func": s_rep or f,
                        "args": (),
                        "kwargs": {"dfilter": -np.inf, "field": f},
                    }

                self_flags = self._flags[f]
            else:
                new_meta = self._flags.history[f].meta[-1]
                self_flags = (
                    self._flags.history[f].hist.iloc[:, -1].replace(np.nan, -np.inf)
                )
            if len(other._flags.history[of].meta) == 0:
                other_meta =  {
                        "func": o_rep or of,
                        "args": (),
                        "kwargs": {"dfilter": -np.inf, "field": of},
                    }

                other_flags = other._flags[of]
            else:
                other_meta = other._flags.history[of].meta[-1]
                other_flags = (
                    other._flags.history[of].hist.iloc[:, -1].replace(np.nan, -np.inf)
                )
            flagscol = self_flags.where(
                getattr(self_flags, OPERATOR_RELATION[operator])(other_flags),
                other_flags,
            )
            self_func = new_meta['kwargs'].get('label', new_meta['func'])
            other_func = other_meta['kwargs'].get('label', other_meta['func'])
            self_operands = [
                int(v.split("_")[-1])
                for v in new_meta["kwargs"].keys()
                if v.startswith("operand_")
            ]
            other_operands = [
                int(v.split("_")[-1])
                for v in other_meta["kwargs"].keys()
                if v.startswith("operand_")
            ]
            if len(self_operands) == 0:
                new_meta["kwargs"] = {"operand_0": new_meta["kwargs"]}
                self_operands = [0]
            if len(other_operands) == 0:
                other_meta["kwargs"] = {
                    f"operand_{max(self_operands) + 1}": other_meta["kwargs"]
                }
                other_operands = [max(self_operands) + 1]


            new_meta["func"] = f'({self_func} {operator} {other_func})'
            new_meta["kwargs"] = {
                f"operand_{k}": new_meta["kwargs"][f"operand_{s}"]
                for k, s in enumerate(self_operands)
            }
            new_meta["kwargs"].update(
                {
                    f"operand_{k + len(self_operands)}": other_meta["kwargs"][
                        f"operand_{s}"
                    ]
                    for k, s in enumerate(other_operands)
                }
            )

            self._flags.history[f] = saqc.core.history.History(flagscol.index).append(flagscol, new_meta)
        return self

    def __and__(self, other):
        return self.copy(deep=True)._booloperate(other, "&")

    def __or__(self, other):
        return self.copy(deep=True)._booloperate(other, "|")

    def __invert__(self):
        out = saqc.SaQC(self.data, self.flags, scheme=self.scheme)
        for f in self.columns:
            out._flags.history[f].append(self._flags[f], {"func": "squeezed", "args": (), "kwargs": {"field": f}})
        return out

    def __lt__(self, other):
        out = self.copy(deep=True)
        for f, of in zip(out.columns,other.columns):
            out._flags.history[f].append(other._flags.history[of].copy(deep=True))
        return out

    def __gt__(self, other):
        return other.__lt__(self)

    def __le__(self, other):
        out = self.copy(deep=True)
        out._data = other.data
        return out
    def __ge__(self, other):
        return other.__le__(self)


    def __iand__(self, other):
        return self.__invert__().__and__(other.__invert__())

    def __ior__(self, other):
        return self.__invert__().__or__(other.__invert__())

    def __add__(self, other):
        o_rep=None
        if pd.api.types.is_scalar(other):
            o_rep=str(other)
            other = saqc.SaQC([pd.Series(other, index=self.data[f].index, name=f) for f in self.data.columns],
                              scheme=self.scheme)

        out = self._booloperate(other, '+', o_rep=o_rep)
        for f, of in zip(self.columns, other.columns):
            out._data[f] = out.data[f] + other.data[of]

        return out

