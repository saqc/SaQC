from __future__ import annotations

import operator
import warnings
from typing import TYPE_CHECKING, Any, Callable, Sequence, Union

import numpy as np
import pandas as pd

import saqc

OPERATOR_RELATION = {"&": "__le__", "|": "__ge__", "+": "__le__", "-":"__le__", "*": "__le__", "/": "__le__", "**": "__le__"}
OPERATOR_FUNCS = {'+': '__add__', '-': '__sub__',"*": "__mul__", "/": "__truediv__", "**": "__pow__"}

if TYPE_CHECKING:
    from saqc import SaQC


class BoolOpsMixin:
    def _booloperate(self, other, operator, s_rep = None, o_rep=None, o_kwargs=None):

        if not isinstance(o_rep,list):
            o_rep = [o_rep]*len(other.columns)
        if not isinstance(o_kwargs, list):
            o_kwargs = [o_kwargs] * len(other.columns)
        o_rep = dict(zip(other.columns, o_rep))
        o_kwargs = dict(zip(other.columns, o_kwargs))
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
                        "func": o_rep[of] or of,
                        "args": (),
                        "kwargs": o_kwargs[of] or {"dfilter": -np.inf, "field": of},
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

    def _arithmeticOperate(self, other, operator):
        o_rep = None
        if isinstance(other, np.ndarray):
            other = pd.DataFrame(other)
        if pd.api.types.is_scalar(other):
            o_rep = str(other)
            o_kwargs = {'value': o_rep, 'type':str(type(other)), 'squeezed':True}
            other = saqc.SaQC([pd.Series(other, index=self.data[f].index, name=f) for f in self.data.columns],
                              scheme=self.scheme)
        elif isinstance(other, Union[pd.DataFrame, pd.Series]):
            other = other.to_frame() if isinstance(other, pd.Series) else other
            o_rep = other.columns.to_list()
            o_kwargs = [{'value': f'column: {k}', 'type': str(other[k].dtype), 'squeezed': True} for k in other.columns]
            other = saqc.SaQC(other, scheme=self.scheme)

        else:
            other = other.__invert__()
        out = self._booloperate(other, operator, o_rep=o_rep, o_kwargs=o_kwargs)
        for f, of in zip(self.columns, other.columns):
            out._data[f] = getattr(out.data[f],OPERATOR_FUNCS[operator])(other.data[of].values)

        return out

    def __and__(self, other, self_assign=False):
        return self.__invert__(s_rep=self_assign)._booloperate(other.__invert__(), "&")

    def __or__(self, other, self_assign=False):
        return self.__invert__(s_rep=self_assign)._booloperate(other.__invert__(), "|")

    def __invert__(self, s_rep=False):
        out = self.copy(deep=True)
        for f in self.columns:
            flags = out._history[f].squeeze()
            _meta = out._history[f].meta
            if len(_meta) > 0:
                ops = [f for f in _meta[-1]['kwargs'] if f.startswith('operand')]
                if len(ops) > 0:
                    unsqueezed = [o for o in ops if not _meta[-1]['kwargs'][o].get('squeezed', False)]
                    if s_rep:
                        func_entry = f"|{f}|"
                        kwargs_entry = {'operand_0':{'squeezed':True, 'type': 'field', 'value': f'{f}_data'}}
                    else:
                        func_entry = f"{_meta[-1]['func']}"
                        kwargs_entry = _meta[-1]['kwargs']
                        if len(unsqueezed) > 0:
                            func_entry = f"|{func_entry}|"
                            for o in unsqueezed:
                                kwargs_entry[o].update({'squeezed':True})
                else:
                    func_entry = f"|{f}|"
                    kwargs_entry = {'operand_0':{'squeezed':True}}

                meta = {"func": func_entry, "args": (), "kwargs": kwargs_entry}
                history = saqc.core.History(flags.index).append(flags, meta=meta)
                out._history[f] = history
        return out

    def __ilshift__(self, other):
        return self.__lshift__(other)

    def __lshift__(self, other, d_vals='left'):
        self=self.copy(deep=True)
        for f, of in zip(self.columns, other.columns):
            if len(other._history[of].meta) > 0:
                for k in range(len(other._history[of].meta)):
                    self._flags.history[f].append(other._history[of].hist.iloc[:,k], other._flags.history[of].meta[k])
            else:
                self._flags.history[f].append(other._flags[of], {'func':'append', 'args':(), 'kwargs':{'field':of}})
        if d_vals=='right':
            self._data = other._data
        return self


    def __iand__(self, other):
        new_cols = self.__and__(other, self_assign=True)
        for f in self.columns:
            self._history[f].append(new_cols._flags[f], meta=new_cols._history[f].meta[-1])

        return self

    def __ior__(self, other):
        new_cols = self.__or__(other, self_assign=True)
        for f in self.columns:
            self._history[f].append(new_cols._flags[f], meta=new_cols._history[f].meta[-1])

        return self

    def __add__(self, other, self_assign=False):
        return self.__invert__(s_rep=self_assign)._arithmeticOperate(other, '+')

    def __radd__(self, other):
        return self.__add__(other, d_vals='right')

    def __iadd__(self, other):
        return self.__lshift__(self.__add__(other, self_assign=True), d_vals='right')

    def __sub__(self, other, self_assign=False):
        return self.__invert__(s_rep=self_assign)._arithmeticOperate(other, '-')

    def __rsub__(self, other):
        return self.__sub__(other)

    def __isub__(self, other):
        return self.__lshift__(self.__sub__(other, self_assign=True), d_vals='right')

    def __truediv__(self, other, self_assign=False):
        return self.__invert__(s_rep=self_assign)._arithmeticOperate(other, '/')

    def __rtruediv__(self, other):
        return self.__pow__(-1).__mul__(other)

    def __itruediv__(self, other):
        return self.__lshift__(self.__truediv__(other, self_assign=True), d_vals='right')
    def __mul__(self, other, self_assign=False):
        return self.__invert__(s_rep=self_assign)._arithmeticOperate(other, '*')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__lshift__(self.__mul__(other, self_assign=True), d_vals='right')

    def __pow__(self, other, self_assign=False):
        return self.__invert__(s_rep=self_assign)._arithmeticOperate(other, '**')

    def __rpow__(self, other):
        raise NotImplementedError('not implemented')

    def __ipow__(self, other):
        return self.__lshift__(self.__pow__(other, self_assign=True), d_vals='right')

    def __xor__(self, other):
        return self.__lshift__(other, d_vals='right')

    def __ixor__(self, other):
        return self.__xor__(other)
