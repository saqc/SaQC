#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional

import numpy as np
import pandas as pd

from lib.types import PandasLike, ArrayLike, T


class Flags(pd.CategoricalDtype):
    def __init__(self, flags):
        assert len(flags) > 2
        super().__init__(flags, ordered=True)

    def unflagged(self):
        return self[0]

    def min(self):
        return self[1]

    def max(self):
        return self[-1]

    def __getitem__(self, idx):
        return self.categories[idx]


class BaseFlagger:
    def __init__(self, flags):
        self.flags = Flags(flags)

    def setFlag(self,
                flags: PandasLike,
                flag: Optional[T] = None,
                **kwargs: Any) -> np.ndarray:
        """
        NOTE:
        this method should really return a numpy array, as
        pandas data structures tend to behave unpredictively
        in assignments, especially if a multi column index is used
        """
        if flag is None:
            flag = self.flags.max()
        else:
            self._check_flag(flag)
        flags = flags.copy()
        flags[flags < flag] = flag
        return flags.values

    def initFlags(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy().astype(self.flags)
        out.loc[:] = self.flags[0]
        return out

    def isFlagged(self, flags: ArrayLike, flag: T = None) -> ArrayLike:
        if flag is None:
            return pd.notnull(flags) & (flags > self.flags[0])
        else:
            self._check_flag(flag)
            return flags >= flag

    def getFlags(self, flags: ArrayLike):
        return flags

    def _check_flag(self, flag):
        if flag not in self.flags:
            raise ValueError(f"Invalid flag '{flag}'. "
                             f"Possible choices are {list(self.flags.categories)[1:]}")

    def nextTest(self):
        pass
