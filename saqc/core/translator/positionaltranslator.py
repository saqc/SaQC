#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from saqc.core.flags import (
    Flags,
    History,
    UNTOUCHED,
    UNFLAGGED,
    GOOD,
    DOUBTFUL,
    BAD,
)
from saqc.lib.types import MaterializedGraph
from saqc.core.translator.basetranslator import Translator, ForwardMap, BackwardMap


class PositionalTranslator(Translator):

    """
    Implements the translation from and to the flagging scheme implemented by CHS
    """

    _FORWARD: ForwardMap = {0: UNFLAGGED, 1: DOUBTFUL, 2: BAD}
    _BACKWARD: BackwardMap = {
        UNTOUCHED: 0,
        UNFLAGGED: 0,
        GOOD: 0,
        DOUBTFUL: 1,
        BAD: 2,
    }

    def __init__(self):
        super().__init__(forward=self._FORWARD, backward=self._BACKWARD)

    def forward(self, flags: pd.DataFrame) -> Flags:
        """
        Translate from 'external flags' to 'internal flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        Returns
        -------
        Flags object
        """

        data = {}
        for field, field_flags in flags.items():

            # explode the flags into sperate columns and drop the leading `9`
            df = pd.DataFrame(
                field_flags.astype(str).str.slice(start=1).apply(tuple).tolist(),
                index=field_flags.index,
            ).astype(int)

            # the exploded values form the History of `field`
            fflags = super()._translate(df, self._FORWARD)
            field_history = History(field_flags.index)
            for _, s in fflags.items():
                field_history.append(s, force=True)
            data[str(field)] = field_history

        return Flags(data)

    def backward(self, flags: Flags) -> pd.DataFrame:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        Returns
        -------
        pd.DataFrame
        """
        out = {}
        for field in flags.columns:
            thist = flags.history[field].hist.replace(self._BACKWARD)
            # Concatenate the single flag values. There are faster and more
            # complicated approaches (see former `PositionalFlagger`), but
            # this method shouldn't be called that often
            ncols = thist.shape[-1]
            init = 9 * 10 ** ncols
            bases = 10 ** np.arange(ncols - 1, -1, -1)

            tflags = init + (thist * bases).sum(axis=1)
            out[field] = tflags

        return pd.DataFrame(out).fillna(-9999).astype(int)
