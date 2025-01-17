#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from saqc.constants import BAD, DOUBTFUL, GOOD, UNFLAGGED
from saqc.core import Flags, History
from saqc.core.frame import DictOfSeries
from saqc.core.translation.basescheme import BackwardMap, ForwardMap, MappingScheme


class PositionalScheme(MappingScheme):
    """
    Implements the translation from and to the flagging scheme implemented by CHS
    """

    DFILTER_DEFAULT = DOUBTFUL + 1

    _FORWARD: ForwardMap = {
        -6: UNFLAGGED,
        -5: UNFLAGGED,
        -2: UNFLAGGED,
        -1: UNFLAGGED,
        0: UNFLAGGED,
        1: DOUBTFUL,
        2: BAD,
    }
    _BACKWARD: BackwardMap = {
        np.nan: 0,
        UNFLAGGED: 0,
        GOOD: 0,
        DOUBTFUL: 1,
        BAD: 2,
    }

    def __init__(self):
        super().__init__(forward=self._FORWARD, backward=self._BACKWARD)

    def toInternal(self, flags: pd.DataFrame) -> Flags:
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
                field_history.append(s.replace(UNFLAGGED, np.nan))
            data[str(field)] = field_history
        return Flags(data)

    def toExternal(self, flags: Flags, **kwargs) -> DictOfSeries:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        Returns
        -------
        DictOfSeries
        """
        out = DictOfSeries()
        for field in flags.columns:
            thist = flags.history[field].hist.replace(self._BACKWARD).astype(float)
            # concatenate the single flag values
            ncols = len(thist.columns)
            init = 9 * 10**ncols
            bases = 10 ** np.arange(ncols - 1, -1, -1)

            tflags = init + (thist * bases).sum(axis=1)
            out[field] = tflags.fillna(-9999).astype(int)

        return out
