#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-

from typing import Mapping

import numpy as np

from saqc.constants import BAD, GOOD, UNFLAGGED
from saqc.core.translation import MappingScheme
from saqc.lib.types import ExternalFlag


class SimpleScheme(MappingScheme):

    """
    Acts as the default Translator, provides a changeable subset of the
    internal float flags
    """

    _FORWARD: dict[ExternalFlag, float] = {
        "UNFLAGGED": UNFLAGGED,
        "BAD": BAD,
        "OK": GOOD,
    }

    _BACKWARD: dict[float, ExternalFlag] = {
        UNFLAGGED: "UNFLAGGED",
        np.nan: "UNFLAGGED",
        BAD: "BAD",
        GOOD: "OK",
    }

    def __init__(self):
        super().__init__(forward=self._FORWARD, backward=self._BACKWARD)
