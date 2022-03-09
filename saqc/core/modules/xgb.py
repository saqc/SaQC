#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Union
from typing_extensions import Literal

from saqc.constants import BAD, FILTER_ALL
import saqc
from saqc.lib.docurator import doc
import saqc.funcs


class Xgb:
    @doc(saqc.funcs.xgb.trainXGB.__doc__)
    def trainXGB(
        self,
        field: str,
        target: str,
        window: Union[str, int],
        target_i: Union[int, list[int], Literal["center", "forward"]],
        predict: Union[Literal["flag", "value"], str],
        mask_target: bool = True,
        training_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("trainXGB", locals())
