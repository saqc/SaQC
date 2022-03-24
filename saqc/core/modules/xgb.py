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
import numpy as np

class Xgb:
    @doc(saqc.funcs.xgb.trainXGB.__doc__)
    def trainXGB(
        self,
        field: str,
        target: str,
        window: Union[str, int],
        target_i: Union[int, list[int], Literal["center", "forward"]],
        predict: Union[Literal["flag", "value"], str],
        model_dir: str,
        id: Optional[str] = None,
        mask_target: Optional[bool] = None,
        filter_predictors: Optional[bool] = None,
        training_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("trainXGB", locals())

    @doc(saqc.funcs.xgb.predictXGB.__doc__)
    def predictXGB(
        self,
        field: str,
        model_dir: str,
        pred_agg: callable = np.nanmean,
        id: Optional[str] = None,
        model_var: Optional[str] = None,
        filter_predictors: Optional[bool] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("predictXGB", locals())
