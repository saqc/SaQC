#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Union
from typing_extensions import Literal

from saqc.constants import BAD, FILTER_ALL, FILTER_NONE
import saqc
from saqc.lib.docurator import doc
import saqc.funcs
import numpy as np


class Xgb:
    @doc(saqc.funcs.xgb.trainModel.__doc__)
    def trainModel(
        self,
        field: str,
        target: str,
        window: Union[str, int],
        target_i: Union[int, list[int], Literal["center", "forward"]],
        mode: Union[Literal["Regressor", "Classifier", "Flagger"], str],
        results_path: str,
        model_folder: Optional[str] = None,
        tt_split: Optional[Union[float, str]] = None,
        mask_target: Optional[bool] = None,
        filter_predictors: Optional[bool] = None,
        training_kwargs: Optional[dict] = None,
        base_estimater: Optional[callable] = None,
        dfilter: float = BAD,
        override: bool = False,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("trainModel", locals())

    @doc(saqc.funcs.xgb.modelPredict.__doc__)
    def modelPredict(
        self,
        field: str,
        results_path: str,
        pred_agg: callable = np.nanmean,
        model_folder: Optional[str] = None,
        filter_predictors: Optional[bool] = None,
        flag: float = BAD,
        dfilter: float = FILTER_NONE,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("modelPredict", locals())

    @doc(saqc.funcs.xgb.modelFlag.__doc__)
    def modelFlag(
        self,
        field: str,
        results_path: str,
        pred_agg: callable = np.nanmean,
        model_folder: Optional[str] = None,
        filter_predictors: Optional[bool] = None,
        dfilter: float = FILTER_NONE,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("modelFlag", locals())

    @doc(saqc.funcs.xgb.modelImpute.__doc__)
    def modelImpute(
        self,
        field: str,
        results_path: str,
        pred_agg: callable = np.nanmean,
        model_folder: Optional[str] = None,
        filter_predictors: Optional[bool] = None,
        dfilter: float = FILTER_NONE,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("modelImpute", locals())
