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
import pandas as pd


class Ml:
    @doc(saqc.funcs.ml.trainModel.__doc__)
    def trainModel(
        self,
        field: str,
        target: str,
        window: Union[str, int],
        target_idx: Union[int, list[int], Literal["center", "forward"]],
        mode: Union[Literal["regressor", "classifier", "flagger"], str],
        path: str,
        test_split: Optional[Union[float, str]] = None,
        feature_mask: Optional[Union[str, np.array, pd.DataFrame, dict]] = None,
        dropna: bool = True,
        ini_kwargs: Optional[dict] = None,
        multi_target_model: Optional[Literal["chain", "multi"]] = "chain",
        base_estimater: Optional[callable] = None,
        errors: Literal['coerce', 'raise'] = 'raise',
        dfilter: float = BAD,
        override: bool = False,
        sfilter: Optional[Callable] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("trainModel", locals())

    @doc(saqc.funcs.ml.modelPredict.__doc__)
    def modelPredict(
        self,
        field: str,
        path: str,
        agg_func: callable = np.nanmean,
        dropna: Optional[bool] = None,
        assign_features: Optional[dict] = None,
        dfilter: float = FILTER_NONE,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("modelPredict", locals())

    @doc(saqc.funcs.ml.modelFlag.__doc__)
    def modelFlag(
        self,
        field: str,
        path: str,
        agg_func: callable = np.nanmean,
        dropna: Optional[bool] = None,
        assign_features: Optional[dict] = None,
        dfilter: float = FILTER_NONE,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("modelFlag", locals())

    @doc(saqc.funcs.ml.modelImpute.__doc__)
    def modelImpute(
        self,
        field: str,
        path: str,
        agg_func: callable = np.nanmean,
        dropna: Optional[bool] = None,
        assign_features: Optional[dict] = None,
        dfilter: float = FILTER_NONE,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("modelImpute", locals())
