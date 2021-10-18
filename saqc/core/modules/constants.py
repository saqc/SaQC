#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
import saqc
from saqc.lib.types import FreqString


class Constants(ModuleBase):
    def flagByVariance(
        self,
        field: str,
        window: FreqString = "12h",
        thresh: float = 0.0005,
        maxna: int = None,
        maxna_group: int = None,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagByVariance", locals())

    def flagConstants(
        self, field: str, thresh: float, window: FreqString, flag: float = BAD, **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagConstants", locals())
