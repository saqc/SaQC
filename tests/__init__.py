#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import saqc
from packaging import version
if version.parse(saqc.__version__) >= version.parse("3.0.0"):
    saqc.options["legacy_call_style"] = False
else:
    saqc.options["legacy_call_style"] = True
    saqc.options["legacy_call_style_warning"] = False
