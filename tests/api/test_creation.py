#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pandas as pd
import numpy as np
import dios


def test_init():
    from saqc import SaQC, Flags

    arr = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
        ]
    )
    data = pd.DataFrame(arr, columns=list("abc"))
    qc = SaQC(data)

    assert isinstance(qc, SaQC)
    assert isinstance(qc._flags, Flags)
    assert isinstance(qc._data, dios.DictOfSeries)