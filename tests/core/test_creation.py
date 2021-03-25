#!/usr/bin/env python

import pandas as pd
import numpy as np
import dios


def test_init():
    from saqc import SaQC, Flags as Flagger

    arr = np.array([
        [0, 1, 2],
        [0, 1, 3],
    ])
    data = pd.DataFrame(arr, columns=list('abc'))
    qc = SaQC(data)

    assert isinstance(qc, SaQC)
    assert isinstance(qc._flagger, Flagger)
    assert isinstance(qc._data, dios.DictOfSeries)
