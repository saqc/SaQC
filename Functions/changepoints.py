"""

"""


def flagChangePoints(
    field,
    stat_func,
    thresh_func,
    window,
    min_periods,
    closed,
    reduce_window,
    reduce_func,
    flag,
):
    """
    Flag data points, where the parametrization of the process, the data is assumed to
    generate by, significantly changes.

    The change points detection is based on a sliding window search.

    Parameters
    ----------
    field : str
        The reference variable, the deviation from wich determines the flagging.

    stat_func : Callable[numpy.array, numpy.array]
         A function that assigns a value to every twin window. The backward-facing
         window content will be passed as the first array, the forward-facing window
         content as the second.

    thresh_func : Callable[numpy.array, numpy.array]
        A function that determines the value level, exceeding wich qualifies a
        timestamps func value as denoting a change-point.

    window : str, tuple of string
        Size of the rolling windows the calculation is performed in. If it is a single
        frequency offset, it applies for the backward- and the forward-facing window.

        If two offsets (as a tuple) is passed the first defines the size of the
        backward facing window, the second the size of the forward facing window.

    min_periods : int or tuple of int
        Minimum number of observations in a window required to perform the changepoint
        test. If it is a tuple of two int, the first refer to the backward-,
        the second to the forward-facing window.

    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.

    reduce_window : {None, str}, default None
        The sliding window search method is not an exact CP search method and usually there wont be
        detected a single changepoint, but a "region" of change around a changepoint.
        If `reduce_window` is given, for every window of size `reduce_window`, there
        will be selected the value with index `reduce_func(x, y)` and the others will be dropped.
        If `reduce_window` is None, the reduction window size equals the
        twin window size, the changepoints have been detected with.

    reduce_func : Callable[[numpy.ndarray, numpy.ndarray], int], default lambda x, y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for every
        reduction window. Second input parameter holds the result from the thresh_func evaluation.
        The default reduction function just selects the value that maximizes the stat_func.

    flag : float, default BAD
        flag to set.

    Returns
    -------
    """
    pass


def assignChangePointCluster(
    field,
    stat_func,
    thresh_func,
    window,
    min_periods,
    closed,
    reduce_window,
    reduce_func,
    set_flags,
    model_by_resids,
    assign_cluster,
    flag,
):
    """
    Assigns label to the data, aiming to reflect continous regimes of the processes the data is assumed to be
    generated by.
    The regime change points detection is based on a sliding window search.

    Note, that the cluster labels will be stored to the `field` field of the input data, so that the data that is
    clustered gets overridden.

    Parameters
    ----------
    field : str
        The reference variable, the deviation from wich determines the flagging.

    stat_func : Callable[[numpy.array, numpy.array], float]
        A function that assigns a value to every twin window. Left window content will be passed to first variable,
        right window content will be passed to the second.

    thresh_func : Callable[numpy.array, numpy.array], float]
        A function that determines the value level, exceeding wich qualifies a timestamps func func value as denoting a
        changepoint.

    window : str, tuple of string
        Size of the rolling windows the calculation is performed in. If it is a single
        frequency offset, it applies for the backward- and the forward-facing window.

        If two offsets (as a tuple) is passed the first defines the size of the
        backward facing window, the second the size of the forward facing window.

    min_periods : int or tuple of int
        Minimum number of observations in a window required to perform the changepoint
        test. If it is a tuple of two int, the first refer to the backward-,
        the second to the forward-facing window.

    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.

    reduce_window : {None, str}, default None
        The sliding window search method is not an exact CP search method and usually there wont be
        detected a single changepoint, but a "region" of change around a changepoint.
        If `reduce_window` is given, for every window of size `reduce_window`, there
        will be selected the value with index `reduce_func(x, y)` and the others will be dropped.
        If `reduce_window` is None, the reduction window size equals the
        twin window size, the changepoints have been detected with.

    reduce_func : Callable[[numpy.array, numpy.array], numpy.array], default lambda x, y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for every
        reduction window. Second input parameter holds the result from the thresh_func evaluation.
        The default reduction function just selects the value that maximizes the stat_func.

    set_flags : bool, default False
        If true, the points, where there is a change in data modelling regime detected gets flagged.

    model_by_resids : bool, default False
        If True, the data is replaced by the stat_funcs results instead of regime labels.

    assign_cluster : bool, default True
        Is set to False, if called by function that oly wants to calculate flags.

    flag : float, default BAD
        flag to set.

    Returns
    -------
    """
    pass
