"""

"""
def flagChangePoints(field, stat_func, thresh_func, bwd_window, min_periods_bwd, fwd_window, min_periods_fwd, closed, reduce_window, reduce_func):
    """
    Flag datapoints, where the parametrization of the process, the data is assumed to generate by, significantly
    changes.
    
    The change points detection is based on a sliding window search.
    
    Parameters
    ----------
    field : str
        The reference variable, the deviation from wich determines the flagging.
    stat_func : Callable[numpy.array, numpy.array]
         A function that assigns a value to every twin window. Left window content will be passed to first variable,
        right window content will be passed to the second.
    thresh_func : Callable[numpy.array, numpy.array]
        A function that determines the value level, exceeding wich qualifies a timestamps stat func value as denoting a
        changepoint.
    bwd_window : str
        The left (backwards facing) windows temporal extension (freq-string).
    min_periods_bwd : {str, int}
        Minimum number of periods that have to be present in a backwards facing window, for a changepoint test to be
        performed.
    fwd_window : {None, str}, default None
        The right (forward facing) windows temporal extension (freq-string).
    min_periods_fwd : {None, str, int}, default None
        Minimum number of periods that have to be present in a forward facing window, for a changepoint test to be
        performed.
    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.
    reduce_window : {None, str}, default None
        The sliding window search method is not an exact CP search method and usually there wont be
        detected a single changepoint, but a "region" of change around a changepoint.
        If `reduce_window` is given, for every window of size `reduce_window`, there
        will be selected the value with index `reduce_func(x, y)` and the others will be dropped.
        If `reduce_window` is None, the reduction window size equals the
        twin window size, the changepoints have been detected with.
    reduce_func : Callable[[numpy.array, numpy.array], np.array], default lambda x, y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for every
        reduction window. Second input parameter holds the result from the thresh_func evaluation.
        The default reduction function just selects the value that maximizes the stat_func.
    
    
    Returns
    -------
    """
    pass


def assignChangePointCluster(field, stat_func, thresh_func, bwd_window, min_periods_bwd, fwd_window, min_periods_fwd, closed, reduce_window, reduce_func, flag_changepoints, model_by_resids, assign_cluster):
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
        A function that determines the value level, exceeding wich qualifies a timestamps stat func value as denoting a
        changepoint.
    bwd_window : str
        The left (backwards facing) windows temporal extension (freq-string).
    min_periods_bwd : {str, int}
        Minimum number of periods that have to be present in a backwards facing window, for a changepoint test to be
        performed.
    fwd_window : {None, str}, default None
        The right (forward facing) windows temporal extension (freq-string).
    min_periods_fwd : {None, str, int}, default None
        Minimum number of periods that have to be present in a forward facing window, for a changepoint test to be
        performed.
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
    flag_changepoints : bool, default False
        If true, the points, where there is a change in data modelling regime detected get flagged bad.
    model_by_resids : bool, default False
        If True, the data is replaced by the stat_funcs results instead of regime labels.
    assign_cluster : bool, default True
        Is set to False, if called by function that oly wants to calculate flags.
    
    Returns
    -------
    """
    pass
