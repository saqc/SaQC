#!/usr/bin/env python
from __future__ import annotations

from typing import Tuple, Type
import pandas as pd
import numpy as np


class Backtrack:
    """
    Saqc internal storage for the history of a (single) flags column.

    The backtrack (BT) stores the history of a flags column. Each time
    ``append`` is called a new column is appended to the BT. The column
    names are increasing integers starting with 0. After initialisation
    the BT is empty and has no columns at all. If an initial `UNFLAGGED`-
    column is desired, it must created manually, or passed via the ``bt``
    parameter. The same way a new BT can be created from an existing one.

    To get the worst flags (highest value) that are currently stored in
    the BT, we provide a ``max()`` method. It returns a pd.Series indicating
    the worst flag per row.

    To counteract the problem, that one may want to force a better flag
    value than the currently worst, ``append`` provide a ``force`` keyword.
    Internal we need to store the force information in an additional mask.

    For more details and a detailed discussion, why this is needed, how this
    works and possible other implementations, see #GH143 [1].
    
    [1] https://git.ufz.de/rdm-software/saqc/-/issues/143

    Parameters
    ----------
    bt : pd.Dataframe, default None
        if None a empty BT is created, otherwise the existing dataframe
        is taken as the initial backtrack.

    mask : pd.Dataframe, default None
        a mask holding the boolean force values. It must match the passed
        ``bt``. If None an matching mask is created, assuming force never
        was passed to any test.

    copy : bool, default False
        If True, the input data is copied, otherwise not.
    """

    def __init__(self, bt: pd.DataFrame = None, mask: pd.DataFrame = None, copy: bool = False):

        # this is a hidden _feature_ and not exposed by the type
        # of the bt parameter and serve as a fastpath for internal
        # fast creation of a new BT, where no checks are needed.
        if isinstance(bt, Backtrack):
            # keep this order, otherwise bt.mask
            # will refer to pd.Dataframe.mask
            mask = bt.mask
            bt = bt.bt

        elif bt is None and mask is None:
            bt = pd.DataFrame()
            mask = pd.DataFrame()

        elif bt is None and mask is not None:
            raise ValueError("Cannot take 'mask' with no 'bt'")

        elif bt is not None and mask is None:
            bt = self._validate_bt(bt)
            mask = pd.DataFrame(True, index=bt.index, columns=bt.columns)

        else:
            bt, mask = self._validate_bt_with_mask(bt, mask)

        if copy:
            bt = bt.copy()
            mask = mask.copy()

        self.bt = bt
        self.mask = mask

    @property
    def index(self) -> pd.Index:
        """
        The index of BT.

        The index is the same for all columns.

        Notes
        -----
        The index should always be equal to the flags series,
        this is BT is associated with. If this is messed up
        something went wrong in saqc internals or in a user-
        defined test.

        Returns
        -------
        index : pd.Index
        """
        return self.bt.index

    @property
    def columns(self) -> pd.Index:
        """
        Columns of the BT.

        The columns are always continuously
        increasing integers, starting from 0.

        Returns
        -------
        columns : pd.Index
        """
        return self.bt.columns

    @property
    def empty(self) -> bool:
        """
        Indicator whether Backtrack is empty.

        True if Backtrack is entirely empty (no items).

        Returns
        -------
        bool
            If Backtrack is empty, return True, if not return False.
        """
        # we take self.mask here, because it cannot have NaN's,
        # but self.bt could have -> see pd.DataFrame.empty
        return self.mask.empty

    def _insert(self, s: pd.Series, nr: int, force=False) -> Backtrack:
        """
        Insert data at an arbitrary position in the BT.

        No validation of series is done here.

        Parameters
        ----------
        s : pd.Series
            the series to insert

        nr : int
            the position to insert

        force : bool, default False
            if True the internal mask is updated accordingly

        Returns
        -------
        Backtrack
        """
        # internal detail:
        # ensure continuous increasing columns
        assert 0 <= nr <= len(self)

        # we dont care about force on first insert
        if self.empty:
            assert nr == 0

            self.mask[nr] = pd.Series(True, index=s.index, dtype=bool)
            self.bt[nr] = s
            return self

        if force:
            touched = np.isfinite(s)
            self.mask.iloc[touched, :nr] = False

        # a column is appended
        if nr == len(self):
            self.mask[nr] = True

        self.bt[nr] = s

        return self

    def append(self, value: pd.Series, force=False) -> Backtrack:
        """
        Create a new BT column and insert given pd.Series to it.

        Parameters
        ----------
        value : pd.Series
            the data to append. Must have dtype float and the index must
            match the index of the BT.

        force : bool, default False
            if True the internal mask is updated in a way that the currently
            set value (series values) will be returned if ``Backtrack.max()``
            is called. This apply for all valid values (not ``np.Nan`` and
            not ``-np.inf``).

        Raises
        ------
        ValueError: on index miss-match or wrong dtype
        TypeError: if value is not pd.Series

        Returns
        -------
        Backtrack: BT with appended series
        """
        s = self._validate_value(value)

        if s.empty:
            raise ValueError('Cannot append empty pd.Series')

        if not self.empty and not s.index.equals(self.index):
            raise ValueError("Index must be equal to BT's index")

        self._insert(value, nr=len(self), force=force)
        return self

    def squeeze(self, n: int) -> Backtrack:
        """
        Squeeze last `n` columns to a single column.

        This **not** changes the result of ``Backtrack.max()``.

        Parameters
        ----------
        n : int
            last n columns to squeeze

        Notes
        -----
        The new column number (column name) will be the lowest of
        the squeezed. This ensure that the column numbers are always
        monotonic increasing.

        Bear in mind, this works inplace, if a copy is needed, call ``copy`` before.

        Returns
        -------
        Backtrack
            squeezed backtrack
        """
        if n <= 1:
            return self

        if n > len(self):
            raise ValueError(f"'n={n}' cannot be greater than columns in the BT")

        # shortcut
        if len(self) == n:
            self.bt = pd.DataFrame()
            self.mask = pd.DataFrame()
            s = self.max()

        else:
            # calc the squeezed series.
            # we dont have to care about any forced series
            # because anytime force was given, the False's in
            # the mask were propagated back over the whole BT
            mask = self.mask.iloc[:, -n:]
            bt = self.bt.iloc[:, -n:]
            s = bt[mask].max(axis=1)

            # slice self down
            # this may leave us in an unstable state, because
            # the last column may not is entirely True, but
            # the following append, will fix this
            self.bt = self.bt.iloc[:, :-n]
            self.mask = self.mask.iloc[:, :-n]

        self.append(s)
        return self

    def max(self) -> pd.Series:
        """
        Get the maximum value per row of the BT.

        Returns
        -------
        pd.Series: maximum values
        """
        return self.bt[self.mask].max(axis=1)

    @property
    def _constructor(self) -> Type['Backtrack']:
        return Backtrack

    def copy(self, deep=True) -> Backtrack:
        """
        Make a copy of the BT.

        Parameters
        ----------
        deep : bool, default True
            - ``True``: make a deep copy
            - ``False``: make a shallow copy

        Returns
        -------
        copy : Backtrack
            the copied BT
        """
        return self._constructor(bt=self, copy=deep)

    def __len__(self) -> int:
        return len(self.bt.columns)

    def __repr__(self):

        if self.empty:
            return str(self.bt).replace('DataFrame', 'Backtrack')

        repr = self.bt.astype(str)
        m = self.mask

        repr[m] = ' ' + repr[m] + ' '
        repr[~m] = '(' + repr[~m] + ')'

        return str(repr)[1:]

    # --------------------------------------------------------------------------------
    # validation
    #

    def _validate_bt_with_mask(self, obj: pd.DataFrame, mask: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        check type, columns, index, dtype and if the mask fits the obj.
        """

        # check bt
        self._validate_bt(obj)

        # check mask
        if not isinstance(mask, pd.DataFrame):
            raise TypeError(f"'mask' must be of type pd.DataFrame, but {type(mask).__name__} was given")

        if any(mask.dtypes != bool):
            raise ValueError("dtype of all columns in 'mask' must be bool")

        if not mask.empty and not mask.iloc[:, -1].all():
            raise ValueError("the values in the last column in mask must be 'True' everywhere.")

        # check combination of bt and mask
        if not obj.columns.equals(mask.columns):
            raise ValueError("'bt' and 'mask' must have same columns")

        if not obj.index.equals(mask.index):
            raise ValueError("'bt' and 'mask' must have same index")

        return obj, mask

    def _validate_bt(self, obj: pd.DataFrame) -> pd.DataFrame:
        """
        check type, columns, dtype of obj.
        """

        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"'bt' must be of type pd.DataFrame, but {type(obj).__name__} was given")

        if any(obj.dtypes != float):
            raise ValueError('dtype of all columns in bt must be float')

        if not obj.empty and (
                not obj.columns.equals(pd.Index(range(len(obj.columns))))
                or obj.columns.dtype != int
        ):
            raise ValueError("column names must be continuous increasing int's, starting with 0.")

        return obj

    def _validate_value(self, obj: pd.Series) -> pd.Series:
        """
        index is not checked !
        """
        if not isinstance(obj, pd.Series):
            raise TypeError(f'value must be of type pd.Series, but {type(obj).__name__} was given')

        if not obj.dtype == float:
            raise ValueError('dtype must be float')

        return obj
