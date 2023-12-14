#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_tools import ToolBase
from matplotlib.widgets import Button, RectangleSelector, TextBox

ASSIGN_SHORTCUT = "enter"
LEFT_MOUSE_BUTTON = 1
RIGHT_MOUSE_BUTTON = 3
SELECTION_MARKER_DEFAULT = {"zorder": 10, "c": "red", "s": 50, "marker": "x"}


class AssignFlagsTool(ToolBase):
    default_keymap = "enter"  # keyboard shortcut
    description = "Assign Flags to selection"

    def __init__(self, *args, cb, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb = cb

    def trigger(self, *args, **kwargs):
        self.cb()


class SelectionOverlay:
    def __init__(
        self,
        ax,
        data,
        selection_marker_kwargs=SELECTION_MARKER_DEFAULT,
    ):
        self.ax = ax
        self.collection = self.ax.scatter(
            data.index,
            data.values,
            **{**SELECTION_MARKER_DEFAULT, **selection_marker_kwargs}
        )
        self.ax.set_xlim(auto=True)
        self.xys = self.collection.get_offsets()

        self.canvas = self.ax.figure.canvas
        # self.canvas.mpl_connect("key_press_event", self.keyPressEvents)

        self.fc = np.tile(self.collection.get_facecolors(), (len(self.xys), 1))
        self.fc[:, -1] = 0
        self.collection.set_facecolors(self.fc)

        self.lc_rect = RectangleSelector(
            ax, self.onLeftSelect, button=[1], use_data_coordinates=True, useblit=True
        )
        self.rc_rect = RectangleSelector(
            ax, self.onRightSelect, button=[3], use_data_coordinates=True, useblit=True
        )
        self.marked = np.zeros(data.shape[0]).astype(bool)
        self.confirmed = False
        self.index = data.index
        # Buttons and Text Boxes:

        self.canvas.manager.toolmanager.add_tool(
            "Assign Flags", AssignFlagsTool, cb=self.assignAndCloseCB
        )

        self.canvas.manager.toolbar.add_tool("Assign Flags", "Flags")
        self.canvas.manager.toolmanager.remove_tool("help")

        self.canvas.draw_idle()

    def onLeftSelect(self, eclick, erelease, _select_to=True):
        upper_left = (
            min(eclick.xdata, erelease.xdata),
            max(eclick.ydata, erelease.ydata),
        )

        bottom_right = (
            max(eclick.xdata, erelease.xdata),
            min(eclick.ydata, erelease.ydata),
        )
        x_cut = (self.xys[:, 0] > upper_left[0]) & (self.xys[:, 0] < bottom_right[0])
        y_cut = (self.xys[:, 1] > bottom_right[1]) & (self.xys[:, 1] < upper_left[1])
        # self.marked[:] = False
        self.marked[x_cut & y_cut] = _select_to

        self.fc[:, -1] = 0
        self.fc[self.marked, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def onRightSelect(self, eclick, erelease):
        self.onLeftSelect(eclick, erelease, _select_to=False)

    def disconnect(self):
        self.lc_rect.disconnect_events()
        self.rc_rect.disconnect_events()

    def assignAndCloseCB(self):  # , vals=None):
        self.confirmed = True
        plt.close(self.ax.figure)

    def keyPressEvents(self, event):
        if event.key == ASSIGN_SHORTCUT:
            self.assignAndCloseCB(event.key)
