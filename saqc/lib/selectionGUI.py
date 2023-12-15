#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backend_tools import ToolBase
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

ASSIGN_SHORTCUT = "enter"
LEFT_MOUSE_BUTTON = 1
RIGHT_MOUSE_BUTTON = 3
SELECTION_MARKER_DEFAULT = {"zorder": 10, "c": "red", "s": 50, "marker": "x"}

class MplScroller(tk.Frame):
    def __init__(self, parent, fig):

        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")
        self.frame = tk.Frame(self.canvas, background="#ffffff")
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)


        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw",
                                  tags="self.frame")

        self.frame.bind("<Configure>", self.onFrameConfigure)
        self.fig = fig
        self.populate()

    def populate(self):
        canvas = FigureCanvasTkAgg(self.fig, master=self.frame)  # A tk.DrawingArea.
        toolbar = NavigationToolbar2Tk(canvas, self.canvas)
        toolbar.update()
        canvas.get_tk_widget().pack()#side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

    def onFrameConfigure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

class AssignFlagsTool(ToolBase):
    default_keymap = "enter"  # keyboard shortcut
    description = "Assign flags to selection."

    def __init__(self, *args, callback, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback

    def trigger(self, *args, **kwargs):
        self.callback()


class SelectionOverlay:
    def __init__(
        self,
        ax,
        data,
        selection_marker_kwargs=SELECTION_MARKER_DEFAULT,
    ):
        self.N = len(data)
        self.ax = ax
        self.collection = [
            self.ax[k].scatter(
                data[k].index,
                data[k].values,
                **{**SELECTION_MARKER_DEFAULT, **selection_marker_kwargs}
            )
            for k in range(self.N)
        ]

        self.xys = [self.collection[k].get_offsets() for k in range(self.N)]
        self.fc = [
            np.tile(self.collection[k].get_facecolors(), (len(self.xys[k]), 1))
            for k in range(self.N)
        ]

        for k in range(self.N):
            self.ax[k].set_xlim(auto=True)
            self.fc[k][:, -1] = 0
            self.collection[k].set_facecolors(self.fc[k])

        self.canvas = self.ax[0].figure.canvas

        self.lc_rect = [
            RectangleSelector(
                ax[k],
                self.onLeftSelectFunc(k),
                button=[1],
                use_data_coordinates=True,
                useblit=True,
            )
            for k in range(self.N)
        ]
        self.rc_rect = [
            RectangleSelector(
                ax[k],
                self.onRightSelectFunc(k),
                button=[3],
                use_data_coordinates=True,
                useblit=True,
            )
            for k in range(self.N)
        ]
        self.marked = [np.zeros(data[k].shape[0]).astype(bool) for k in range(self.N)]
        self.confirmed = False
        self.index = [data[k].index for k in range(self.N)]

        # add assignment button to the toolbar
        #self.canvas.manager.toolmanager.add_tool(
        #    "Assign Flags", AssignFlagsTool, callback=self.assignAndCloseCB
        #)

        #self.canvas.manager.toolbar.add_tool("Assign Flags", "Flags")
        #self.canvas.manager.toolmanager.remove_tool("help")

        self.canvas.draw_idle()

    def onLeftSelectFunc(self, ax_num):
        return lambda x, y, z=ax_num: self.onLeftSelect(x, y, z)

    def onRightSelectFunc(self, ax_num):
        return lambda x, y, z=ax_num: self.onRightSelect(x, y, z)

    def onLeftSelect(self, eclick, erelease, ax_num=0, _select_to=True):
        upper_left = (
            min(eclick.xdata, erelease.xdata),
            max(eclick.ydata, erelease.ydata),
        )

        bottom_right = (
            max(eclick.xdata, erelease.xdata),
            min(eclick.ydata, erelease.ydata),
        )
        x_cut = (self.xys[ax_num][:, 0] > upper_left[0]) & (
            self.xys[ax_num][:, 0] < bottom_right[0]
        )
        y_cut = (self.xys[ax_num][:, 1] > bottom_right[1]) & (
            self.xys[ax_num][:, 1] < upper_left[1]
        )
        # self.marked[:] = False
        self.marked[ax_num][x_cut & y_cut] = _select_to

        self.fc[ax_num][:, -1] = 0
        self.fc[ax_num][self.marked[ax_num], -1] = 1
        self.collection[ax_num].set_facecolors(self.fc[ax_num])
        self.canvas.draw_idle()

    def onRightSelect(self, eclick, erelease, ax_num=0):
        self.onLeftSelect(eclick, erelease, ax_num=ax_num, _select_to=False)

    def disconnect(self):
        for k in range(self.N):
            self.lc_rect[k].disconnect_events()
            self.rc_rect[k].disconnect_events()

    def assignAndCloseCB(self):  # , vals=None):
        self.confirmed = True
        plt.close(self.ax[0].figure)
