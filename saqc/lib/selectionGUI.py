#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_tools import ToolBase
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector

ASSIGN_SHORTCUT = "enter"
LEFT_MOUSE_BUTTON = 1
RIGHT_MOUSE_BUTTON = 3
SELECTION_MARKER_DEFAULT = {"zorder": 10, "c": "red", "s": 50, "marker": "x"}
# if scrollable GUI: determines number of figures per x-size of the screen
FIGS_PER_SCREEN = 2
# or hight in inches (if given overrides number of figs per screen):
FIG_HIGHT_INCH = None


class MplScroller(tk.Frame):
    def __init__(self, parent, fig):
        tk.Frame.__init__(self, parent)
        # frame - canvas - window combo that enables scrolling:
        self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")
        # binding linux-known mousewheel shortcuts for mousewheel scrolling:
        self.canvas.bind_all("<Button-4>", lambda x: self.mouseWheeler(-1))
        self.canvas.bind_all("<Button-5>", lambda x: self.mouseWheeler(1))
        # windows-known mousewheel shortcuts for mousewheel scrolling:
        # ....

        self.frame = tk.Frame(self.canvas, background="#ffffff")
        self.vert_scrollbar = tk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.vert_scrollbar.set)

        self.vert_scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window(
            (4, 4), window=self.frame, anchor="nw", tags="self.frame"
        )

        self.frame.bind("<Configure>", self.scrollAreaCallBack)

        # keeping references
        self.parent = parent
        self.fig = fig
        tk.Button(self.canvas, text="Discard (and quit)", command=self.quitFunc).pack()
        # adjusting content to the scrollable view
        self.figureSizer()
        self.figureShifter()
        self.scrollContentGenerator()

    def mouseWheeler(self, direction):
        self.canvas.yview_scroll(direction, "units")

    def assignationGenerator(self, selector):
        tk.Button(
            self.canvas,
            text="Assign Flags",
            command=lambda s=selector: self.quitFunc(s),
        ).pack()

    def quitFunc(self, selector=None):
        if selector:
            selector.confirmed = True
        plt.close(self.fig)
        self.quit()

    def scrollContentGenerator(self):
        canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        toolbar = NavigationToolbar2Tk(canvas, self.canvas)
        toolbar.update()
        canvas.get_tk_widget().pack()
        canvas.draw()

    def scrollAreaCallBack(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def figureSizer(self):
        window = plt.get_current_fig_manager().window
        f_size = list(window.wm_maxsize())
        px = 1 / plt.rcParams["figure.dpi"]
        f_size = [ws * px for ws in f_size]
        if not FIG_HIGHT_INCH:
            f_size[1] = f_size[1] * len(self.fig.axes) * FIGS_PER_SCREEN**-1
        else:
            f_size[1] = FIG_HIGHT_INCH * len(self.fig.axes)
        self.fig.set_size_inches(f_size[0], f_size[1])

    def figureShifter(self):
        window = plt.get_current_fig_manager().window
        screen_hight = window.wm_maxsize()[1]
        fig_hight = self.fig.get_size_inches()
        ratio = fig_hight[1] / screen_hight
        to_shift = ratio
        for k in range(len(self.fig.axes)):
            print(self.fig.axes[k].get_position().bounds[1])
            b = self.fig.axes[k].get_position().bounds
            self.fig.axes[k].set_position((b[0], b[1] + to_shift, b[2], b[3]))


class AssignFlagsTool(ToolBase):
    description = "Assign flags to selection."

    def __init__(self, *args, callback, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback

    def trigger(self, *args, **kwargs):
        self.callback()


class SelectionOverlay:
    def __init__(
        self, ax, data, selection_marker_kwargs=SELECTION_MARKER_DEFAULT, parent=None
    ):
        self.parent = parent
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

        if not parent:
            # add assignment button to the toolbar
            self.canvas.manager.toolmanager.add_tool(
                "Assign Flags", AssignFlagsTool, callback=self.assignAndCloseCB
            )
            self.canvas.manager.toolbar.add_tool("Assign Flags", "Flags")
            self.canvas.manager.toolmanager.remove_tool("help")
        else:
            parent.assignationGenerator(self)
            self.canvas.mpl_connect("key_press_event", self.keyPressEvents)

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

    def assignAndCloseCB(self, val=None):
        self.confirmed = True
        plt.close(self.ax[0].figure)

    def keyPressEvents(self, event):
        if event.key == ASSIGN_SHORTCUT:
            if self.parent is None:
                self.assignAndCloseCB()
            else:
                self.parent.quitFunc(self)
