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
from matplotlib.widgets import RectangleSelector, SpanSelector, MultiCursor
from matplotlib.dates import date2num
import matplotlib as mpl

ASSIGN_SHORTCUT = "enter"
LEFT_MOUSE_BUTTON = 1
RIGHT_MOUSE_BUTTON = 3
SELECTION_MARKER_DEFAULT = {"zorder": 10, "c": "red", "s": 50, "marker": "x"}
# if scrollable GUI: determines number of figures per x-size of the screen
FIGS_PER_SCREEN = 2
# or hight in inches (if given overrides number of figs per screen):
FIG_HIGHT_INCH = None

BLIT_MARKERS = False


class BlitManager:
    def __init__(self, canvas):
        self.canvas = canvas
        self._artists = len(self.canvas.figure.axes)*[None]
        self._bg = None
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        print('on_draw triggered')
        print(event)
        if event is not None:
            if event.canvas != self.canvas:
                raise RuntimeError
        self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_animated()

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            if a is not None:
                fig.draw_artist(a)

    def add_artist(self, art, loc):
        """Add a new Artist object to the Blit Manager"""
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        if self._artists[loc] is not None:
            self._artists[loc].remove()
        self._artists[loc] = art

    def update(self):
        """Update the screen with animated artists."""

        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            self.canvas.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            self.canvas.blit(self.canvas.figure.bbox)
        # let the GUI event loop process anything it has to do
        self.canvas.flush_events()

class MplScroller(tk.Frame):
    def __init__(self, parent, fig):
        tk.Frame.__init__(self, parent)
        # frame - canvas - window combo that enables scrolling:
        self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")
        self.canvas.bind_all("<Button-4>", lambda x: self.mouseWheeler(-1))
        self.canvas.bind_all("<Button-5>", lambda x: self.mouseWheeler(1))

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

        # background
        self._bg = None

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
        self.marker_handles = self.N*[None]
        for k in range(self.N):
            self.ax[k].set_xlim(auto=True)

        self.canvas = self.ax[0].figure.canvas
        self.blit_manager = BlitManager(self.canvas)
        self.selection_marker_kwargs={**SELECTION_MARKER_DEFAULT, **selection_marker_kwargs}
        self.rc_rect = None
        self.lc_rect = None
        self.spawn_selector(type='rect')
        self.current_slc='rect'
        self.marked = [np.zeros(data[k].shape[0]).astype(bool) for k in range(self.N)]
        self.confirmed = False
        self.index = [data[k].index for k in range(self.N)]
        self.data = [data[k].values for k in range(self.N)]
        self.numidx = [date2num(self.index[k]) for k in range(self.N)]
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
        self.canvas.mpl_connect("button_press_event", self.buttonPressEvent)

        self.canvas.draw_idle()

    def onLeftSelectFunc(self, ax_num):
        return lambda x, y, z=ax_num: self.onLeftSelect(x, y, z)

    def onRightSelectFunc(self, ax_num):
        return lambda x, y, z=ax_num: self.onRightSelect(x, y, z)

    def onLeftSelect(self, eclick, erelease, ax_num=0, _select_to=True):
        if not isinstance(eclick,np.float64):
            upper_left = (
                min(eclick.xdata, erelease.xdata),
                max(eclick.ydata, erelease.ydata),
            )

            bottom_right = (
                max(eclick.xdata, erelease.xdata),
                min(eclick.ydata, erelease.ydata),
            )
            x_cut = (self.numidx[ax_num] > upper_left[0]) & (
                self.numidx[ax_num] < bottom_right[0]
            )
            y_cut = (self.data[ax_num] > bottom_right[1]) & (
                self.data[ax_num] < upper_left[1]
            )
            self.marked[ax_num][x_cut & y_cut] = _select_to
        else:
            x_cut = (self.numidx[ax_num] > eclick) & (
                self.numidx[ax_num] < erelease
            )
            self.marked[ax_num][x_cut] = _select_to


        xl = self.ax[ax_num].get_xlim()
        marker_artist = self.ax[ax_num].scatter(self.index[ax_num][self.marked[ax_num]], self.data[ax_num][self.marked[ax_num]], **self.selection_marker_kwargs)
        if BLIT_MARKERS:
            self.blit_manager.add_artist(marker_artist, ax_num)
            self.blit_manager.update()
        else:
            if self.marker_handles[ax_num] is not None:
                self.marker_handles[ax_num].remove()
            self.marker_handles[ax_num] = marker_artist
            self.canvas.draw_idle()

        self.ax[ax_num].set_xlim(xl)


    def onRightSelect(self, eclick, erelease, ax_num=0):
        self.onLeftSelect(eclick, erelease, ax_num=ax_num, _select_to=False)

    def disconnect(self):
        for k in range(self.N):
            self.lc_rect[k].disconnect_events()
            self.rc_rect[k].disconnect_events()

    def spawn_selector(self, type='rect'):
        if self.rc_rect:
            for k in range(self.N):
                self.rc_rect[k].disconnect_events()
                self.lc_rect[k].disconnect_events()
        if type=='rect':
            self.lc_rect = [
                RectangleSelector(
                    self.ax[k],
                    self.onLeftSelectFunc(k),
                    button=[1],
                    use_data_coordinates=True,
                    useblit=True,
                )
                for k in range(self.N)
            ]
            self.rc_rect = [
                RectangleSelector(
                    self.ax[k],
                    self.onRightSelectFunc(k),
                    button=[3],
                    use_data_coordinates=True,
                    useblit=True,
                )
                for k in range(self.N)
            ]
        elif type=='span':

            self.lc_rect = [
                SpanSelector(
                    self.ax[k],
                    self.onLeftSelectFunc(k),
                    'horizontal',
                    button=[1],
                    useblit=True
                )
                for k in range(self.N)
            ]
            self.rc_rect = [
                SpanSelector(
                    self.ax[k],
                    self.onRightSelectFunc(k),
                    'horizontal',
                    button=[3],
                    useblit=True,
                )
                for k in range(self.N)
            ]


    def assignAndCloseCB(self, val=None):
        self.confirmed = True
        plt.close(self.ax[0].figure)

    def keyPressEvents(self, event):
        if event.key == ASSIGN_SHORTCUT:
            if self.parent is None:
                self.assignAndCloseCB()
            else:
                self.parent.quitFunc(self)
        elif event.key == 'shift':
            if self.current_slc=='rect':
                self.spawn_selector('span')
                self.current_slc='span'
            elif self.current_slc=='span':
                self.spawn_selector('rect')
                self.current_slc='rect'
        print(event.key)

    def buttonPressEvent(self, event):
        print(event)

