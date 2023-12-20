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
from matplotlib.dates import date2num
from matplotlib.widgets import MultiCursor, RectangleSelector, SpanSelector

ASSIGN_SHORTCUT = "enter"
DISCARD_SHORTCUT = "escape"
LEFT_MOUSE_BUTTON = 1
RIGHT_MOUSE_BUTTON = 3
SELECTION_MARKER_DEFAULT = {"zorder": 10, "c": "red", "s": 50, "marker": "x"}
# if scrollable GUI: determines number of figures per x-size of the screen
FIGS_PER_SCREEN = 2
# or hight in inches (if given overrides number of figs per screen):
FIG_HIGHT_INCH = None
BFONT = ("Times", "16")
VARFONT = ("Times", "12")
CP_WIDTH = 15
SELECTOR_DICT = {"rect": "Rectangular", "span": "Span"}


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

        self.control_panel = tk.Frame(self.canvas, bg='DarkGray')#, borderwidth=0)
        self.control_panel.pack(side=tk.LEFT, anchor="n")
        # adding buttons
        self.quit_button = tk.Button(
            self.control_panel,
            text="Discard and Quit.",
            command=self.assignAndQuitFunc,
            #relief=tk.RAISED,
            bg="red",
            width=CP_WIDTH,
            relief="flat",
            overrelief="groove",
            #borderwidth=1
            #font=BFONT,
        )
        self.quit_button.grid(column=3, row=0, pady=5.5, padx=2.25)#'4.4p')
        # selector info display
        self.current_slc_entry = tk.StringVar(self.control_panel)
        # tk.Label(self.canvas, textvariable=self.current_slc_entry, width=20).pack(
        #    anchor="w", side=tk.BOTTOM
        # )

        tk.Label(
            self.control_panel,
            textvariable=self.current_slc_entry,
            width=CP_WIDTH,
            #font=BFONT,
        ).grid(column=1, row=0, pady=5.5, padx=2.25)

        # binding overview

        self.binding_status = [
            tk.IntVar(self.control_panel) for k in range(len(self.fig.axes))
        ]
        if len(self.binding_status) > 100:
            for v in enumerate(self.binding_status):
                tk.Checkbutton(
                    self.control_panel,
                    text=self.fig.axes[v[0]].get_title(),
                    variable=v[1],
                    width=CP_WIDTH,
                    anchor="w",
                    font=VARFONT,
                ).grid(
                    column=0, row=2 + v[0]
                )  # .pack(anchor="w", side=tk.BOTTOM)

        # adjusting content to the scrollable view
        self.figureSizer()
        self.figureShifter()
        self.scrollContentGenerator()

    def mouseWheeler(self, direction):
        self.canvas.yview_scroll(direction, "units")

    def assignationGenerator(self, selector):
        tk.Button(
            self.control_panel,
            text="Assign Flags",
            command=lambda s=selector: self.assignAndQuitFunc(s),
            bg="green",
            width=CP_WIDTH,
            #font=BFONT,
        ).grid(
            column=0, row=0, pady=5.5, padx=2.25
        )

    def assignAndQuitFunc(self, selector=None):
        if selector:
            selector.confirmed = True
        plt.close(self.fig)
        self.quit()


    def scrollContentGenerator(self):
        canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        toolbar = NavigationToolbar2Tk(canvas, self.canvas)
        toolbar.update()
        toolbar.pack(side=tk.TOP)
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
        self.marker_handles = self.N * [None]
        for k in range(self.N):
            self.ax[k].set_xlim(auto=True)

        self.canvas = self.ax[0].figure.canvas
        self.selection_marker_kwargs = {
            **SELECTION_MARKER_DEFAULT,
            **selection_marker_kwargs,
        }
        self.rc_rect = None
        self.lc_rect = None
        self.current_slc = "rect"
        self.spawn_selector(type=self.current_slc)

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
        self.canvas.mpl_connect("button_press_event", self.buttonPressEvents)
        self.canvas.draw_idle()

    def onLeftSelectFunc(self, ax_num):
        return lambda x, y, z=ax_num: self.onLeftSelect(x, y, z)

    def onRightSelectFunc(self, ax_num):
        return lambda x, y, z=ax_num: self.onRightSelect(x, y, z)

    def onLeftSelect(self, eclick, erelease, ax_num=0, _select_to=True):
        if self.current_slc == "rect":
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
            s_mask = x_cut & y_cut

        elif self.current_slc == "span":
            x_cut = (self.numidx[ax_num] > eclick) & (self.numidx[ax_num] < erelease)
            s_mask = x_cut

        ax_num = np.array([ax_num])
        if (self.current_slc == "span") and (self.parent is not None):
            stati = np.array([s.get() for s in self.parent.binding_status]).astype(bool)
            print(f"stati={stati}")
            if stati.any():
                stati_w = np.where(stati)[0]
                if ax_num[0] in stati_w:
                    ax_num = stati_w

        for num in ax_num:
            self.marked[num][s_mask] = _select_to
            xl = self.ax[num].get_xlim()
            yl = self.ax[num].get_ylim()
            marker_artist = self.ax[num].scatter(
                self.index[num][self.marked[num]],
                self.data[num][self.marked[num]],
                **self.selection_marker_kwargs,
            )

            if self.marker_handles[num] is not None:
                self.marker_handles[num].remove()
            self.marker_handles[num] = marker_artist
            self.ax[num].set_xlim(xl)
            self.ax[num].set_ylim(yl)

        self.canvas.draw_idle()

    def onRightSelect(self, eclick, erelease, ax_num=0):
        self.onLeftSelect(eclick, erelease, ax_num=ax_num, _select_to=False)

    def disconnect(self):
        for k in range(self.N):
            self.lc_rect[k].disconnect_events()
            self.rc_rect[k].disconnect_events()

    def spawn_selector(self, type="rect"):
        if self.rc_rect:
            for k in range(self.N):
                self.rc_rect[k].disconnect_events()
                self.lc_rect[k].disconnect_events()
        if type == "rect":
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
        elif type == "span":
            self.lc_rect = [
                SpanSelector(
                    self.ax[k],
                    self.onLeftSelectFunc(k),
                    "horizontal",
                    button=[1],
                    useblit=True,
                )
                for k in range(self.N)
            ]
            self.rc_rect = [
                SpanSelector(
                    self.ax[k],
                    self.onRightSelectFunc(k),
                    "horizontal",
                    button=[3],
                    useblit=True,
                )
                for k in range(self.N)
            ]
        if self.parent:
            self.parent.current_slc_entry.set(SELECTOR_DICT[self.current_slc])

    def assignAndCloseCB(self, val=None):
        self.confirmed = True
        plt.close(self.ax[0].figure)

    def discardAndCloseCB(self, val=None):
        plt.close(self.ax[0].figure)

    def keyPressEvents(self, event):
        if event.key == ASSIGN_SHORTCUT:
            if self.parent is None:
                self.assignAndCloseCB()
            else:
                self.parent.assignAndQuitFunc(self)
        if event.key == DISCARD_SHORTCUT:
            if self.parent is None:
                self.discardAndCloseCB()
            else:
                self.parent.assignAndQuitFunc(None)

        elif event.key == "shift":
            if self.current_slc == "rect":
                self.current_slc = "span"
                self.spawn_selector("span")

            elif self.current_slc == "span":
                self.current_slc = "rect"
                self.spawn_selector("rect")

    def buttonPressEvents(self, event):
        print(event)
