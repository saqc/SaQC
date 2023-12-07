#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.pyplot import close
from matplotlib.widgets import Button, RectangleSelector, TextBox

ADD_SHORTCUT = "a"
UNDO_SHORTCUT = "z"
REMOVE_SHORTCUT = "r"
ASSIGN_SHORTCUT = "enter"
SELECTION_MARKER_DEFAULT = {
    "marker": "X",
    "c": "red",
    "s": 100,
    "alpha": 1,
    "zorder": 100,
}


class FlaGUI:
    def __init__(
        self,
        gui_axes,
        collection,
        index,
        flag_val=255.0,
        label_val="",
        selection_marker_kwargs=SELECTION_MARKER_DEFAULT,
    ):
        self.canvas = gui_axes["plot"].figure.canvas
        self.canvas.mpl_connect("key_press_event", self.keyPressEvents)
        self.collection = collection
        self.marker_handles = {}
        self.xys = collection.get_offsets()
        self.index = index
        self.label = None
        self.flag = None
        self.fc = np.tile(collection.get_facecolors(), (len(self.xys), 1))
        self.fc[:, -1] = 0
        self.collection.set_facecolors(self.fc)
        self.rect = RectangleSelector(
            gui_axes["plot"], self.onselect, use_data_coordinates=True
        )
        self.marked = np.zeros(len(index)).astype(bool)
        self.selection = np.zeros(len(index)).astype(int)
        self.axes = gui_axes
        self.confirmed = False
        self.select_count = 0
        self.s_marker_kwargs = {**SELECTION_MARKER_DEFAULT, **selection_marker_kwargs}

        # Buttons and Text Boxe:
        self.flagButton = Button(self.axes["flag_button"], f"Add ({ADD_SHORTCUT})")
        self.flagButton.on_clicked(self.flagButtonCB)

        self.undoButton = Button(self.axes["undo_button"], f"Back ({UNDO_SHORTCUT})")
        self.undoButton.on_clicked(self.undoButtonCB)

        self.removeButton = Button(
            self.axes["remove_button"], f"Remove ({REMOVE_SHORTCUT})"
        )
        self.removeButton.on_clicked(self.removeButtonCB)

        self.assignAndClose = Button(
            self.axes["assign_button"], f"Assign ({ASSIGN_SHORTCUT})\n (ends session)"
        )
        self.assignAndClose.on_clicked(self.assignAndCloseCB)

        self.flagValueBox = TextBox(
            self.axes["flag_box"], "Flagging Level", initial=flag_val
        )
        self.labelValueBox = TextBox(
            self.axes["label_box"], "Flags Label", initial=label_val
        )
        self.canvas.draw_idle()

    def onselect(self, eclick, erelease):
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
        self.marked[:] = False
        self.marked[x_cut & y_cut] = True

        self.fc[:, -1] = 0
        self.fc[self.marked, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.rect.disconnect_events()

    def flagButtonCB(self, val):
        add_mask = self.marked & (self.selection == 0)
        if not add_mask.any():
            return
        self.select_count += 1
        self.selection[add_mask] = self.select_count
        self.drawSelection(s_index=self.select_count)
        self.marked[:] = 0
        self.canvas.draw_idle()

    def undoButtonCB(self, val):
        if self.select_count == 0:
            return
        undo_mask = self.selection == self.select_count
        if undo_mask.sum() > 0:
            self.selection[undo_mask] = 0
            self.marker_handles[self.select_count].remove()
            self.marker_handles.pop(self.select_count)
            self.select_count -= 1
            self.marked = undo_mask
            self.fc[:, -1] = 0
            self.fc[undo_mask, -1] = 1
            self.collection.set_facecolors(self.fc)
            self.canvas.draw_idle()

    def removeButtonCB(self, vals):
        to_redraw = set(self.selection[self.marked])
        self.selection[self.marked] = 0
        to_redraw.discard(0)
        for r in to_redraw:
            self.marker_handles[r].remove()
            self.marker_handles.pop(r)
            self.drawSelection(s_index=r)
        self.canvas.draw_idle()

    def assignAndCloseCB(self, vals):
        self.confirmed = True
        if self.labelValueBox.text == "":
            self.label = None
        else:
            self.label = self.labelValueBox.text
        if self.flagValueBox.text == "UNFLAGGED":
            self.flag = -np.inf
        else:
            self.flag = self.flagValueBox.text
        close(self.axes["plot"].figure)

    def keyPressEvents(self, event):
        if event.key == ADD_SHORTCUT:
            self.flagButtonCB(event.key)
        if event.key == UNDO_SHORTCUT:
            self.undoButtonCB(event.key)
        if event.key == REMOVE_SHORTCUT:
            self.removeButtonCB(event.key)
        if event.key == ASSIGN_SHORTCUT:
            self.assignAndCloseCB(event.key)

    def drawSelection(self, s_index=None):
        if s_index is None:
            draw_mask = self.selection > 0
        else:
            draw_mask = self.selection == s_index
        dates = self.index[draw_mask]
        xl = self.axes["plot"].get_xlim()
        handle = self.axes["plot"].scatter(
            x=dates, y=np.array(self.xys[draw_mask][:, 1]), **self.s_marker_kwargs
        )
        if s_index is None:
            self.marker_handles.update({self.select_count: handle})
        else:
            self.marker_handles.update({s_index: handle})
        self.axes["plot"].set_xlim(xl)
