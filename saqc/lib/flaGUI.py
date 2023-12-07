import numpy as np

from matplotlib.pyplot import close
from matplotlib.widgets import RectangleSelector, Button, TextBox

FLAG_SHORTCUT = 'ö'
UNDO_SHORTCUT = 'z'
REMOVE_SHORTCUT = 'r'


class FlaGUI:
    """
    Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    gui_axes : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, gui_axes, collection, index, alpha_other=0, flag_val=255., label_val=''):
        self.canvas = gui_axes['plot'].figure.canvas
        self.canvas.mpl_connect('key_press_event', self.keyPressEvents)
        self.collection = collection
        self.alpha_other = alpha_other
        self.marker_handles = {}
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        self.index = index

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a face color')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.fc[:, -1] = self.alpha_other
        self.collection.set_facecolors(self.fc)
        self.rect = RectangleSelector(gui_axes['plot'], self.onselect, use_data_coordinates=True)
        self.marked = np.zeros(len(index)).astype(bool)
        self.selection = np.zeros(len(index)).astype(int)
        self.ax = gui_axes
        self.confirmed = False


        # Flagging Button Definition:
        self.flagButton = Button(self.ax['flag_button'], f'Add Selection \n ({FLAG_SHORTCUT})')
        self.flagButton.on_clicked(self.flagButtonCB)
        self.select_count = 0

        self.undoButton = Button(self.ax['undo_button'], f'undo \n ({UNDO_SHORTCUT})')
        self.undoButton.on_clicked(self.undoButtonCB)

        self.removeButton = Button(self.ax['remove_button'], f'remove \n ({REMOVE_SHORTCUT})')
        self.removeButton.on_clicked(self.removeButtonCB)

        self.assignAndClose = Button(self.ax['assign_button'], f'assign selection')
        self.assignAndClose.on_clicked(self.assignAndCloseCB)

        self.flagValueBox = TextBox(self.ax['flag_box'], 'Flagging Level', initial=flag_val)
        self.labelValueBox = TextBox(self.ax['label_box'], 'Label', initial=label_val)
        self.canvas.draw_idle()

    def onselect(self, eclick, erelease):
        upper_left = (min(eclick.xdata, erelease.xdata), max(eclick.ydata, erelease.ydata))
        bottom_right = (max(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata))
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
        # transfer marks to selection and draw the "selected" marker to the selected positions
        add_mask = self.marked & (self.selection==0)
        if not add_mask.any():
            return
        self.select_count += 1
        self.selection[add_mask] = self.select_count
        self.drawSelection(s_index=self.select_count)
        self.marked[:]=0
        self.canvas.draw_idle()

    def undoButtonCB(self, val):
        if self.select_count==0:
            return
        undo_mask = self.selection == self.select_count
        if undo_mask.sum() > 0:
            self.selection[undo_mask]=0
            self.marker_handles[self.select_count].remove()
            self.marker_handles.pop(self.select_count)
            self.select_count -= 1
            self.marked = undo_mask
            self.fc[:,-1] = 0
            self.fc[undo_mask, -1] =1
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
        if self.labelValueBox.text == '':
            self.labelValueBox.text = None
        if self.flagValueBox.text == 'UNFLAGGED':
            self.flagValueBox.text = -np.inf
        close(self.ax['plot'].figure)

    def keyPressEvents(self, event):
        if event.key==FLAG_SHORTCUT:
            self.flagButtonCB(event.key)
        if event.key==UNDO_SHORTCUT:
            self.undoButtonCB(event.key)
        if event.key==REMOVE_SHORTCUT:
            self.removeButtonCB(event.key)

    def drawSelection(self, s_index=None):

        if s_index is None:
            draw_mask = self.selection > 0
        else:
            draw_mask = self.selection == s_index
        dates = self.index[draw_mask]
        xl = self.ax['plot'].get_xlim()
        handle = self.ax['plot'].scatter(x=dates, y=np.array(self.xys[draw_mask][:, 1]), marker='X', c='red', s=100)
        if s_index is None:
            self.marker_handles.update({self.select_count:handle})
        else:
            self.marker_handles.update({s_index:handle})
        self.ax['plot'].set_xlim(xl)


