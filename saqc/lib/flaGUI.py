import numpy as np

from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector, Button
import pandas as pd


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

    def __init__(self, gui_axes, collection, index, alpha_other=0):
        self.canvas = gui_axes['plot'].figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        self.index = index

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.fc[:, -1] = self.alpha_other
        self.rect = RectangleSelector(gui_axes['plot'], self.onselect, use_data_coordinates=True)
        self.ind = np.zeros(len(index)).astype(bool)
        self.ax = gui_axes

        # Flagging Button Definition:
        self.flagButton = Button(self.ax['flag_button'], 'Assign Flags')
        self.flagButton.on_clicked(self.flagButtonCB)

    def onselect(self, eclick, erelease):
        #path = Path(np.array(
        #    [[eclick.xdata, eclick.ydata], [eclick.xdata, erelease.ydata], [erelease.xdata, erelease.ydata],
        #     [erelease.xdata, eclick.ydata]]))
        upper_left = (min(eclick.xdata, erelease.xdata), max(eclick.ydata, erelease.ydata))
        bottom_right = (max(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata))
        x_cut = (self.xys[:, 0] > upper_left[0]) & (self.xys[:, 0] < bottom_right[0])
        y_cut = (self.xys[:, 1] > bottom_right[1]) & (self.xys[:, 1] < upper_left[1])

        self.ind |= x_cut & y_cut

        #self.ind = np.nonzero(path.contains_points(self.xys))[0]

        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.rect.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def flagButtonCB(self, val):
        dates = self.index[self.ind.squeeze()]
        xl = self.ax['plot'].get_xlim()
        self.ax['plot'].scatter(x=dates, y=np.array(self.xys[self.ind.squeeze()][:, 1]), marker='X', c='b', s=100)
        self.ax['plot'].set_xlim(xl)
        self.canvas.draw()
        self.ind[:] = False
        self.fc[:, -1] = self.alpha_other


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    data_size = 1000
    #pts = ax
    data = pd.Series(np.sin(np.linspace(0,100, data_size)), index=pd.date_range('2000', periods=data_size,freq='10min'))
    ax.plot(data)
    pts = ax.scatter(data.index, data.values)
    ax.set_xlim(auto=True)
    selector = FlaGUI(ax, pts)
    plt.show()
    selector.disconnect()


    # After figure is closed print the coordinates of the selected points
    print('\nSelected points:')
    print(selector.xys[selector.ind])