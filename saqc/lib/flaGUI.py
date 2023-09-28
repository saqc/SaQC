import numpy as np

from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector
import pandas as pd


class SelectFromCollection:
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
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))
        self.fc[:, -1] = self.alpha_other
        self.rect = RectangleSelector(ax, self.onselect, use_data_coordinates=True)
        self.ind = []

    def onselect(self, eclick, erelease):
        path = Path(np.array(
            [[eclick.xdata, eclick.ydata], [eclick.xdata, erelease.ydata], [erelease.xdata, erelease.ydata],
             [erelease.xdata, eclick.ydata]]))

        #path = Path(np.array([[eclick.ydata, eclick.xdata],[eclick.ydata,erelease.xdata],[erelease.ydata,erelease.xdata], [erelease.ydata,eclick.xdata]]))
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        #self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc[self.ind,:])
        self.canvas.draw_idle()

    def disconnect(self):
        self.rect.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    data_size = 1000
    #pts = ax
    data = pd.Series(np.sin(np.linspace(0,100, data_size)), index=pd.date_range('2000', periods=data_size,freq='10min'))
    ax.plot(data)
    pts = ax.scatter(data.index, data.values)
    ax.set_xlim(auto=True)
    selector = SelectFromCollection(ax, pts)
    plt.show()
    selector.disconnect()

    # After figure is closed print the coordinates of the selected points
    print('\nSelected points:')
    print(selector.xys[selector.ind])