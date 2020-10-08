from typing import Callable

from matplotlib import pyplot as plt


class LivePlot:
    """Wrap which makes real-time plotting easy."""

    def __init__(self, update_func: Callable):
        """Creates new LivePlot.

        Args:
            update_func: Function used to redraw the plot.
        """
        self._update_func = update_func

    def redraw(self, *new_args, pause_time=0.01):
        """Redraws the plot with the new data.

        Args:
            new_args: Argument(s) for update_func to update the plot.
            pause_time: Interval used by plt.pause().
        """
        self._update_func(*new_args)
        plt.pause(pause_time)
