from typing import Callable
import functools

import time

import matplotlib
from matplotlib import pyplot as plt
import numpy as np


class LivePlot:
    """Wrap which makes real-time plotting easy."""

    def __init__(self, create_func: Callable, update_func: Callable):
        """Creates new LivePlot.

        Args:
            create_func: Function used to initialize the plot. It should take no args.
            update_func: Function used to redraw the plot. As its first argument
                it should expect unpacked variable returned by create_func.
        """
        _plot_data = create_func()
        self._update_func = functools.partial(update_func, *_plot_data)

    def redraw(self, new_data, pause_time=0.01):
        """Redraws the plot with the new data.

        Args:
            new_data: Argument for update_func to update the plot.
            pause_time: Interval used by plt.pause().
        """
        self._update_func(new_data)
        plt.pause(pause_time)


if __name__ == '__main__':
    # Draw sinus changing in time

    def _make_plot():
        fig, ax = plt.subplots(1, 1)
        x = np.arange(100)
        points = ax.plot(x, np.sin(x))[0]
        return x, points

    def _redraw(x, points, t: float):
        points.set_data(x, np.sin(x + t))

    live = LivePlot(_make_plot, _redraw)

    for t in range(100):
        live.redraw(t)

