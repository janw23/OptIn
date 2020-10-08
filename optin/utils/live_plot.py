from typing import Callable, Tuple

from matplotlib import pyplot as plt
import numpy as np


class LivePlot:
    """Wrap which makes real-time plotting easy."""

    def __init__(self, update_func: Callable):
        """
        Args:
            update_func: Function used to update the plot before redrawing.
        """
        self._update_func = update_func

    def redraw(self, *new_args, pause_time=0.001):
        """Redraws the plot with the new data.

        Args:
            new_args: Argument(s) for update_func to update the plot.
            pause_time: Interval used by plt.pause().
        """
        self._update_func(*new_args)
        plt.pause(pause_time)


class LinspaceLivePlot(LivePlot):
    """Plots values over evenly-spaced arguments from x-axis."""

    def __init__(self, interval=(-1.0, 1.0), y_lim=(0, 1)):
        """
        Args:
            interval (Tuple[float, float]): Interval from which arguments are sampled.
            y_lim (Tuple[float, float]):  Defines visible slice of y-axis.
        """
        _, self._ax = plt.subplots(1, 1)

        self._ax.set_ylim(*y_lim)
        self._update_x_axis(interval)
        self._points = self._ax.plot(self._x_axis, self._x_axis)[0]

        def _update_func(y: np.ndarray):
            # Updates plot before redrawing.
            if y.size != self._steps_count:
                self._update_x_axis(self._interval, y.size)

            self._points.set_data(self._x_axis, y)

        super().__init__(_update_func)

    def _update_x_axis(self, interval: Tuple[float, float], steps_count: int = 50):
        """Regenerates array of evenly-spaced samples from x-axis."""
        self._steps_count = steps_count
        self._interval = interval
        self._ax.set_xlim(*interval)
        self._x_axis = np.linspace(*interval, steps_count)
