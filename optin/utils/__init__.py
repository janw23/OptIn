from typing import Iterable


def clip(val: float, low: float, high: float):
    """Clips [val] between [low] and [high]."""
    return min(high, max(low, val))


def lerp(t: float, a: float, b: float):
    """Linearly interpolates between [a] and [b]
    according to interpolant [t] clipped in range [0, 1]."""
    return a + (b - a) * clip(t, 0.0, 1.0)


def clear_terminal():
    """Prints escape sequence which clears terminal."""
    print(chr(27) + "[2J")


def proximate(a: float, b: float, eps=1e-6):
    """Checks whether values are close to each other."""
    return a - eps <= b <= a + eps


def not_none(iterable_: Iterable):
    """Returns list of not-None objects from iterable."""
    return [obj for obj in iterable_ if obj is not None]


class Average:
    def __init__(self, last_steps: int):
        self._last_steps = last_steps
        self._data = [0] * last_steps
        self._index = 0
        self._sum = 0

    def push(self, val: float):
        self._sum += val
        self._sum -= self._data[self._index]
        self._data[self._index] = val
        self._index = (self._index + 1) % self._last_steps
        return self._sum / self._last_steps
