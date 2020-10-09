from abc import ABC
from abc import abstractmethod
import random

import torch


class SampleStorage(ABC):
    """Provides convenient way to store samples."""

    @abstractmethod
    def push(self, sample):
        """Adds sample to the storage."""
        raise NotImplementedError

    @abstractmethod
    def get(self, index):
        """Returns sample at index."""
        raise NotImplementedError


class TensorStorage(SampleStorage):
    # todo Write tests.
    """Stores samples as a batch of torch tensors."""

    def __init__(
            self, capacity: int, sample_size: int,
            push_method: str = 'roll', dtype=torch.float32):
        """
        Args:
            capacity: Max number of samples that can be stored.
            sample_size: Number of elements in a single sample.
            dtype: Data type of elements in a sample.
            push_method: Specifies method used to insert new samples.
        """
        assert capacity > 0 and sample_size > 0
        # Samples are internally kept as batch of flattened
        # tensors in a tensor with shape (batch size, sample size).
        self._data = torch.zeros(capacity, sample_size, dtype=dtype)

        self._capacity = capacity
        self._samples_count = 0
        self._sample_shape = (1, sample_size)

        self._push_methods = {'roll': self._push_method_roll, 'random': self._push_method_random}
        self._push_method_func = None
        self.set_push_method(push_method)

    @property
    def size(self):
        """Number of currently stored samples."""
        return self._samples_count

    @property
    def capacity(self):
        """Maximum number of samples that storage can hold."""
        return self._capacity

    @property
    def is_full(self):
        """True iff number of stored samples equals storage capacity."""
        return self._samples_count == self._capacity

    @property
    def is_empty(self):
        """True iff number of stored samples is 0."""
        return self._samples_count == 0

    def push(self, sample: torch.Tensor):
        """Adds sample to the storage using current push method."""
        assert sample.shape == self._sample_shape
        self._push_method_func(sample)

    def set_push_method(self, push_method: str):
        """Sets method used by push() to insert new samples.

        Args:
            push_method:
                'roll' - samples are organized by insertion order starting with most recent at 0.
                'random' - samples are inserted randomly."""
        assert push_method in self._push_methods
        self._push_method_func = self._push_methods[push_method]

    def get(self, index: int):
        """Returns sample at index % size."""
        assert not self.is_empty
        return self._data[index % self._samples_count]

    def get_all(self):
        """Returns batch of all stored samples."""
        assert not self.is_empty
        return self._data[:self._samples_count]

    def _insert(self, sample: torch.Tensor, index: int):
        """Inserts sample at index."""
        self._data[index, :] = sample[0, :]

    def _push_method_size(self, sample: torch.Tensor):
        """Inserts sample at index equal to the current storage size. Storage cannot be full."""
        # This push method is for internal use only.
        assert not self.is_full
        self._insert(sample, self._samples_count)
        self._samples_count += 1

    def _push_method_roll(self, sample: torch.Tensor):
        """Inserts samples so that they are organized by
        insertion time and the most recent is at index 0."""
        self._data = self._data.roll(1, 0)
        self._insert(sample, 0)

        if not self.is_full:
            self._samples_count += 1

    def _push_method_random(self, sample: torch.Tensor):
        """If storage is full, inserts sample at random index.
        Otherwise inserts sample at index equal to the current size."""
        if self.is_full:
            index = random.randrange(0, self._capacity)
            self._insert(sample, index)
        else:
            self._push_method_size(sample)
