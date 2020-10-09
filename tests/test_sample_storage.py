import random

import pytest
import torch
import numpy as np

from optin.sample_storage import TensorStorage


def _set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TestTensorStorage:
    push_methods = ['roll', 'random']

    def test_init(self):
        storage = TensorStorage(5, 3)
        assert storage.is_empty and not storage.is_full
        assert storage.size == 0 and storage.capacity == 5

    @pytest.mark.parametrize('push_method', push_methods)
    def test_push(self, push_method):
        """General push() test that should be passed for all push methods."""
        _set_seeds()

        sample_size = 3
        storage = TensorStorage(5, sample_size, push_method)

        # Adding samples over the capacity should not fail.
        # Adding samples of wrong size should fail.
        for samples_added in range(storage.capacity * 3):
            assert storage.size == min(samples_added, storage.capacity)
            assert storage.is_full == (samples_added >= storage.capacity)

            sample = self._random_sample(sample_size)
            storage.push(sample)

            for wrong_size in range(sample_size * 3):
                if wrong_size != sample_size:
                    sample = self._random_sample(wrong_size)

                    with pytest.raises(AssertionError):
                        storage.push(sample)

    @staticmethod
    def _random_sample(size: int):
        return torch.randn(1, size)
