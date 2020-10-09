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

    @pytest.mark.parametrize('push_method', push_methods)
    def test_init(self, push_method):
        storage = TensorStorage(5, 3, push_method)
        assert storage.is_empty and not storage.is_full
        assert storage.size == 0 and storage.capacity == 5

    @pytest.mark.parametrize('push_method', push_methods)
    def test_init_wrong_params(self, push_method):
        with pytest.raises(AssertionError):
            TensorStorage(0, 1, push_method)
        with pytest.raises(AssertionError):
            TensorStorage(1, 0, push_method)
        with pytest.raises(AssertionError):
            TensorStorage(0, 0, push_method)
        with pytest.raises(AssertionError):
            TensorStorage(3, 3, 'wrong_push_method')

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

    def test_push_method_roll(self):
        """Test whether 'roll' push method inserts samples correctly."""
        sample_size = 3
        storage = TensorStorage(5, sample_size, push_method='roll')

        # Create pool of unique samples to be pushed into the storage.
        samples_pool = [self._filled_sample(sample_size, i) for i in range(storage.capacity * 3)]

        for pool_index in range(len(samples_pool)):
            # Add next sample from the pool to the storage.
            sample = samples_pool[pool_index]
            storage.push(sample)

            # Test whether order of samples in the storage is correct.
            for storage_index in range(storage.size):
                pool_sample = samples_pool[pool_index - storage_index]
                storage_sample = storage.get(storage_index)
                assert torch.equal(pool_sample, storage_sample)

    @pytest.mark.skip(reason='Test is not implemented yet.')
    def test_push_method_random(self):
        # todo It may work by comparing effects of changing the seed.
        assert False

    @pytest.mark.skip(reason='Test is not implemented yet.')
    def test_set_push_method(self):
        # todo Implement
        assert False

    @pytest.mark.parametrize('push_method', push_methods)
    def test_get(self, push_method):
        """Test indexing and shape of sample returned by get()."""
        sample_size = 3
        sample_shape = (1, sample_size)
        storage = TensorStorage(5, sample_size, push_method)

        for pushed_count in range(storage.capacity * 3):
            sample = self._filled_sample(sample_size, pushed_count)
            storage.push(sample)

            all_samples = storage.get_all()

            def _correct_sample_at_index(index):
                # Check whether get()'s indexing gives the same sample as torch's indexing.
                return torch.equal(storage.get(index), all_samples[index].unsqueeze(0))

            correct_indices = list(range(-storage.size, storage.size))

            wrong_indices = list(range(-storage.size * 3, -storage.size))
            wrong_indices += list(range(storage.size, storage.size * 3))

            for index in correct_indices:
                assert storage.get(index).shape == sample_shape
                assert _correct_sample_at_index(index)

            for index in wrong_indices:
                with pytest.raises(IndexError):
                    assert _correct_sample_at_index(index)

    @pytest.mark.parametrize('push_method', push_methods)
    def test_get_on_par_with_get_all(self, push_method):
        """Test whether elements returned by get() are on par with get_all()."""
        sample_size = 3
        storage = TensorStorage(5, sample_size, push_method)

        for pushed_samples in range(storage.capacity * 3):
            sample = self._filled_sample(sample_size, pushed_samples)
            storage.push(sample)

            reconstructed_batch = torch.cat([storage.get(i) for i in range(storage.size)])
            assert torch.equal(reconstructed_batch, storage.get_all())

    @pytest.mark.parametrize('push_method', push_methods)
    def test_get_empty_storage(self, push_method):
        """Test get() and get_all() behaviour on empty storage."""
        sample_size = 3
        storage = TensorStorage(5, sample_size, push_method)

        # It should not be possible to take samples from empty storage.
        with pytest.raises(AssertionError):
            storage.get(0)
        with pytest.raises(AssertionError):
            storage.get_all()

        sample = self._filled_sample(sample_size, 2)
        storage.push(sample)

        assert torch.equal(sample, storage.get(0))
        assert torch.equal(sample, storage.get_all())

    @staticmethod
    def _random_sample(size: int):
        return torch.randn(1, size)

    @staticmethod
    def _filled_sample(size: int, value: float):
        """Creates sample filled with elements equal to value."""
        return torch.ones(1, size).mul_(value)
