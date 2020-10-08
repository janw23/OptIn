from typing import Optional, Tuple, List
import random

import torch
import torch.nn as nn

from optin.predictor.simple_predictor import SimplePredictor
from optin.utils import Average


class RandomBatchStorage:
    """Keeps batch of specified number of samples
    and replaces old with new ones randomly."""

    def __init__(self, batch_size: int, sample_shape: Tuple[int, ...]):
        # batch_size - number of samples in the batch
        # sample_shape - shape of one sample where first dimension is equal to 1
        #                because first dimension means batch size
        self.batch_size = batch_size
        self.sample_shape = sample_shape
        assert sample_shape[0] == 1

        self._batch_data = torch.zeros(batch_size, *sample_shape[1:], dtype=torch.float32)
        self._kept_samples_count = 0

    @property
    def is_ready(self):
        """Checks whether storage is ready to be used."""
        return self._kept_samples_count == self.batch_size

    def push_sample(self, sample: torch.Tensor):
        """Adds sample to the storage."""
        assert sample.shape == self.sample_shape

        if not self.is_ready:
            # Add samples until the storage is full.
            self._batch_data[self._kept_samples_count, :] = sample[0, :]
            self._kept_samples_count += 1
        else:
            # Replace random old sample with the new one.
            batch_index = random.randrange(0, self.batch_size)
            self._batch_data[batch_index, :] = sample[0, :]

    def get_data(self):
        """Returns batch of kept samples."""
        assert self.is_ready
        return self._batch_data


class SequentialStorage:
    # todo Unify with RandomBatchStorage
    """Keeps specified number of previous states."""

    def __init__(self, storage_size: int, sample_shape: Tuple[int, ...]):
        self.storage_size = storage_size  # Number of samples to keep
        self.sample_shape = sample_shape
        assert sample_shape[0] == 1

        self._kept_samples_count = 0
        self._storage_data = torch.zeros(storage_size, *sample_shape[1:], dtype=torch.float32)

    @property
    def is_ready(self):
        """Checks whether storage is ready to be used."""
        return self._kept_samples_count == self.storage_size

    def push_sample(self, sample: torch.Tensor):
        """Adds sample to the storage."""
        assert sample.shape == self.sample_shape

        # Replace the oldest sample
        self._storage_data[0, :] = sample[0, :]
        # Shift all samples so that the new one is the most recent
        self._storage_data = self._storage_data.roll(-1, 0)

        if not self.is_ready:
            self._kept_samples_count += 1

    def get_data(self):
        assert self.is_ready
        return self._storage_data


class AgentMind:
    Tensor = torch.Tensor

    def __init__(self, observation_size: int):
        self.observation_size = observation_size
        self.action_size = 1
        self.reward_size = 1
        self.hidden_size = 10

        self._last_action = None
        self._last_observation = None

        self._last_observations_count = 10  # Number of last observations to use in predictor
        self._prediction_future_steps = 4  # Number of steps prediction is in the future
        self._last_observations_size = observation_size * self._last_observations_count

        self._reward_predictor = SimplePredictor(
            self._last_observations_size,
            self.action_size, self.hidden_size, self.reward_size)

        self._memory_chunks = (observation_size, self.action_size, self.reward_size)
        self._memory_sequence_chunks = \
            (self._last_observations_size, self.action_size, self.reward_size)

        # Sample form (observation, action, reward)
        # Remembers sequence of last states
        self._memory_last_states = SequentialStorage(
            storage_size=self._last_observations_count + self._prediction_future_steps,
            sample_shape=(1, sum(self._memory_chunks)))

        # Sample form: (*observations, action, reward)
        # Remembers state sequences leading to reward = 1
        self._memory_sequences_positive = RandomBatchStorage(
            batch_size=5, sample_shape=(1, sum(self._memory_sequence_chunks)))
        # Remembers state sequences leading to reward = 0
        self._memory_sequences_negative = RandomBatchStorage(
            batch_size=5, sample_shape=(1, sum(self._memory_sequence_chunks)))

    def react(self, observation: List[float]):
        """Returns action based on [observation] from environment."""
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        raw_action = self._determine_raw_action(observation)
        self._remember_last_state(observation, raw_action)

        action = self._transform_raw_action(raw_action)
        return action

    def _determine_raw_action(self, observation: Tensor):
        """Determines raw action based on [observation]."""
        action = torch.zeros(1, 1, requires_grad=True, dtype=torch.float32)
        optim = torch.optim.SGD([action], 0.4)

        if not self._memory_last_states.is_ready:
            return torch.zeros(1, 1)

        memory_data = self._memory_last_states.get_data()
        # Take n - 1 most recent last states in memory
        last_observations = memory_data[-(self._last_observations_count - 1):]
        # Extract observations from states
        last_observations = last_observations[:, :self.observation_size].reshape(1, -1)
        # Add the most recent state which has not yet been written to memory
        last_observations = torch.cat((last_observations, self._last_observation), dim=1)

        for optimizer_step in range(20):
            optim.zero_grad()
            prediction = self._reward_predictor(last_observations, action)
            loss = torch.abs(torch.ones(1, 1) - prediction).square()
            loss.backward()
            optim.step()

        print('prediction:', _slider(prediction.item()), end=' ')

        action.requires_grad_(False)
        return action

    def _transform_raw_action(self, action: Tensor) -> int:
        """Transforms raw [action] into its final
         form compatible with the environment."""
        # Sigmoid activation
        action = torch.sigmoid(action).item()
        print('action: %.2f' % action, _slider(action), end=' ')  # Debugging

        # Make action binary
        action = self._probability_choice(1, 0, action)
        return action

    def receive_reward(self, reward: float, learn: bool = True):
        """Receives [reward] for the last performed action."""
        reward = torch.tensor(reward, dtype=torch.float32).view(1, 1)
        self._save_in_memory(self._last_observation, self._last_action, reward)
        if learn:
            self._learn()

    def _learn(self):
        if self._memory_sequences_positive.is_ready:
            memory_data = self._memory_sequences_positive.get_data()
            unpacked_data = self._unpack_memory_data(memory_data, self._memory_sequence_chunks)
            self._reward_predictor.learn(*unpacked_data)

        if self._memory_sequences_negative.is_ready:
            memory_data = self._memory_sequences_negative.get_data()
            unpacked_data = self._unpack_memory_data(memory_data, self._memory_sequence_chunks)
            self._reward_predictor.learn(*unpacked_data)

    def _remember_last_state(self, last_observation: Tensor, last_action: Tensor):
        """Keeps last received observation and performed action."""
        self._last_action = last_action
        self._last_observation = last_observation

    def _save_in_memory(self, observation: Tensor, action: Tensor, reward: Tensor):
        """Saves data in agent's working memory."""
        memory_sample = torch.cat((observation, action, reward), dim=1)
        self._memory_last_states.push_sample(memory_sample)

        def _extract_sequence(memory_data: torch.Tensor, last_observations: int, future_steps: int):
            # Memory data is in the form of batch of samples:
            # (batch_dim (greater is more recent), contiguous: (observation, action, next_reward))
            # We want to extract contiguous sequence:
            # ([last_observations] *observations, action, reward received in [future_steps])
            observation_data, action_data, reward_data = \
                self._unpack_memory_data(memory_data, self._memory_chunks)

            observations = observation_data[:last_observations].reshape(1, -1)
            action = action_data[last_observations].view(1, -1)
            reward = reward_data[last_observations + future_steps - 1].view(1, -1)

            return torch.cat((observations, action, reward), dim=1)

        if self._memory_last_states.is_ready:
            memory_data = self._memory_last_states.get_data()
            memory_sequence_sample = _extract_sequence(
                memory_data, self._last_observations_count, self._prediction_future_steps)

            if memory_sequence_sample[0, -1].item() > 0.5:  # Reward is positive
                self._memory_sequences_positive.push_sample(memory_sequence_sample)
            else:  # Reward is negative
                self._memory_sequences_negative.push_sample(memory_sequence_sample)

    def _unpack_memory_data(self, memory_data: Tensor, chunks):
        """Splits [memory_data] into its subcomponents."""
        return torch.split(memory_data, chunks, dim=1)

    def _probability_choice(self, choice_a, choice_b, prob: float):
        """Returns [choice_a] with probability [prob]. Else returns [choice_b]"""
        assert 0 <= prob <= 1
        toss = random.random()
        return choice_a if toss <= prob else choice_b


def _slider(val: float, size: int = 20):
    assert 0 <= val <= 1
    pos = round(val * size)
    ar = [' '] * (size + 1)
    ar[pos] = 'O'
    ar = ['|'] + ar + ['|']
    return ''.join(ar)


if __name__ == '__main__':
    import gym

    env = gym.make('CartPole-v0')
    agent = AgentMind(4)

    avg_steps = Average(5)
    avg = 0

    for i_episode in range(20000):
        observation = env.reset()

        for t in range(1000):

            env.render()

            action = agent.react(observation)

            observation, _, done, info = env.step(action)
            reward = 1.0 if not done else 0.0

            agent.receive_reward(reward)
            print('average steps:', avg, _slider(avg / 1000))

            if done:
                avg = avg_steps.push(t)
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env.close()
