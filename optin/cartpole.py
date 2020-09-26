from typing import Optional, Tuple, List
import random

import torch
import torch.nn as nn

from optin.predictor.simple_predictor import SimplePredictor


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


class AgentMind:
    Tensor = torch.Tensor

    def __init__(self, observation_size: int):
        self.observation_size = observation_size
        self.action_size = 1
        self.reward_size = 1

        self._last_action = None
        self._last_observation = None

        self.randomness = 3.  # Randomness in the performed action
        self.randomness_decay = 0.999

        self._reward_predictor = SimplePredictor(
            observation_size, self.action_size, 1, self.reward_size)
        # Remembers samples leading to reward = 1
        self._memory_positive = RandomBatchStorage(5, (1, observation_size + 2))
        # Remembers samples leading to reward = 0
        self._memory_negative = RandomBatchStorage(5, (1, observation_size + 2))

    def react(self, observation: List[float]):
        """Returns action based on [observation] from environment."""
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        action = torch.zeros(1, 1, requires_grad=True, dtype=torch.float32)
        # Final transformation before returning action to the environment
        action_transform = nn.Sigmoid()

        optim = torch.optim.SGD([action], 0.4)

        for optimizer_step in range(20):
            optim.zero_grad()
            prediction = self._reward_predictor(observation, action)
            loss = torch.abs(torch.ones(1, 1) - prediction).square()
            loss.backward()
            optim.step()

        action.requires_grad_(False)  # todo Does is change anything?
        action += torch.zeros(1, 1).uniform_(-1, 1) * self.randomness
        self.randomness *= self.randomness_decay
        self._remember(observation, action)
        action = action_transform(action).item()
        print('action: %.2f' % action, _slider(action), end=' ')
        return self._probability_choice(1, 0, action)

    def get_reward(self, reward: float):
        """Receives [reward] for the last performed action."""
        reward = torch.tensor(reward, dtype=torch.float32).view(1, 1)

        # todo RT
        with torch.no_grad():
            prediction = self._reward_predictor(self._last_observation, self._last_action)
            print('prediction:', _slider(prediction.item()))

        self._save_in_memory(self._last_observation, self._last_action, reward)
        self._learn()

    def _learn(self):
        if self._memory_positive.is_ready:
            memory_data = self._memory_positive.get_data()
            unpacked_data = self._unpack_memory_data(memory_data)
            self._reward_predictor.learn(*unpacked_data)

        if self._memory_negative.is_ready:
            memory_data = self._memory_negative.get_data()
            unpacked_data = self._unpack_memory_data(memory_data)
            self._reward_predictor.learn(*unpacked_data)

    def _remember(self, last_observation: Tensor, last_action: Tensor):
        # todo Change name
        """Keeps last received observation and performed action."""
        self._last_action = last_action
        self._last_observation = last_observation
        pass

    def _save_in_memory(self, observation: Tensor, action: Tensor, reward: Tensor):
        """Saves data in agent's working memory."""
        memory_sample = torch.cat((observation, action, reward), dim=1)

        if reward.item() > 0.5:
            self._memory_positive.push_sample(memory_sample)
        else:
            self._memory_negative.push_sample(memory_sample)

    def _unpack_memory_data(self, memory_data: Tensor):
        """Splits [memory_data] into its subcomponents."""
        chunks = (self.observation_size, self.action_size, self.reward_size)
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

    for i_episode in range(20000):
        observation = env.reset()

        for t in range(1000):
            env.render()

            action = agent.react(observation)

            observation, _, done, info = env.step(action)
            reward = 1.0 if not done else 0.0

            agent.get_reward(reward)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env.close()
