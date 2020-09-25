# This file contains my very first experiment.
import time
import random

import torch

from optin.utils import clip, clear_terminal


class AgentEnvironment:

    def __init__(self):
        self.agent_pos = 0.0
        self.agent_target_pos = 0.0

    def control_agent(self, agent_input: float):
        self.agent_pos += clip(agent_input, -1.0, 1.0)

    def draw_state(self):
        def _draw_pos(pos: float):
            pos = pos * 5 + 5  # from (-1, 1) to (0, 10)
            pos = round(pos)
            assert 0 <= pos <= 10
            drawing = ['o'] * 11
            drawing[pos] = 'O'
            return ''.join(drawing)

        print('target:', _draw_pos(self.agent_target_pos))
        print('actual:', _draw_pos(self.agent_pos))

    def get_agent_state(self):
        return (self.agent_pos, self.agent_target_pos)


class SimpleAgent:

    def __init__(self, env: AgentEnvironment):
        self.env = env

    def make_action(self):
        pos, target_pos = env.get_agent_state()

        state = torch.tensor(pos)
        target_state = torch.tensor(target_pos)

        action = torch.tensor(0.0, requires_grad=True)
        optim = torch.optim.Adam([action], 0.1)

        for optimizer_step in range(1):
            predicted_next_state = self.predict_next_state(state, action)

            loss = torch.abs(target_state - predicted_next_state)

            loss.backward()
            optim.step()
            optim.zero_grad()

        return action.item()

    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor):
        # Extremely simple predictor
        next_state = state + action
        return next_state


if __name__ == '__main__':
    env = AgentEnvironment()
    agent = SimpleAgent(env)

    for i in range(100):
        action = agent.make_action()
        env.control_agent(action)

        clear_terminal()
        env.draw_state()
        print('action', action)

        if i % 10 == 0:
            env.agent_target_pos = random.random() * 2 - 1.0

        time.sleep(0.2)
