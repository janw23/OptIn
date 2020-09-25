# This file contains my very first experiment.
import time
import random

import torch
import torch.nn as nn

from optin.utils import clip, lerp, clear_terminal


class AgentEnvironment:

    def __init__(self):
        self.agent_pos = 0.0
        self.agent_target_pos = 0.0

    def control_agent(self, agent_input: float):
        self.agent_pos += clip(agent_input, -1.0, 1.0)
        self.agent_pos = clip(self.agent_pos, -1., 1.)

    def draw_state(self):
        def _draw_pos(pos: float):
            pos = (pos + 1) / 2  # from (-1, 1) to (0, 1)
            pos = lerp(pos, 1, 30)
            pos = round(pos)
            drawing = ['o'] * 31
            drawing[pos] = 'O'
            return ''.join(drawing)

        print('target:', _draw_pos(self.agent_target_pos))
        print('actual:', _draw_pos(self.agent_pos))

    def get_agent_state(self):
        return (self.agent_pos, self.agent_target_pos)


class SimpleStatePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(2, 5)
        self.linear2 = nn.Linear(5, 5)
        self.linear3 = nn.Linear(5, 1)

        self.relu = nn.ReLU()

        self.optim = torch.optim.SGD(self.parameters(), 0.001)

    def forward(self, last_state: torch.Tensor, action: torch.Tensor):
        x = torch.stack((last_state, action), dim=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


    def optimize(
            self, last_state: torch.Tensor, last_action: torch.Tensor,
            next_state: torch.Tensor
    ):
        self.optim.zero_grad()
        predicted_state = self(last_state, last_action)
        loss = torch.abs(next_state - predicted_state).square()
        loss.backward()
        self.optim.step()

        print('state predictor loss:', loss.item())


class SimpleAgent:

    def __init__(self, env: AgentEnvironment):
        self.env = env
        self.state_predictor = SimpleStatePredictor()

        self.prev_state = torch.zeros(1)
        self.prev_action = torch.zeros(1)

    def make_action(self):
        pos, target_pos = env.get_agent_state()

        state = torch.tensor(pos)
        target_state = torch.tensor(target_pos)

        self.state_predictor.optimize(self.prev_state, self.prev_action, state)

        action = torch.tensor(0.0, requires_grad=True)
        optim = torch.optim.Adam([action], 0.1)

        for optimizer_step in range(4):
            predicted_next_state = self.state_predictor(state, action)

            loss = torch.abs(target_state - predicted_next_state)

            loss.backward()
            optim.step()
            optim.zero_grad()

        self.prev_state = state
        self.prev_action = action

        return action.item()

    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor):
        # Extremely simple predictor
        next_state = state + action
        return next_state


if __name__ == '__main__':
    env = AgentEnvironment()
    agent = SimpleAgent(env)

    for i in range(10000):
        clear_terminal()

        action = agent.make_action()
        env.control_agent(action)

        env.draw_state()
        print('action', action)

        if i % 10 == 0:
            env.agent_target_pos = random.random() * 2 - 1.0

        time.sleep(0.2)
