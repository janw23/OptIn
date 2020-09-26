from typing import Optional

import torch
import torch.nn as nn

from optin.utils import not_none


class SimplePredictor(nn.Module):
    """Tries to make a prediction based on state and action."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int, prediction_size: int):
        super().__init__()

        # Misc
        self.state_size = state_size
        self.action_size = action_size
        self.prediction_size = prediction_size

        # Model layers
        self.linear1 = nn.Linear(state_size + action_size, hidden_size)
        self.activation1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, prediction_size)
        self.activation2 = nn.Sigmoid()

        # Optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=0.005)
        self.loss_fn = nn.MSELoss()

    def correct_shapes(
            self, state: Optional[torch.Tensor] = None,
            action: Optional[torch.Tensor] = None,
            prediction: Optional[torch.Tensor] = None
    ):
        """Checks whether given inputs have correct shapes."""
        defined = not_none((state, action, prediction))
        if len(defined) == 0:
            return True

        batch_size = defined[0].shape[0]
        return (state is None or state.shape == (batch_size, self.state_size)) \
               and (action is None or action.shape == (batch_size, self.action_size)) \
               and (prediction is None or prediction.shape == (batch_size, self.prediction_size))

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Makes a prediction based on [state] and [action]."""
        assert self.correct_shapes(state=state, action=action)

        x = torch.cat((state, action), dim=1)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)

        assert self.correct_shapes(prediction=x)
        return x

    def learn(self, state: torch.Tensor, action: torch.Tensor, correct_prediction: torch.Tensor):
        """Tries to learn from example of [state] and [action] resulting in [correct_prediction]."""
        assert self.correct_shapes(state=state, action=action, prediction=correct_prediction)

        self.optim.zero_grad()

        prediction = self(state, action)
        assert prediction.shape == correct_prediction.shape
        loss = self.loss_fn(prediction, correct_prediction)
        loss.backward()

        self.optim.step()
