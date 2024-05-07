import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT

from connect2.connect2game import Connect2Game

FULLY_CONNECTED_SIZE = 16


class Connect2Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.game_cls = Connect2Game

        self.fully_connected_1 = nn.Linear(
            in_features=self.game_cls.size, out_features=FULLY_CONNECTED_SIZE
        )
        self.fully_connected_2 = nn.Linear(
            in_features=FULLY_CONNECTED_SIZE, out_features=FULLY_CONNECTED_SIZE
        )

        self.action_head = nn.Linear(
            in_features=FULLY_CONNECTED_SIZE, out_features=self.game_cls.action_size
        )
        self.value_head = nn.Linear(in_features=FULLY_CONNECTED_SIZE, out_features=1)

        self.get_policy_loss = nn.CrossEntropyLoss()
        self.get_value_loss = nn.MSELoss()

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == self.game_cls.size
        x = func.relu(self.fully_connected_1(x))
        x = func.relu(self.fully_connected_2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        policy = func.softmax(action_logits, dim=1)
        value = torch.tanh(value_logit)

        return policy, value

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        policy_loss, value_loss, loss = self._step(batch, batch_idx)
        self.log("train/loss", loss)
        self.log("train/policy_loss", policy_loss)
        self.log("train/value_loss", value_loss)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        policy_loss, value_loss, loss = self._step(batch, batch_idx)
        self.log("val/loss", loss)
        self.log("val/policy_loss", policy_loss)
        self.log("val/value_loss", value_loss)
        return loss

    def _step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, target_policy, target_value = batch
        pred_policy, pred_output = self(inputs)
        policy_loss = self.get_policy_loss(pred_policy, target_policy)
        value_loss = self.get_value_loss(pred_output, target_value.view(-1, 1))
        loss = policy_loss + value_loss
        return policy_loss, value_loss, loss

    # - for output - #

    def predict(self, state):
        state = (
            torch.FloatTensor(state.astype(np.float32))
            .to(self.device)
            .view(1, self.game_cls.size)
        )
        self.eval()
        with torch.no_grad():
            policy, value = self.forward(state)

        return policy.data.cpu().numpy()[0], value.data.cpu().numpy()[0]
