from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT

FULLY_CONNECTED_SIZE = 16


class Connect2Model(L.LightningModule):
    def __init__(self, board_size, action_size):
        super().__init__()
        self.size = board_size
        self.action_size = action_size

        self.fully_connected_1 = nn.Linear(
            in_features=self.size, out_features=FULLY_CONNECTED_SIZE
        )
        self.fully_connected_2 = nn.Linear(
            in_features=FULLY_CONNECTED_SIZE, out_features=FULLY_CONNECTED_SIZE
        )
        # "cross entropy loss" to train from MCTS

        self.action_head = nn.Linear(
            in_features=FULLY_CONNECTED_SIZE, out_features=self.action_size
        )
        self.value_head = nn.Linear(in_features=FULLY_CONNECTED_SIZE, out_features=1)

        self.get_policy_loss = nn.CrossEntropyLoss()
        self.get_value_loss = nn.MSELoss()

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
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
        policy_loss = self.get_policy_loss(target_policy, pred_policy)
        value_loss = self.get_value_loss(target_value, pred_output)
        loss = policy_loss + value_loss
        return policy_loss, value_loss, loss

    # - for output - #

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            policy, value = self.forward(board)

        return policy.data.cpu().numpy()[0], value.data.cpu().numpy()[0]

    def select_action(self, board, game):
        policy, _ = self.predict(board)

        legal_moves = game.get_valid_moves(board)
        policy = policy * legal_moves
        policy = policy / np.sum(policy)

        choice = np.random.choice(len(policy), p=policy)
        return choice
