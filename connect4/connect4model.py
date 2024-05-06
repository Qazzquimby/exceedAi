import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

from monte_carlo_tree_search import MCTS

FULLY_CONNECTED_SIZE = 16


class Connect4Model(nn.Module):
    def __init__(self, board_size, action_size, device):
        super().__init__()

        self.device = device
        self.size = board_size[0] * board_size[1]
        self.action_size = action_size

        self.fully_connected_1 = nn.Linear(
            in_features=self.size, out_features=FULLY_CONNECTED_SIZE
        )
        self.fully_connected_2 = nn.Linear(
            in_features=FULLY_CONNECTED_SIZE, out_features=FULLY_CONNECTED_SIZE
        )

        self.action_head = nn.Linear(
            in_features=FULLY_CONNECTED_SIZE, out_features=self.action_size
        )
        self.value_head = nn.Linear(in_features=FULLY_CONNECTED_SIZE, out_features=1)

        self.to(device)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x = func.relu(self.fully_connected_1(x))
        x = func.relu(self.fully_connected_2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        policy = func.softmax(action_logits, dim=1)
        value = torch.tanh(value_logit)

        return policy, value

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            policy, value = self.forward(board)

        return policy.data.cpu().numpy()[0], value.data.cpu().numpy()[0]

    def select_action(self, board, game):
        # TODO duplicate
        policy, _ = self.predict(board)

        legal_moves = game.get_valid_moves(board)
        policy = policy * legal_moves
        policy = policy / np.sum(policy)

        choice = np.random.choice(len(policy), p=policy)
        return choice

    def select_action_from_sim(self, board_for_player, game, args):
        self.mcts = MCTS(game=game, model=self, args=args)
        root = self.mcts.run(model=self, state=board_for_player, to_play=1)

        action = root.select_action(temperature=0)
        return action
