import abc
from pathlib import Path

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

checkpoints_dir = Path("checkpoints")


class Game(abc.ABC):
    def __init__(self):
        pass

    def get_init_board(self):
        raise NotImplementedError

    def get_action_size(self):
        raise NotImplementedError

    def get_next_state(self, board, player, action):
        raise NotImplementedError

    def has_legal_moves(self, board):
        raise NotImplementedError

    def get_valid_moves(self, board):
        raise NotImplementedError

    def mask_invalid_moves(self, state, action_probs):
        valid_moves = self.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        return action_probs

    def is_win(self, board: np.ndarray, player: int):
        raise NotImplementedError

    def get_reward_for_player(self, board, player):
        # return None if not ended, 1 if player the given player 1, -1 if they lost

        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return None

        return 0

    def get_board_from_perspective(self, board, player):
        raise NotImplementedError

    def auto_play(self, player1, player2):
        num_games = 5

        for game_num in range(num_games):
            self.auto_play_game(player1, player2)

    def auto_play_game(self, player1, player2):
        state = self.get_init_board()
        current_player = 1

        turn = 0
        while True:
            board_for_player = self.get_board_from_perspective(state, current_player)

            if current_player == 1:
                action = player1.select_action(board_for_player, game=self)
            else:
                action = player2.select_action(board_for_player, game=self)

            state, next_player = self.get_next_state(state, current_player, action)
            print(f"turn {turn}: {state}")

            reward = self.get_reward_for_player(state, current_player)

            if reward is not None:
                if reward == 1:
                    print(f"Player {current_player} wins")
                elif reward == -1:
                    print(f"Player {current_player*-1} wins")
                else:
                    print("Draw")
                return

            current_player = next_player
            turn += 1
